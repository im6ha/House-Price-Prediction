import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.special import softmax
from pingouin import multivariate_normality


"""
1)- Logistic Regression
"""

def logistic_function(observations:np.ndarray, parameters:np.ndarray)->np.ndarray:
    """
    Calculates the logistic function which is the Probability that the observation is 'expensive'
        observations: 2D array representing the observations, shape=(n, p)
        parameters: array of the parameters, with the first parameter representing the bias
    """
    
    linear_comb = np.dot(observations, parameters.T)

    return 1/( 1+np.exp(-linear_comb) )


def compute_cost(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the log-loss function using pre-calculated predictions
        predictions: array of the predictions
        targets: array of the corresponding targets
    """
    eps = 1e-15
    predictions=np.clip(predictions, eps, 1 - eps)
    cost = -np.mean(targets *np.log(predictions) +(1 -targets) *np.log(1 -predictions))
    return float(cost)


def gradient_descent(observations:np.ndarray, targets:np.ndarray, alpha:float, epsilon:float, initial_parameters:np.ndarray, max_iters:int)->tuple[np.ndarray, dict]:
    """
    Performs gradient descent and returns the parameters (first element is the bias) and a dictionary whose keys are iterations and values are costs (used for convergence)
        observations: Dataframe of all the observations, shape=(n, p)
        targets: array of the targets (of the observations)
        alpha: learning rate, specifies the size of the steps (smaller alpha -> slower convergence)
        epsilon: stopping condition, stop when the difference in cost is less than epsilon (smaller epsilon -> closer to the minimum)
        initial_parameters: initial values given to the parameters
        max_iters: maximum number of iterations for one value of the tuned parameter
    """
    #initialize all parameters
    parameters =initial_parameters.copy()
    iters_costs = {}
    n_samples =observations.shape[0]

    predictions = logistic_function(observations, parameters)
    current_cost = compute_cost(predictions, targets)

    for iteration in range(1, max_iters + 1):
        grad =(observations.T @ (predictions - targets)) / n_samples
        new_parameters=parameters - alpha * grad
        new_predictions= logistic_function(observations, new_parameters)
        new_cost =compute_cost(new_predictions, targets)
    
        cost_diff =current_cost -new_cost
        iters_costs[iteration] = new_cost
        if cost_diff <0:
            print(f"Warning: Divergence for alpha={alpha}")
            break
        if cost_diff< epsilon:
            print(f"Converged at iteration {iteration}.")
            break
        parameters = new_parameters
        predictions = new_predictions
        current_cost = new_cost
    if iteration == max_iters:
        print("Reached maximum iterations.")
    return parameters, iters_costs
            


"""
2)- KNN
"""

def K_nearest_neighbors(k:int, dataset:np.ndarray, targets:np.ndarray, point:np.ndarray)->dict:
    """
    Finds the K-Nearest Neighbors to Point from the Dataset, and returns a sorted dictionary with keys as the 4 classes and values as the proportion of each class
        k: the number of neighbors to consider, lower k means higher flexibility
        dataset: the dataset used to get the neighbors from (the training set)
        targets: the targets of the dataset
        point: the point whose target will be predicted
    """
    #euclidean distance
    distances = np.linalg.norm(dataset - point, axis=1)
    closest_indices = np.argpartition(distances, k)[:k]
    neighbors_targets = targets[closest_indices]
    result = dict()
    for target_class in ['cheap', 'moderate', 'expensive', 'very expensive']:
        result[target_class] = (neighbors_targets==target_class).sum() / k
    result = dict( sorted(result.items(), key=lambda x: x[1]) )
    return result

def KNN_prediction(k:int, dataset:np.ndarray, targets:np.ndarray, testing_dataset:np.ndarray)->np.ndarray:
    """
    Calculates the prediction of the testing dataset using KNN regression from the dataset and its targets, returns an array for the predictions
        k: the number of neighbors to consider, lower k means higher flexibility
        dataset: the dataset used to get the neighbors from (the training set)
        targets: the targets of the dataset
        testing_dataset: the points whose targets will be predicted
    """
    if isinstance(dataset, pd.DataFrame):
        dataset= dataset.to_numpy()
    if isinstance(testing_dataset, pd.DataFrame):
        testing_dataset= testing_dataset.to_numpy()

    predictions = []
    for row in testing_dataset:
        probs = K_nearest_neighbors(k, dataset, targets, row)
        t = max(probs, key=probs.get)
        predictions.append(t)
    return np.array(predictions)

def error_rate(targets:np.ndarray, predictions:np.ndarray):
    """
    Calculates the error rate (proportion of wrong predictions)
        targets: real values
        predictions: predicted values
    """
    return (targets != predictions).mean()

def KNN_classification_tuning(k_range:range, training_dataset:pd.DataFrame, training_targets:np.ndarray, testing_dataset:pd.DataFrame, testing_targets:np.ndarray)->dict:
    """
    Tests KNN Classification for different K's in k_range from the training set on the testing set, and returns a dictionary whose keys are the different values of k and values are the error rates
        k_range: range for k values
        training_set: the dataset used to get the neighbors from
        training_targets: targets of the training_set
        testing_dataset: the points whose targets will be predicted
        testing_targets: targets of the testing_set
    """
    K_error = dict()
    training_dataset= training_dataset.to_numpy()
    testing_dataset=testing_dataset.to_numpy()

    for k in k_range:
        print(f'k= {k}')
        #predict
        predictions = KNN_prediction(k, training_dataset, training_targets, testing_dataset)
        #calculate the error_rate
        K_error[k] = error_rate(testing_targets, predictions)
    return K_error

"""
3)- Generative Models
"""

def prior_probabilities(targets:np.ndarray)->dict:
    """
    Estimates the prior probabilities for each class (estimated as a fraction) and returns a dictionary whose keys are the classes and values are their proportion
        targets: targets of every observation
    """
    classes=['cheap', 'moderate', 'expensive', 'very expensive']
    results = dict()
    total_size = len(targets)
    for class_val in classes:
        results[class_val] = np.sum(targets == class_val) / total_size
    return results

def means_estimate(data:pd.DataFrame, targets:np.ndarray):
    """
    Calculates the mean for every column in data in different classes (unique values of targets)
        data: Dataframe of observations
        targets: targets of observations
    """
    data = data.copy()
    data['targets'] = targets
    result = data.groupby('targets').mean()
    #reorder
    result = result.reindex(['cheap', 'moderate', 'expensive', 'very expensive'])
    return result
    

def covariance_matrices(data:pd.DataFrame, targets:np.ndarray, class_means:np.ndarray)->list:
    """
    Calculates the covariance matrix for every class, and returns them as a list
        data: Dataframe of observations
        targets: targets of observations
        class_means: 2d array whose rows are mean vectors of different classes
    """
    #calculate the covariance matrix of each class separately and append them to a list
    covariance = list()
    classes = ['cheap', 'moderate', 'expensive', 'very expensive']
    for i, cls in enumerate(classes):
        cls_observations = data[targets == cls]
        ceneterd = cls_observations.to_numpy() - class_means[i]
        ceneterd_T = ceneterd.T
        covariance.append( (ceneterd_T @ ceneterd )/(len(cls_observations)-1) )
    
    return covariance

def sample_sizes(data:pd.DataFrame, targets:np.ndarray)->np.ndarray:
    """
    Calculates the sample sizes of each class and returns them as an array
        data: Dataframe of observations
        targets: targets of observations
        class_means: DataFrame whose rows are mean vectors of different classes
    """
    sizes=list()
    classes = ['cheap', 'moderate', 'expensive', 'very expensive']
    for cls in classes:
        cls_observations = data[targets == cls]
        sizes.append(len(cls_observations))
    return np.array(sizes)

def pooled_covariance_matrix(covariance_matrices:list, sample_sizes:np.ndarray):
    """
    Calculates the pooled covariance matrix from a list of covariance matrices
        covariance_matrices: list of covariance matrices
        sample_sizes: sample size of every class
    """
    covariance_matrices = covariance_matrices.copy()
    for i in range(len(covariance_matrices)):
        covariance_matrices[i] *= (sample_sizes[i]-1)
    return np.sum(covariance_matrices, axis=0) / (np.sum(sample_sizes) - len(covariance_matrices))

def lda_probability(observation:np.ndarray, class_means: list|np.ndarray, covariance:np.ndarray, prior_probabilities: list|np.ndarray, cls:str):
    """
    Calculates the probability that the observation is in a given class using Linear Discriminant Analysis
        observation: the observation to predict for
        class_means: a list (or array) representing the mean vectors of each class
        covariance: pooled covariance matrix
        prior_probabilities: a list (or array) representing the prior probabilities of each class
        cls: the class for which the probability is calculated. must be 'cheap' or 'moderate' or 'expensive' or 'very expensive'
    """
    mapping={'cheap':0, 'moderate':1,'expensive':2,'very expensive':3}
    if cls not in mapping.keys():
        raise ValueError("class must be must be 'cheap' or 'moderate' or 'expensive' or 'very expensive'")
    cls_index = mapping[cls]

    densities = np.array( [ stats.multivariate_normal.pdf(observation, class_means[i], covariance) for i in range(4) ] )

    denominator = np.sum( densities*prior_probabilities )
    if denominator == 0:
        return 0.0

    return (prior_probabilities[cls_index] * densities[cls_index] )/ denominator

def qda_probability(observation:np.ndarray, class_means: list|np.ndarray, covariance:list, prior_probabilities: list|np.ndarray, cls:str):
    """
    Calculates the probability that the observation is in a given class using Quadratic Discriminant Analysis
        observation: the observation to predict for
        class_means: a list (or array) representing the mean vectors of each class
        covariance: list of covariance matrices
        prior_probabilities: a list (or array) representing the prior probabilities of each class
        cls: the class for which the probability is calculated. must be 'cheap' or 'moderate' or 'expensive' or 'very expensive'
    """
    mapping={'cheap':0, 'moderate':1,'expensive':2,'very expensive':3}
    if cls not in mapping.keys():
        raise ValueError("class must be must be 'cheap' or 'moderate' or 'expensive' or 'very expensive'")
    cls_index = mapping[cls]

    densities = np.array( [ stats.multivariate_normal.pdf(observation, class_means[i], covariance[i], allow_singular=True) for i in range(4) ] )

    return (prior_probabilities[cls_index] * densities[cls_index] )/ np.sum( densities*prior_probabilities )

def discriminant_analysis_one_point(point:np.ndarray, class_means: list|np.ndarray, covariance:np.ndarray|list, prior_probabilities: list|np.ndarray, lda:bool)->dict:
    """
    Predicts the class of one point using LDA or QDA
    observations: DataFrame of the data to use
            class_means: a list (or array) representing the mean vectors of each class
            covariance: list of covariance matrices
            prior_probabilities: a list (or array) representing the prior probabilities of each class
            lda: True to use LDA, else QDA is used
    """
    probabilities= dict()
    for cls in ['cheap', 'moderate', 'expensive', 'very expensive']:
                if lda:
                    probabilities[cls] = lda_probability(point, class_means, covariance, prior_probabilities, cls)
                else:
                    probabilities[cls] = qda_probability(point, class_means, covariance, prior_probabilities, cls)
    return probabilities
    
def discriminant_analysis_all_probas(observations:pd.DataFrame, class_means: list|np.ndarray, covariance:np.ndarray|list, prior_probabilities: list|np.ndarray, lda:bool)->list:
    """
    Predicts a class for every observation (every row) in observations using LDA or QDA, returns a list of predicted probabilities of each class
        observations: DataFrame of the data to use
        class_means: a list (or array) representing the mean vectors of each class
        covariance: list of covariance matrices
        prior_probabilities: a list (or array) representing the prior probabilities of each class
        lda: True to use LDA, else QDA is used
    """
    predictions = list()
    observations = observations.copy()
    observations = observations.to_numpy()
    for obs in observations:
        probabilities = discriminant_analysis_one_point(obs, class_means, covariance, prior_probabilities, lda)
        predictions.append(probabilities)
    return predictions

def discriminant_analysis_prediction(observations:pd.DataFrame, class_means: list|np.ndarray, covariance:np.ndarray|list, prior_probabilities: list|np.ndarray, lda:bool)->list:
    """
    Predicts a class for every observation (every row) in observations using LDA or QDA, returns a list of predicted classes
        observations: DataFrame of the data to use
        class_means: a list (or array) representing the mean vectors of each class
        covariance: list of covariance matrices
        prior_probabilities: a list (or array) representing the prior probabilities of each class
        lda: True to use LDA, else QDA is used
    """
    predictions = discriminant_analysis_all_probas(observations, class_means, covariance, prior_probabilities, lda)
    predicted_classes = list()
    for prediction in predictions:
        predicted_class = max(prediction, key=prediction.get)
        predicted_classes.append(predicted_class)
    return predicted_classes

def parameters_estimate_perclass(observations:pd.DataFrame, targets:list|np.ndarray)->dict:
    """
    Calculates the means and std of every predictor within each class, returns a dictionary with the following structure: { cls: ({predictor_name: mean}, {predictor_name: std}) }
        observations: DataFrame of observations
        targets: targets of observations
    """
    results = dict()
    #combine the observations and targets into one dataframe
    data = observations.copy()
    data['targets'] = targets
    grouped = {name: group.drop(columns=['targets']) for name, group in data.groupby(targets)}

    for cls in ['cheap', 'moderate', 'expensive', 'very expensive']:
        m = dict(grouped[cls].mean())
        s = dict(grouped[cls].std())
        results[cls] = (m, s)
    return results


def naive_bayes_probabilities(observation:pd.Series, mean_std_estimates:dict, prior_probabilities:list|np.ndarray)->dict:
    """
    Calculates the LOG probability (without dividing on the summing term since it's constant accross all classes and doesn't affect the maximum) of the observation being in each class, and returns a dictionary {class_name; probability}
    the probabilities are caluclated using Normal Distribution (in quantitative variables) and Bernoulli Distribution (in qualitative)
        observation: one observation
        mean_std_estimates: dictionary with the following structure: { cls: ({predictor_name: mean}, {predictor_name: std}) }
        prior_probabilities: a list (or array) representing the prior probabilities of each class

    """
    results = dict()
    classes = ['cheap', 'moderate', 'expensive', 'very expensive']
    for index, cls in enumerate(classes):
        predictor_probabilities = list()

        #quantitative variables
        for pred in ['longitude', 'latitude', 'housing_median_age', 'bedrooms_per_room','population_per_household', 'rooms_per_household', 'log_total_rooms','log_total_bedrooms', 'log_population', 'log_households','log_median_income']:
            
            pred_mean = mean_std_estimates[cls][0][pred]
            pred_std = mean_std_estimates[cls][1][pred]
            probability = stats.norm.logpdf(observation[pred], loc=pred_mean, scale=pred_std)
            predictor_probabilities.append(probability)
        
        #qualitative variables
        for pred in ['<1h ocean', 'inland', 'near ocean', 'island']:
            pred_mean = mean_std_estimates[cls][0][pred]
            probability = stats.bernoulli.logpmf(observation[pred], p =pred_mean)
            predictor_probabilities.append(probability)

        results[cls] = np.sum(predictor_probabilities) + np.log( prior_probabilities[index] )

    classes = ['cheap', 'moderate', 'expensive', 'very expensive']
    normalized_probs = softmax([results[c] for c in classes])
    return dict(zip(classes, normalized_probs))

def naive_bayes_prediction(observations:pd.DataFrame, mean_std_estimates:dict, prior_probabilities:list|np.ndarray)->list:
    """
    Predicts the class of every observation using Naive Bayes, and returns the list of predicted classes
        observations: DataFrame of observations
        mean_std_estimates: dictionary with the following structure: { cls: ({predictor_name: mean}, {predictor_name: std}) }
        prior_probabilities: a list (or array) representing the prior probabilities of each class
    """
    predictions = []
    for index, obs in observations.iterrows():
        results = naive_bayes_probabilities(obs, mean_std_estimates, prior_probabilities)
        predicted_class = max(results, key=results.get)
        predictions.append(predicted_class)
    return predictions

def henze_zirkler_test(training_data:pd.DataFrame)->dict[str:float]:
    """
    Performs Henze-Zirkler test to check if within every class the predictors follow a multivariate normal distribution, and returns the p-value in every class
    returns dict {class_name:p_value} and also prints tha p-values
        training_data: the training DataGrame including the target column 'median_house_value'
    """
    results= dict()
    predictors = list(training_data.columns)
    predictors.remove("median_house_value")
    for cls in training_data["median_house_value"].unique():
        cls_predictors = training_data.loc[training_data["median_house_value"] ==cls, predictors]

        hz, pval, normal = multivariate_normality(cls_predictors)
        print(f"{cls}: p-value= {pval}")
        results[cls] = pval 
    return results

