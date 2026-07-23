import pandas as pd
import numpy as np
import scipy.stats as stats
from pingouin import multivariate_normality


"""
1)- Logistic Regression
"""

def logistic_function(observations:np.ndarray, parameters:np.ndarray)->np.ndarray:
    """
    Calculates the logistic function which is the Probability that the observation is 'expensive'
        observations: Dataframe representing the observations, shape=(n, p)
        parameters: array of the parameters, with the first parameter representing the bias
    """


    linear_comb = np.dot(observations, parameters.T)

    return 1/( 1+np.exp(-linear_comb) )

def cost_function(observations:pd.DataFrame, parameters:np.ndarray, targets:np.ndarray)->float:
    """
    Calculates and returns the cost function which is the log-loss function
        observations: Dataframe of all the observations, shape=(n, p)
        parameters: array of the parameters, with the first parameter representing the bias, shape=(p+1)
        targets: array of the targets (of the observations)
    """
    observations = np.column_stack( ( np.ones(observations.shape[0]), observations) )

    predictions_of_success= logistic_function(observations, parameters)
    predictions_of_failure = 1-predictions_of_success

    #apply log
    predictions_of_success = np.log(predictions_of_success)
    predictions_of_failure = np.log(predictions_of_failure)

    #multiply by the targets
    predictions_of_success = np.dot(targets, predictions_of_success)
    predictions_of_failure = np.dot((1-targets), predictions_of_failure)

    predictions = predictions_of_failure + predictions_of_success

    return -np.sum(predictions)

def gradient(observations:pd.DataFrame, parameters:np.ndarray, targets:np.ndarray)->np.ndarray:
    """
    Calculates the gradient and returns an array whose first element is the partial derivative with respect to the bias
        observations: Dataframe of all the observations, shape=(n, p)
        parameters: array of the parameters, with the first parameter representing the bias, shape=(p+1)
        targets: array of the targets (of the observations)
    """
    observations = observations.to_numpy()
    observations = np.column_stack( ( np.ones(observations.shape[0]), observations) )
    observations_T= observations.T
    predictions = logistic_function(observations, parameters)
    diff = predictions-targets
    return observations_T@ diff /observations.shape[0]

def gradient_descent(observations:pd.DataFrame, targets:np.ndarray, alpha:float, epsilon:float, initial_parameters:np.ndarray)->tuple[np.ndarray, dict]:
    """
    Performs gradient descent and returns the parameters (first element is the bias) and a dictionary whose keys are iterations and values are costs (used for convergence)
        observations: Dataframe of all the observations, shape=(n, p)
        targets: array of the targets (of the observations)
        alpha: learning rate, specifies the size of the steps (smaller alpha -> slower convergence)
        epsilon: stopping condition, stop when the difference in cost is less than epsilon (smaller epsilon -> closer to the minimum)
        initial_parameters: initial values given to the parameters
    """
    #initialize all parameters
    parameters = initial_parameters

    #difference in cost
    difference = np.inf

    iters_costs = dict()
    iterations=0
    while(difference>epsilon):
        iterations+=1
        old_cost = cost_function(observations, parameters, targets)
        gradient_val = gradient(observations, parameters, targets)
        #update
        parameters = parameters - alpha * gradient_val

        new_cost = cost_function(observations, parameters, targets)
        print(new_cost)
        iters_costs[iterations] = new_cost
        difference = old_cost - new_cost

        if difference < 0:
            print('Cost is increasing, alpha is too big')
            return (parameters, iters_costs)
    
    return (parameters, iters_costs)
            


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
        result[target_class] = (neighbors_targets==target_class).sum()
    result = dict( sorted(result.items(), key=lambda x: x[1] ) )
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
        t = list(K_nearest_neighbors(k, dataset, targets, row).keys())[0]
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
    classes= np.unique(targets)
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
    #reorder, since by default they'll be sorted alphabetically
    result = result.iloc[[0, 2, 1, 3]]
    return result
    

def covariance_matrices(data:pd.DataFrame, targets:np.ndarray, class_means:pd.DataFrame)->list:
    """
    Calculates the covariance matrix for every class, and returns them as a list
        data: Dataframe of observations
        targets: targets of observations
        class_means: DataFrame whose rows are mean vectors of different classes
    """
    #calculate the covariance matrix of each class separately and append them to a list
    covariance = list()

    for i, cls in enumerate(np.unique(targets)):
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
    for cls in np.unique(targets):
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

def discriminant_analysis_prediction(observations:pd.DataFrame, class_means: list|np.ndarray, covariance:np.ndarray|list, prior_probabilities: list|np.ndarray, lda:bool)->list:
    """
    Predicts a class for every observation (every row) in observations using LDA or QDA, returns a list of predicted classes
        observations: DataFrame of the data to use
        class_means: a list (or array) representing the mean vectors of each class
        covariance: list of covariance matrices
        prior_probabilities: a list (or array) representing the prior probabilities of each class
        lda: True to use LDA, else QDA is used
    """
    predictions = list()
    probabilities=dict()
    observations = observations.copy()
    observations = observations.to_numpy()
    for obs in observations:
        for cls in ['cheap', 'moderate', 'expensive', 'very expensive']:
            if lda:
                probabilities[cls] = lda_probability(obs, class_means, covariance, prior_probabilities, cls)
            else:
                probabilities[cls] = qda_probability(obs, class_means, covariance, prior_probabilities, cls)

        predicted_class = max(probabilities, key=probabilities.get)
        predictions.append(predicted_class)
    return predictions

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
    return results

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

