from pandas import DataFrame, Series
import numpy as np
from scipy import stats
from src.clean import standard_scaling

"""
1)- Evaluation Metrics
"""

def RSS(targets:np.ndarray, predictions: np.ndarray)->float:
    """
    Calculates the Residual Sum of Squares
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """

    return ((targets - predictions)**2).sum()

def MSE(targets:np.ndarray, predictions: np.ndarray)->float:
    """
    Calculates the Mean Squared Error
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """

    return RSS(targets, predictions) / len(targets)

def RSE(targets:np.ndarray, predictions: np.ndarray, p: int)->float:
    """
    Calculates the Residual Standard Error (which can also be an estimate for the standard deviation of the irreducible error epsilon)
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """

    return np.sqrt(RSS(targets, predictions) / (len(targets) - p - 1) )

def TSS(targets:np.ndarray)->float:
    """
    Calculates the Original Sum of Variation in the Data
        targets: array of the targets (y)
    """
    mean_val = targets.mean()

    return ((targets-mean_val)**2).sum()

def R_Square(targets:np.ndarray, predictions: np.ndarray)->float:
    """
    Calculates the proportion of the explained variability by fitting the linear model
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """
    TSS_val = TSS(targets)
    RSS_val = RSS(targets, predictions)

    return (TSS_val-RSS_val)/TSS_val

def mallows_cp(targets:np.ndarray, predictions: np.ndarray, p: int)->float:
    """
    Calculates Mallow's Cp
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """
    RSS_val = RSS(targets, predictions)
    #it's the RSE**2, just calculated again to calculate RSS only once
    eps_variance = RSS_val / (len(targets) - p - 1)
    numerator = RSS_val + 2*p*eps_variance
    n=len(targets)
    return numerator/n

def aic(targets:np.ndarray, predictions: np.ndarray, p: int)->float:
    """
    Calculates Akaike Information Criterion (AIC)
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """
    RSS_val = RSS(targets, predictions)
    eps_variance = RSS_val / (len(targets) - p - 1)
    n=len(targets)
    return (RSS_val + 2*p*eps_variance) / n

def bic(targets:np.ndarray, predictions: np.ndarray, p: int)->float:
    """
    Calculates Bayesian Information Criterion (BIC)
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """
    RSS_val = RSS(targets, predictions)
    eps_variance = RSS_val / (len(targets) - p - 1)
    n=len(targets)
    return (RSS_val + np.log(n)*p*eps_variance) / n

def adjusted_r2(targets:np.ndarray, predictions: np.ndarray, p: int)->float:
    """
    Calculates the Adjusted R**2
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """
    RSS_val = RSS(targets, predictions)
    TSS_val = TSS(targets)
    n=len(targets)
    return 1 - ( (RSS_val/(n-p-1)) / (TSS_val/(n-1)) )

def F_statistic(targets:np.ndarray, predictions: np.ndarray, p:int)->tuple[float, float]:
    """
    Calculates the F-Statistic which is used in hypothesis testing, returns the F-statistic and the p-value (probability of the null hypothesis being true)
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """
    unexplained_variance = RSS(targets, predictions)

    explained_variance = TSS(targets)-unexplained_variance

    num_dof = p
    denum_dof = len(targets) - p - 1

    F_stat = (explained_variance/num_dof)/(unexplained_variance/denum_dof)
    p_value = stats.f.sf(F_stat, num_dof, denum_dof)

    return (F_stat, p_value)

def standard_errors(dataset:DataFrame, targets:np.ndarray, predictions: np.ndarray, p: int)->np.ndarray:
    """
    Calculates the standard errors, returns an array where each element represents a standard error of a parameter (the first element is the standard error of the bias)
        dataset: data matrix
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """
    estimated_irreducible_variance = (RSE(targets, predictions, p))**2
    X = dataset.to_numpy()
    y = targets.to_numpy()
    X = np.insert(X,0, np.ones( len(y) ), axis=1 )
    X_T = X.T

    covariance_matrix = estimated_irreducible_variance * ( np.linalg.inv(X_T @ X) )
    return np.sqrt(np.diag(covariance_matrix))

def t_statistics(dataset:DataFrame, targets:np.ndarray, predictions: np.ndarray, parameters: np.ndarray)->np.ndarray:
    """
    Calculates the t-statistics, returns an array where each element represents a t-statistic of a parameter (the first element is the t-statistic of the bias)
        dataset: data matrix
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        parameters: array of parameters, where the first element is the bias (intercept)
    """
    errors = standard_errors(dataset, targets, predictions, len(parameters)-1)
    return parameters/errors

def p_values(dataset:DataFrame, targets:np.ndarray, predictions: np.ndarray, parameters: np.ndarray)->np.ndarray:
    """
    Calculates the p-values, returns an array where each element represents a p-value of a parameter (the first element is the p-value of the bias)
        dataset: data matrix
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        parameters: array of parameters, where the first element is the bias (intercept)
    """
    t_stats = t_statistics(dataset, targets, predictions, parameters)
    df = len(targets) - len(parameters)
    return 2 * stats.t.sf(np.abs(t_stats), df)


"""
2)- Linear Regression
"""
def normal_equation(data:DataFrame|np.ndarray, targets:Series|np.ndarray) -> np.ndarray:
    """
    Estimates the Least Squares Coefficients using Normal Equation, return an array whose first element is the bias (intercept) and the remaining elements are the weights
        data: data matrix
        y: targets column
    """
    if not isinstance(data, np.ndarray):
        X = data.to_numpy()
    else:
        X = data
    if not isinstance(targets, np.ndarray):
        y = targets.to_numpy()
    else:
        y = targets
    #add the column of ones to X
    X= np.insert(X, 0, np.ones( len(y) ), axis=1 )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def destandardize_params(parametes:np.ndarray, original_stats:dict)->list:
    """
    Gives the parameters for the unscaled features using the original means and std's
        parameters: LSE parameters for the standardized features
        original_stats: dictionary containing the original mean and std for all features
    """
    
    stats = list(original_stats.values())
    scaled_features = len(stats)
    means = np.array([stats[i][0] for i in range(scaled_features)])
    stds = np.array([stats[i][1] for i in range(scaled_features)])
    #weights
    weights = parametes[1:scaled_features+1] / stds

    #bias
    bias = parametes[0] - (weights * means).sum()

    return [float(bias)] + weights.tolist()

def linear_predict(data:DataFrame, parameters:np.ndarray)-> np.ndarray:
    """
    Predicts the response of every row of the data using a Linear Model with the given parameters and returns the array of predictions (element i of the prediction array corresponds to the prediction of the ith row of the data matrix)
        data: data matrix
        parameters: array of parameters, where the first element is the bias (intercept)
    """
    data = data.to_numpy()
    weights = parameters[1:]
    intercept= parameters[0]

    return (intercept + (data @ weights.T)).T

def stepwise_selection(training_set:DataFrame, training_targets:np.ndarray, predictor_names:list[str], metric:callable, maximize:bool=False)->tuple[np.ndarray,float, list[str]]:
    """
    Performs Hybrid Stepwise Selection, and Evaluate using the Metric. Returns a tuple (parameter_estimates with first element representing the bias, best_metric, list of predictors used in the model)
        training_set: DataFrame of the training set used to fit the models
        training_targets: array representing the targets of the training_set
        predictor_names: list of the names of all the observed (and featured) predictors
        metric: function returning a float for the metric used in the models evaluation (ex: AIC, BIC, Adjusted R**2)
        maximize: bool, indicating if the goal is to maximize the metric. default is minimize
    """
    used_predictors = []
    available_predictors = predictor_names.copy()

    #these flags will be set to false when forward (or backward) selection improves the model
    forward_flag=False
    backward_flag = False

    #start by the null model
    predictions = training_targets.mean()
    best_metric = metric(training_targets, predictions, 0)
    best_parameters = [training_targets.mean()]

    #stop when all predictors are included or both forward and backward selections didn't improve the model
    while(len(available_predictors)>0 and (not forward_flag or not backward_flag)):
        forward_flag=True
        backward_flag = True
        #variables that hold the name of the predictors that will be potentially added (or removed)
        pred_to_add=None
        pred_to_remove=None
        #Forward Selection
        for predictor in available_predictors:
            X = training_set[used_predictors + [predictor] ]
            parameters = normal_equation(X, training_targets)
            predictions = linear_predict(X, parameters)
            metric_val = metric(training_targets, predictions, len(used_predictors) + 1)
            if( (maximize and metric_val>best_metric) or (not maximize and metric_val<best_metric) ):
                best_metric = metric_val
                best_parameters = parameters
                pred_to_add = predictor
        if pred_to_add is not None:
            print(f'Adding the predictor {pred_to_add} resulting in {metric.__name__}={best_metric}')
            available_predictors.remove(pred_to_add)
            used_predictors.append(pred_to_add)
            forward_flag=False

        #Backward Selection
        for predictor in used_predictors:
            used_pred_copy = used_predictors.copy()
            used_pred_copy.remove(predictor)
            X = training_set[used_pred_copy]
            parameters = normal_equation(X, training_targets)
            predictions = linear_predict(X, parameters)
            metric_val = metric(training_targets, predictions, len(used_pred_copy))
            if( (maximize and metric_val>best_metric) or (not maximize and metric_val<best_metric) ):
                best_metric = metric_val
                best_parameters = parameters
                pred_to_remove = predictor
        
        if pred_to_remove is not None:
            print(f'Removing the predictor {pred_to_remove} resulting in {metric.__name__}={best_metric}')
            available_predictors.append(pred_to_remove)
            used_predictors.remove(pred_to_remove)
            backward_flag=False
    return (best_parameters, best_metric, used_predictors)


"""
3)- KNN Regression
"""

def K_nearest_neighbors(k:int, dataset:np.ndarray, targets:np.ndarray, point:np.ndarray)->float:
    """
    Finds the K-Nearest Neighbors to Point from the Dataset, and returns the average of their targets
        k: the number of neighbors to consider, lower k means higher flexibility
        dataset: the dataset used to get the neighbors from (the training set)
        targets: the targets of the dataset
        point: the point whose target will be predicted
    """
    #euclidean distance
    distances = np.linalg.norm(dataset - point, axis=1)
    closest_indices = np.argpartition(distances, k)[:k]
    neighbors_targets = targets[closest_indices]
    return neighbors_targets.mean()

def KNN_prediction(k:int, dataset:np.ndarray, targets:np.ndarray, testing_dataset:np.ndarray)->np.ndarray:
    """
    Calculates the prediction of the testing dataset using KNN regression from the dataset and its targets, returns an array for the predictions
        k: the number of neighbors to consider, lower k means higher flexibility
        dataset: the dataset used to get the neighbors from (the training set)
        targets: the targets of the dataset
        testing_dataset: the points whose targets will be predicted
    """
    if isinstance(dataset, DataFrame):
        dataset= dataset.to_numpy()
    if isinstance(testing_dataset, DataFrame):
        testing_dataset= testing_dataset.to_numpy()

    predictions = np.zeros(len(testing_dataset))
    for i, row in enumerate(testing_dataset):
        t = K_nearest_neighbors(k, dataset, targets, row)
        predictions[i]= t
    return predictions


def KNN_regression_tuning(k_range:range, training_dataset:DataFrame, training_targets:np.ndarray, testing_dataset:DataFrame, testing_targets:np.ndarray)->dict:
    """
    Tests KNN Regression for different K's in k_range from the training set on the testing set, and returns a dictionary whose keys are the different values of k and values are the MSE
        k_range: range for k values
        training_set: the dataset used to get the neighbors from
        training_targets: targets of the training_set
        testing_dataset: the points whose targets will be predicted
        testing_targets: targets of the testing_set
    """
    K_MSE = dict()
    training_dataset= training_dataset.to_numpy()
    testing_dataset=testing_dataset.to_numpy()
    for k in k_range:
        #predict
        predictions = KNN_prediction(k, training_dataset, training_targets, testing_dataset)
        #calculate the MSE
        K_MSE[k] = MSE(testing_targets, predictions)
    return K_MSE


