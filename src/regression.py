from pandas import DataFrame, Series
import numpy as np
from scipy import stats

def normal_equation(data:DataFrame, targets:Series) -> np.array:
    """
    Estimates the Least Squares Coefficients using Normal Equation, return an array whose first element is the bias (intercept) and the remaining elements are the weights, and also the time of execution in seconds
        data: data matrix
        y: targets column
    """
    X = data.to_numpy()
    y = targets.to_numpy()
    #add the column of ones to X
    X= np.insert(X, 0, np.ones( len(y) ), axis=1 )

    X_T = X.T
    temp = X_T @ X
    temp = np.linalg.inv(temp)
    temp = temp @ X_T 

    return temp@y


def linear_predict(data:DataFrame, parameters:np.array)-> np.array:
    """
    Predicts the response of every row of the data using a Linear Model with the given parameters and returns the array of predictions (element i of the prediction array corresponds to the prediction of the ith row of the data matrix)
        data: data matrix
        parameters: array of parameters, where the first element is the bias (intercept)
    """
    data = data.to_numpy()
    weights = parameters[1:]
    intercept= parameters[0]

    return (intercept + (data @ weights.T)).T

def RSS(targets:np.array, predictions: np.array)->float:
    """
    Calculates the Residual Sum of Squares
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """

    return ((targets - predictions)**2).sum()

def MSE(targets:np.array, predictions: np.array)->float:
    """
    Calculates the Mean Squared Error
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """

    return RSS(targets, predictions) / len(targets)

def RSE(targets:np.array, predictions: np.array, p: int)->float:
    """
    Calculates the Residual Standard Error (which can also be an estimate for the standard deviation of the irreducible error epsilon)
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        p: number of predictors
    """

    return np.sqrt(RSS(targets, predictions) / (len(targets) - p - 1) )

def TSS(targets:np.array)->float:
    """
    Calculates the Original Sum of Variation in the Data
        targets: array of the targets (y)
    """
    mean_val = targets.mean()

    return ((targets-mean_val)**2).sum()

def R_Square(targets:np.array, predictions: np.array)->float:
    """
    Calculates the proportion of the explained variability by fitting the linear model
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """
    TSS_val = TSS(targets)
    RSS_val = RSS(targets, predictions)

    return (TSS_val-RSS_val)/TSS_val

def F_statistic(targets:np.array, predictions: np.array, p:int)->tuple[float, float]:
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

def standard_errors(dataset:DataFrame, targets:np.array, predictions: np.array, p: int)->np.array:
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

def t_statistics(dataset:DataFrame, targets:np.array, predictions: np.array, parameters: np.array)->np.array:
    """
    Calculates the t-statistics, returns an array where each element represents a t-statistic of a parameter (the first element is the t-statistic of the bias)
        dataset: data matrix
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
        parameters: array of parameters, where the first element is the bias (intercept)
    """
    errors = standard_errors(dataset, targets, predictions, len(parameters)-1)
    return parameters/errors

def p_values(dataset:DataFrame, targets:np.array, predictions: np.array, parameters: np.array)->np.array:
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