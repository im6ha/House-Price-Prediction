from pandas import DataFrame
import numpy as np

"""
1)- Logistic Regression
"""


def logistic_function(observations:np.ndarray, parameters:np.array)->np.array:
    """
    Calculates the logistic function which is the Probability that the observation is 'expensive'
        observations: Dataframe representing the observations, shape=(n, p)
        parameters: array of the parameters, with the first parameter representing the bias
    """


    linear_comb = np.dot(observations, parameters.T)

    return 1/( 1+np.exp(-linear_comb) )

def cost_function(observations:DataFrame, parameters:np.array, targets:np.array)->float:
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

def gradient(observations:DataFrame, parameters:np.array, targets:np.array)->np.array:
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

def gradient_descent(observations:DataFrame, targets:np.array, alpha:float, epsilon:float, initial_parameters:np.array)->tuple[np.array, dict]:
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


def K_nearest_neighbors(k:int, dataset:np.array, targets:np.array, point:np.array)->dict:
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
        result[target_class] = neighbors_targets[neighbors_targets==target_class].count()
    result = dict( sorted(result.items(), key=lambda x: x[1] ) )
    return result

def KNN_prediction(k:int, dataset:np.array, targets:np.array, testing_dataset:np.array)->np.array:
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
        t = K_nearest_neighbors(k, dataset, targets, row).keys()[0]
        predictions[i]= t
    return predictions
