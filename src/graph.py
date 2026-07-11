import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import numpy as np
import time

"""
The functions are designed to work for the California Housing dataset specifically
"""

def univariate_plot(dataset : DataFrame, start_col:int, end_col:int) -> None:
    """
    Plots appropriate graphs for univariate analysis (histograms and boxplots) for a set of columns from the dataset
        dataset: Pandas DataFrame
        start_col: index of first column to standardize
        end_col: index of last column to standardize (EXCLUSIVE)
    """
    cols_count = end_col -start_col

    fig, axes = plt.subplots(cols_count, 2, figsize=(15, 6*cols_count))
    axes = axes.flatten()
    colors=['#7209B7', '#3A0CA3', '#4361EE', 
    '#4CC9F0', '#4895EF', '#560BAD', 
    '#B5179E', '#E01E37', '#F72585']
    for i, col in enumerate(range(start_col, end_col)):
        axes[2*i].hist(dataset.iloc[:, col], color=colors[i%9], edgecolor='black', alpha=0.8 )
        axes[2*i].set_title( dataset.columns[col] )
        axes[2*i].set_xlabel('Value')
        axes[2*i].set_ylabel('Frequency')

        axes[2*i+1].boxplot(dataset.iloc[:, col])
        axes[2*i+1].set_title( dataset.columns[col] )
        axes[2*i+1].set_xlabel('Value')
        axes[2*i+1].set_ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()

def locations_scatter(dataset: DataFrame)-> None:
    """
    Plots a scatter plot for the longitude vs latitude
        dataset: Pandas DataFrame
    """
    plt.scatter(dataset['longitude'], dataset['latitude'])
    plt.title("longitude vs latitude")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()

def correlation_heatmap(dataset: DataFrame)-> None:
    """
    Plots the correlation heatmap between every pair of numerical column
        dataset: Pandas DataFrame
    """
    #exclude last column since it's not yet numerical
    matrix = dataset.iloc[:, :9].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def bivariate_plot(dataset: DataFrame)-> None:
    """
    Plots the scatter plots between every numerical column and the response (target) column
        dataset: Pandas DataFrame
    """

    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    axes= axes.flatten()

    colors=['#7209B7', '#3A0CA3', '#4361EE', 
    '#4CC9F0', '#4895EF', '#560BAD', 
    '#B5179E', '#E01E37']

    for i in range(8):
        axes[i].scatter(dataset.iloc[:, i], dataset["median_house_value"], c=colors[i])
        axes[i].set_title(f"{dataset.columns[i]} vs median_house_value")
        axes[i].set_ylabel("median_house_value")
        axes[i].set_xlabel(dataset.columns[i])
    
    plt.tight_layout()
    plt.show()

def residuals_vs_predicted_values(targets:np.ndarray, predictions:np.ndarray):
    """
    Plots the Residuals vs Predicted Values
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """
    residuals = targets - predictions
    plt.scatter(predictions, residuals, c='red', s=10)
    plt.xlabel("The Predicted Values")
    plt.ylabel("Residuals")
    plt.title('Residuals vs Predicted Values')
    #add a line at residuals=0
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.show()

def residuals_histogram(targets:np.ndarray, predictions:np.ndarray):
    """
    Plots a histogram for the Residuals (used to check the normality of the residuals)
        targets: array of the targets (y)
        predictions: array of the prediction (y-hat)
    """
    residuals = targets - predictions
    plt.hist(residuals)
    plt.xlabel("The Residuals")
    plt.ylabel("Values")
    plt.title('Residuals Histogram')
    plt.show()

def K_Cost(training_k_cost:dict, testing_k_cost:dict, MSE=True):
    """
    Plots the training MSE (or error rate) and testing MSE (or error rate) with respect to k
        training_k_mse: dictionary with keys as values of k and values as the calculated training MSE
        testing_k_mse:  dictionary with keys as values of k and values as the calculated testing MSE
        MSE: True if the values are mse, False if they are error rates
    """
    name = 'MSE' if MSE else 'Error Rate'
    k = training_k_cost.keys()
    training_cost = training_k_cost.values()
    testing_cost = testing_k_cost.values()
    if k != testing_k_cost.keys():
        raise ValueError('The K values must be similair in both parameters')
    
    plt.plot(k, training_cost, label=f"Training {name}", color='red')
    plt.plot(k, testing_cost, label=f"Testing {name}", color='blue')
    plt.xlabel("K")
    plt.ylabel(name)
    plt.title(f"{name} vs K")
    plt.legend()
    plt.grid(True)
    plt.show()

def gradient_descent_tuning(observations:DataFrame, targets:np.ndarray, tuning_values, default_value: float, parameter:str, gradient_descent):
    """
    Runs gradient descent with multiple values for alpha/epsilon, in each run we capture: the cost and time. Two graphs are plotted
        observations: Dataframe of all the observations, shape=(n, p)
        targets: array of the targets (of the observations)
        tuning_values: range of values that tuned parameter takes throughout iterations
        default_value: value taken by the non-tuned parameter
        parameter: 'A' if we tune based on alpha or 'E' if we tune based on epsilon
        gradient_descent: function object of gradient descent
    """
    zeros = np.zeros(shape=(observations.shape[1] + 1))
    iterations = 0
    times = []
    costs = []
    for value in tuning_values:
        iterations+=1
        print(f'iteration #{iterations} starts now')

        if parameter == 'A':
            alpha = value
            epsilon = default_value
        elif parameter == 'E':
            alpha = default_value
            epsilon = value
        else:
            raise ValueError("parameter can be eiher 'A' or 'E'")
        
        start = time.time()
        params, cost = gradient_descent(observations, targets, alpha, epsilon, zeros)
        end = time.time()
        times.append((end-start)*1000)
        costs.append(cost.values()[-1])
    
    parameter_name = 'alpha' if parameter=='A' else 'epsilon'

    fig, axes = plt.subplots(2,1, figsize=(10, 12))
    axes[0].plot(tuning_values, times, color='green')
    axes[0].set_xlabel(parameter_name)
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title(f'{parameter_name} vs Time')
    axes[0].grid(True)

    axes[1].plot(tuning_values, costs, label='Cost', color='red')
    axes[1].set_xlabel(parameter_name)
    axes[1].set_ylabel('Cost')
    axes[1].set_title(f'{parameter_name} vs Cost')
    axes[1].grid(True)

def gradient_descent_convergence(iterations, costs):
    plt.plot(iterations, costs, color='blue')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Convergence")
    plt.grid(True)
    plt.show()