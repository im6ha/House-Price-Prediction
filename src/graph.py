import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

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
    cols_count = end_col - start_col

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