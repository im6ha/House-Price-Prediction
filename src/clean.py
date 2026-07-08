import pandas as pd
from numpy import log1p
def one_hot_encoding(column:pd.Series) -> pd.DataFrame:
    """
    Performs one-hot encoding on a column (variable) and returns the new columns (variables), by creating a dummy boolean variable for every unique value of the original variable
        column: Pandas Series (a column of a dataframe)
    """
    values = column.unique()
    #avoid dummy trap by removing one dummy variable
    values = values[1:]
    new_variables = pd.DataFrame()

    for value in values:
        new_variable = (column == value).astype(int)
        new_variables[f'{value}'.lower()] = new_variable
    
    return new_variables

def standard_scaling(dataset: pd.DataFrame, start_col:int, end_col:int) -> tuple[pd.DataFrame, dict[str, (float, float) ]]:
    """
    Performs standard scaling on a set of columns from the dataset, and returns the new scaled dataset and a dictionary whose keys are the scaled columns names and values are tuples (original_mean, original_std)
        dataset: Pandas DataFrame representing the full dataset
        start_col: index of first column to standardize
        end_col: index of last column to standardize (EXCLUSIVE)
    """
    stats = dict()
    for i in range(start_col, end_col):
        og_mean = dataset.iloc[:, i].mean()
        og_std = dataset.iloc[:, i].std()
        col_name = dataset.columns[i]
        dataset.iloc[:, i] = (dataset.iloc[:, i] - og_mean) / og_std
        stats[col_name] = (og_mean, og_std)

    return (dataset, stats)

def log_transform(dataset: pd.DataFrame, col_names: list[str]) -> pd.DataFrame:
    """
    Performs log transformation on a set of columns from the dataset, and returns the new dataset and adding the prefix 'log_' to the transformed columns
        dataset: Pandas Dataframe
        col_names: a list containing the NAMES of the columns to transform
    """
    for col in col_names:
        dataset[f'log_{col}'] = log1p(dataset[col])
        dataset = dataset.drop(columns=col)
    return dataset

def classification_formulation(Y_column:pd.Series, original_mean:float, original_std:float)->pd.Series:
    """
    Performs Classification Formulation to a standard scaled column, and returns the new column:
    cheap: less than 150K$
    moderate: between 150K$ and 250K$
    expensive: between 250K$ and 350K$
    very expensive: greater than 350K$
        Y_column: scaled column to formulate
        original_mean: original mean of the column
        original_std: original standard deviation of the column
    """
    cheap = (150000 - original_mean)/original_std
    moderate = (250000 - original_mean)/original_std
    expensive = (350000 - original_mean)/original_std
    cheap_mask = Y_column <=cheap
    moderate_mask =(Y_column > cheap) & (Y_column <= moderate)
    expensive_mask = (Y_column > moderate) & (Y_column <= expensive)
    very_expensive_mask = Y_column>expensive
    Y_column.loc[cheap_mask]= "cheap"
    Y_column.loc[moderate_mask] ="moderate"
    Y_column.loc[expensive_mask] = "expensive"
    Y_column.loc[very_expensive_mask] = "very expensive"

    return Y_column

def binary_classification_formulation(Y_column:pd.Series, original_mean:float, original_std:float)->pd.Series:
    """
    Performs Binary Classification Formulation to a standard scaled column, and returns the new column:
    cheap: less than 300K$
    expensive: greater than 300K$
        Y_column: scaled column to formulate
        original_mean: original mean of the column
        original_std: original standard deviation of the column
    """
    boundary = (300000 - original_mean)/original_std
    cheap_mask = Y_column <=boundary
    expensive_mask =Y_column > boundary
    Y_column.loc[cheap_mask]= 0
    Y_column.loc[expensive_mask] = 1
    return Y_column