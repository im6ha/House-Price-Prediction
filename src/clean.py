import pandas as pd
from numpy import log1p, ndarray
def one_hot_encoding(column:pd.Series) -> pd.DataFrame:
    """
    Performs one-hot encoding on a column (feature) and returns the new columns (variables), by creating a dummy boolean variable for every unique value of the original variable
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

def standard_scaling(dataset: pd.DataFrame, start_col:int, end_col:int, skip_col:int, stats: dict[str, tuple[float, float]] | None = None) -> tuple[pd.DataFrame, dict[str, tuple[float, float] ]]:
    """
    Performs standard scaling on a set of columns from the dataset, and returns the new scaled dataset and a dictionary whose keys are the scaled columns names and values are tuples (original_mean, original_std)
        dataset: Pandas DataFrame representing the full dataset
        start_col: index of first column to standardize
        end_col: index of last column to standardize (EXCLUSIVE)
        skip_col: index of a column to skip (not scale, done because the target column is left in the middle)
        stats: stats to use as the original mean and std instead of recalculating
    """
    dataset_copy = dataset.copy()
    stats_result = dict()
    for i in range(start_col, end_col):
        if i == skip_col:
            continue
        col_name = dataset.columns[i]
        og_mean = dataset_copy.iloc[:, i].mean() if stats is None else stats[col_name][0]
        og_std = dataset_copy.iloc[:, i].std() if stats is None else stats[col_name][1]
        col_name = dataset_copy.columns[i]
        dataset_copy.iloc[:, i] = (dataset_copy.iloc[:, i] - og_mean) / og_std
        stats_result[col_name] = (og_mean, og_std)

    return (dataset_copy, stats_result)

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

def classification_formulation(Y_column:pd.Series)->pd.Series:
    """
    Performs Classification Formulation to a standard scaled column, and returns the new column:
    cheap: less than 150K$
    moderate: between 150K$ and 250K$
    expensive: between 250K$ and 350K$
    very expensive: greater than 350K$
        Y_column: scaled column to formulate
    """
    cheap = 150000
    moderate = 250000
    expensive = 350000
    cheap_mask = Y_column <=cheap
    moderate_mask =(Y_column > cheap) & (Y_column <= moderate)
    expensive_mask = (Y_column > moderate) & (Y_column <= expensive)
    very_expensive_mask = Y_column>expensive
    Y_column.loc[cheap_mask]= "cheap"
    Y_column.loc[moderate_mask] ="moderate"
    Y_column.loc[expensive_mask] = "expensive"
    Y_column.loc[very_expensive_mask] = "very expensive"

    return Y_column

def binary_classification_formulation(Y_column:pd.Series)->pd.Series:
    """
    Performs Binary Classification Formulation to a standard scaled column, and returns the new column:
    cheap: less than 300K$
    expensive: greater than 300K$
        Y_column: scaled column to formulate
    """
    boundary = 300000
    cheap_mask = Y_column <=boundary
    expensive_mask =Y_column > boundary
    Y_column.loc[cheap_mask]= 0
    Y_column.loc[expensive_mask] = 1
    return Y_column