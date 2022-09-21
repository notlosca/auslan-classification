import numpy as np
import pandas as pd
from typing import Tuple

def extract_matrix(time_series_data:list) -> np.array:
    """
    Extract the matrix of the time series passed

    Args:
        time_series_data (list): list of event of the time series.
        Each event is a dictionary

    Returns:
        np.array: matrix m x n.
        m is the length of the time series (how many events were acquired)
        n is the number of features
    """
    ts_matrix = np.array([list(i.values()) for i in time_series_data])
    return ts_matrix

def compute_mean_and_std(X_train:np.ndarray) -> Tuple[np.array, np.array]:
    """
    Compute mean and standard deviation of the np.ndarray X of time series.
    Each entry of X is a (time series) matrix itself, that's why we loop.
    Stack vertically each matrix and compute mean and standard deviation.

    Args:
        X_train (np.ndarray): Data matrix.

    Returns:
        Tuple[np.array, np.array]: Mean and standard deviation of the entire np.ndarray of time series X.
    """
    X_train_no_nan = []
    
    # select entries != nan
    for i in range(X_train.shape[0]):
        sample = X_train[i]
        mask = ~np.isnan(sample)
        sample_no_nan = sample[mask].reshape(-1,22)
        X_train_no_nan.append(sample_no_nan)
    
    X_train_no_nan = np.vstack(X_train_no_nan)
    mean_ary = np.mean(X_train_no_nan, axis=0)
    std_ary = np.std(X_train_no_nan, axis=0)
    
    return mean_ary, std_ary

def apply_z_standardization(X:np.ndarray, mean_ary:np.ndarray, std_ary:np.ndarray) -> np.ndarray:
    """
    Apply z standardization to each time series of the matrix X.
    Mean and standard deviation are passed as parameters.
    Use compute_mean_and_std to compute them.

    Args:
        X (np.ndarray): np.ndarray of time series to standardize.
        mean_ary (np.ndarray): Array of mean values. One entry for each predictor.
        std_ary (np.ndarray): Array of std_dev values. One entry for each predictor.

    Returns:
        np.ndarray: The scaled np.ndarray.
    """
    for i in range(X.shape[0]):
        # select the sample
        sample = X[i]
        
        # select entries != nan
        mask = ~np.isnan(sample)
        sample_no_nan = sample[mask].reshape(-1,22)
        
        # scale data
        scaled = (sample_no_nan - mean_ary) / std_ary
        
        # update values
        X[i].ravel()[:len(scaled.ravel())] = scaled.ravel()
        
    return X

def scale_time_series(time_series_dataset:pd.Series, mean_ary:np.array, std_ary:np.array) -> np.ndarray:
    """
    Scale each time series in a dataset by subtracting the mean and dividing by standard deviation

    Args:
        time_series_dataset (pd.Series): Time series dataset
        mean_ary (np.array): Mean vector, one entry for each predictor
        std_ary (np.array): Standard deviation vector, one entry for each predictor

    Returns:
        np.ndarray: scaled time series dataset in the numpy ndarray format. Each entry is a time series.
    """
    scaled_ts = np.zeros(shape=(len(time_series_dataset)))
    
    for i in range(len(time_series_dataset)):
        ts = time_series_dataset.iloc[i] # time x predictors
        ts = (ts-mean_ary)/std_ary
        scaled_ts[i] = ts
    
    return scaled_ts

# each row of the Series object is an array. Classifiers won't read it. We create a matrix of values.
def from_series_to_matrix(num_predictors:int, time_series:pd.Series) -> np.ndarray:
    """
    Function used to transform the pandas Series to a matrix.
    Used to feed classifiers.

    Args:
        num_predictors (int): numbers of predictors
        time_series (pd.Series): time series data

    Returns:
        np.ndarray: NxM matrix where:
        - N is the number of examples
        - M is the number of predictors
    """
    a = np.zeros(shape=(len(time_series), num_predictors))
    for i in range(len(time_series)):
        for j in range(num_predictors):
            a[i,j] = time_series.iloc[i][j]
    return a

# compute_mean_feature_vector used to compute baseline
def compute_mean_feature_vector(time_series:pd.Series) -> np.ndarray:
    """
    Compute the mean of each field of each time series example

    Args:
        time_series (pd.Series): time series

    Returns:
        np.ndarray: feature vector for each entry
    """
    num_predictors = time_series.iloc[0].shape[-1]
    return from_series_to_matrix(num_predictors, time_series.apply(lambda x: np.mean(x, axis=0)))

#interpolate the dataframe
def interpolate_time_series(x:np.ndarray, n_new_coords:int) -> np.ndarray:
    """
    Function used to interpolate a time series.
    The resulting time series will have n_new_coords points.

    Args:
        x (np.ndarray): Time series in matrix form. 
        - Rows: time instants
        - Columns: predictors (features)
        n_new_coords (int): desired number of time instants

    Returns:
        np.ndarray: New time series in matrix form. The rows are now n_new_coords. Columns stay still.
    """
    n_old_coords, n_predictors = x.shape
    x_new = np.zeros((n_new_coords, n_predictors))
    for i in range(n_predictors):
        x_new[:, i] = np.interp(np.linspace(0, n_old_coords, num=n_new_coords), np.array(list(range(n_old_coords))), x[:, i])
    return x_new

def interpolate_data(X:pd.Series, n_new_coords:int) -> pd.Series:
    """
    Apply interpolation to the passed pandas Series

    Args:
        X (pd.Series): pandas Series, each row contain a time series
        n_new_coords (int): desired number of time instants

    Returns:
        pd.Series: the pandas Series containing interpolated time series
    """
    X_new = X.apply(lambda x : interpolate_time_series(x, n_new_coords))
    return X_new

def concatenate_examples(X:pd.Series, by_row:bool=True) -> np.ndarray:
    """
    It returns the matrix containing the final features vector for each sample (row).
    Each final features vector is the corresponding horizontally stacked time series.
    

    Args:
        X (pd.Series): input pandas Series containing the samples.
        Each sample is a time series that has been interpolated.

    Returns:
        np.ndarray: a matrix where each row is a feature vector.
        Each row represents a sample.
    """
    new_x = np.zeros((len(X), (X.iloc[0].shape[1]*X.iloc[0].shape[0])))
    if by_row:
        for i in range(len(X)):
            new_x[i] = X.iloc[i].flatten()
    else:
        for i in range(len(X)):
            new_x[i] = X.iloc[i].flatten(order='F')
    return new_x

def fill_return_array(longest_series_shape:Tuple, time_series:pd.Series, flag_value:int=10000) -> np.array:
    """
    Fill the time_series matrix with nan to match the longest_series_shape and return it as an array

    Args:
        longest_series_shape (Tuple): Maximum length time series.
        time_series (pd.Series): Time series to fill.

    Returns:
        np.array: The new time series.
    """
    new_series = np.full(longest_series_shape, flag_value, dtype=np.float64)
    new_series.ravel()[:time_series.size] = time_series.ravel()
    return new_series.ravel()

def restore_time_series(X:np.array, flag_value:int=10000) -> np.array:
    """
    DEPRECATED since tslearn was used!
    Restore the time series by removing the flag_value
    and reshaping the time series.

    Args:
        X (np.array): Time series.
        flag_value (int, optional): Flag value used to fill the time series. Defaults to 10000.

    Returns:
        np.array: The restored time series.
    """
    idx = -1
    for i in range(len(X)):
        if X[i] == flag_value:
            idx = i
            break
    if idx == -1:
        return X.reshape((-1,22))    
    
    return X[:idx].reshape((-1,22))