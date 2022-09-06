import pandas as pd
import numpy as np

def extract_matrix(time_series_data:list) -> np.array:
    """extract the matrix of the time series passed!

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

def compute_S_matrix(ts_series:pd.Series, means:np.array, vars:np.array) -> tuple:
    """function to compute the S matrix of shape n x N (n = number of predictors, N = number of examples). 
    Such matrix will be used to compute
    the weight vector needed by Eros norm

    Args:
        -ts_series (pd.Series): Series containing the dataset of time series.
        Each entry is a list of vectors. 
        Each vector is a component of the i-th time series
        -means (np.array): array containing the means of the features in order to scale them
        -vars (np.array): array containing the vars of the features in order to scale them

    Returns:
        tuple[np.array, list]: returns the matrix S and the list of
        right eigenvectors matrices computed for each time series
    """
    s_matrix = np.zeros(shape=(len(ts_series), ts_series.iloc[0].shape[-1]))
    v_list = [] # list of right eigenvector matrix
    for i in range(len(ts_series)):
        ts = ts_series.iloc[i] # time x predictors
        #The matrix S will be nxN where n is the predictor dimension and N is the number of time-series examples.
        #Hence, we will use the transpose to compute the covariance matrix.
        ts = (ts - means)/vars
        ts = ts.T # predictors x time
        #Compute the covariance matrix of the i-th example of the dataset
        #cov_ts = np.corrcoef(ts)
        cov_ts = np.cov(ts)
        # Compute the SVD of the covariance matrix
        u, s, v_t = np.linalg.svd(cov_ts)
        s_matrix[i] = s
        v_list.append(v_t.T)
    return s_matrix.T, v_list

def compute_weight_vector(S:np.ndarray, aggregation:str='mean', algorithm:int=1) -> np.array:
    """compute the weight vector used in the computation of Eros norm

    Args:
        S (np.ndarray): matrix containing eigenvalues of each predictor
        aggregation (str, optional): aggregation function to use. Defaults to 'mean'.
        algorithm(int): choose the algorithm to use to compute weight vector.
        - Algorithm 1: do not normalize rows of the S matrix. Perform directly the computation of w
        - Algorithm 2: first normalize rows of the S matrix and then compute w.
    Returns:
        np.array: return the normalized weight vector
    """
    n = S.shape[0] # number of predictors
    if (algorithm == 2):
        # first normalize each eigenvalues
        S = S/np.sum(S, axis=0)
    if (aggregation == 'mean'):
        w = np.mean(S, axis=-1)
    elif (aggregation == 'min'):
        w = np.min(S, axis=-1)
    elif (aggregation == 'max'):
        w = np.max(S, axis=-1)
    return w/np.sum(w)

def eros_norm(weight_vector:np.array, A:np.array, B:np.array):
    """compute eros norm

    Args:
        weight_vector (np.array): weight vector
        A (np.array): time_series_1
        B (np.array): time_series_2

    Returns:
        float: distance between the 2 time series. Bounded in (0,1]
    """
    # since we want to use a_i and b_i which 
    # are the orthonormal column vectors of A and B,
    # we decide to transpose A and B
    A = A.T
    B = B.T
    n = A.shape[0] # number of predictors

    
    eros = 0
    
    for i in range(n):
        eros += weight_vector[i]*np.abs(np.dot(A[i], B[i]))
    return eros

def compute_kernel_matrix(num_examples:int, weight_vector:np.array, v_list:list) -> np.array:
    """compute the kernel matrix to be used in PCA

    Args:
        num_examples (int): number of examples in the dataset
        weight_vector (np.array): weight vector 
        v_t_list (list[np.array]): list of right eigenvector matrices

    Returns:
        np.array: kernel matrix with pairwise eros norm
    """
    N = num_examples
    K_eros = np.zeros(shape=(N,N))

    for i in range(N):
        j = 0
        while (j <= i):
            K_eros[i,j] = eros_norm(weight_vector, v_list[i], v_list[j])
            if (i != j): 
                K_eros[j,i] = K_eros[i,j]
            j += 1

    # check whether the kernel matrix is positive semi definite (PSD) or not
    is_psd = np.all(np.linalg.eigvals(K_eros) >= 0)
    #is_psd = True
    print(np.min(np.linalg.eigvals(K_eros)))
    threshold = 1e-10
    # if not PSD, add to the diagonal the minimal value among eigenvalues of K_eros
    if is_psd == False:
        delta = np.min(np.linalg.eigvals(K_eros))
        delta_ary = [np.abs(delta) + threshold for _ in range(K_eros.shape[0])]
        K_eros += np.diag(delta_ary)
    is_psd = np.all(np.linalg.eigvals(K_eros) >= 0)
    if is_psd == True:
        print("now PSD")
    else:
        print("not PSD")
    return K_eros

def perform_PCA(num_examples:int, weight_vector:np.array, v_list:list) -> tuple:
    """extract principal components in the feature space

    Args:
        num_examples (int): number of examples in the dataset
        weight_vector (np.array): weight vector 
        v_t_list (list[np.array]): list of right eigenvector matrices

    Returns:
        tuple[np.ndarray, np.array]:
        - K_eros matrix
        - eigenvectors (principal components) of the feature space
    """
    K_eros = compute_kernel_matrix(num_examples, weight_vector, v_list)
    O = np.ones(shape=(num_examples,num_examples))
    O *= 1/num_examples
    K_eros_mc = K_eros - O@K_eros - K_eros@O + O@K_eros@O # K_eros mean centered
    is_psd = np.all(np.linalg.eigvals(K_eros_mc) >= 0)
    print(f"K eros mean centered is {'not ' if not is_psd else ''}PSD")
    
    ####### added #######
    threshold = 10e-10
    if is_psd == False:
        delta = np.min(np.linalg.eigvals(K_eros_mc))
        delta_ary = [np.abs(delta) + threshold for _ in range(K_eros_mc.shape[0])]
        K_eros_mc += np.diag(delta_ary)
    is_psd = np.all(np.linalg.eigvals(K_eros_mc) >= 0)
    print(f"K eros mean centered is {'not ' if not is_psd else ''}PSD")
    ####### added #######
    
    
    eig_vals, eig_vecs = np.linalg.eig(K_eros_mc)
    #return K_eros, eig_vecs, eig_vals
    
    ####### added #######
    return K_eros_mc, eig_vecs, eig_vals
    ####### added #######
     

def project_test_data(num_training_examples:int, num_test_examples:int, weight_vector:np.array, v_list_train:list, v_list_test:list, K_eros_train:np.ndarray, V:np.ndarray) -> tuple:
    """compute the K eros test kernel matrix used to project test data

    Args:
        num_examples_train (int): number of examples in the training dataset
        num_examples_test (int): number of examples in the test dataset
        weight_vector (np.array): weight vector 
        v_list_train (list[np.array]): list of right eigenvector matrices of the training dataset
        v_list_test (list[np.array]): list of right eigenvector matrices of the test dataset

    Returns:
        np.array: kernel matrix with pairwise eros norm
    """
    N_train = num_training_examples
    N_test = num_test_examples
    K_eros_test = np.zeros(shape=(N_test,N_train))

    for i in range(N_test):
        for j in range(N_train):
            K_eros_test[i,j] = eros_norm(weight_vector, v_list_test[i], v_list_train[j])
    
    O_test = np.ones(shape=(N_test, N_train))*(1/N_train)
    O_train = np.ones(shape=(N_train, N_train))*(1/N_train)

    K_eros_test_mc = K_eros_test - O_test@K_eros_train - K_eros_test@O_train + O_test@K_eros_train@O_train

    Y = K_eros_test_mc @ V
    
    # return Y, K_eros_test
    return Y, K_eros_test_mc

# each row of the Series object is an array. Classifiers won't read it. We create a matrix of values.
def from_series_to_matrix(num_predictors:int, time_series:pd.Series) -> np.ndarray:
    """Function used to transform the pandas Series to a matrix.
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
    """Compute the mean of each field of each time series example

    Args:
        time_series (pd.Series): time series

    Returns:
        np.ndarray: feature vector for each entry
    """
    num_predictors = time_series.iloc[0].shape[-1]
    return from_series_to_matrix(num_predictors, time_series.apply(lambda x: np.mean(x, axis=0)))

#interpolate the dataframe
def interpolate_time_series(x:np.ndarray, n_new_coords:int)->np.ndarray:
    n_old_coords, n_predictors = x.shape
    x_new = np.zeros((n_new_coords, n_predictors))
    for i in range(n_predictors):
        x_new[:, i] = np.interp(np.linspace(0, n_old_coords, num=n_new_coords), np.array(list(range(n_old_coords))), x[:, i])
    return x_new

def interpolate_data(X:pd.Series, n_new_coords:int)->pd.Series:
    X_new = X.apply(lambda x : interpolate_time_series(x, n_new_coords))
    return X_new

def concatenate_examples(X:pd.Series)->np.ndarray:
    new_x = np.zeros((len(X), (X.iloc[0].shape[1]*X.iloc[0].shape[0])))
    for i in range(len(X)):
        new_x[i] = X.iloc[i].flatten()
    return new_x

