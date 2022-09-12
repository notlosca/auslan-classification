import pandas as pd
import numpy as np
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

def compute_S_matrix(ts_series:pd.Series, means:np.array, vars:np.array) -> tuple:
    """
    Function to compute the S matrix of shape n x N (n = number of predictors, N = number of examples). 
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
    """
    Compute the weight vector used in the computation of Eros norm

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
    """
    Compute eros norm

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
    """
    Compute the kernel matrix to be used in PCA

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
    """
    Extract principal components in the feature space

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
    """
    Compute the K eros test kernel matrix used to project test data

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

def concatenate_examples(X:pd.Series) -> np.ndarray:
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
    for i in range(len(X)):
        new_x[i] = X.iloc[i].flatten()
    return new_x

def fill_nan_return_array(longest_series_shape:Tuple, time_series:pd.Series) -> np.array:
    """
    Fill the time_series matrix with nan to match the longest_series_shape and return it as an array

    Args:
        longest_series_shape (Tuple): Maximum length time series.
        time_series (pd.Series): Time series to fill.

    Returns:
        np.array: The new time series.
    """
    new_series = np.full(longest_series_shape, 10000)
    new_series.ravel()[:time_series.size] = time_series.ravel()
    return new_series.ravel()

def restore_time_series(X:np.array, flag_value:int=10000) -> np.array:
    """
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

# DTW stack overflow/wikipedia
from scipy.spatial import distance

def DTW(A:np.array, B: np.array) -> float:
    """
    Compute the DTW score between 2 time series.

    Args:
        A (np.array): First time series.
        B (np.array): Second time series.

    Returns:
        float: DTW score.
    """
    new_a = restore_time_series(A)
    new_b = restore_time_series(B)
    
    len_a = new_a.shape[0]
    len_b = new_b.shape[0]
    
    pointwise_distance = distance.cdist(new_a, new_b, metric='sqeuclidean')
    cumdist = np.matrix(np.ones((len_a+1,len_b+1)) * np.inf)
    cumdist[0,0] = 0

    for id_a in range(len_a):
        for id_b in range(len_b):
            minimum_cost = np.min([cumdist[id_a, id_b+1],
                                   cumdist[id_a+1, id_b],
                                   cumdist[id_a, id_b]])
            cumdist[id_a+1, id_b+1] = pointwise_distance[id_a,id_b] + minimum_cost

    return cumdist[len_a, len_b]


#-------------#
# DTW implementation from scratch based on graphs. Not sure

import networkx as nx
from itertools import combinations
from typing import List, Tuple

def euclidean_distance(x:np.array, y:np.array, return_squared:bool=False) -> float:
    """
    Compute the Euclidean distance between 2 arrays: x, y

    Args:
        x (np.array): First array
        y (np.array): Second array
        squared (bool, optional): Whether to return the squared form or not. Defaults to False.

    Returns:
        float: The computed Euclidean distance
    """
    if return_squared:
        return np.sum(np.square(x-y))
    return np.sqrt(np.sum(np.square(x-y)))

def compute_cost_matrix(X:np.ndarray, Y:np.ndarray, f:str='squared_euclidean') -> np.ndarray:
    """
    Compute the cost matrix between two time series.
    If they differ in lengths, the matrix will be m x n, otherwise n x n.
    The two time series must have the same number of features (predictors).

    Args:
        X (np.ndarray): First time series of the form time instants x features.
        Y (np.ndarray): Second time series of the form time instants x features.

    Returns:
        np.ndarray: The cost matrix
    """
    n = X.shape[0]
    m = Y.shape[0]
    C = np.zeros(shape=(n,m))
    
    for i in range(n):
        for j in range(m):
            
            if f == 'squared_euclidean':
                C[i,j] = euclidean_distance(X[i,:], Y[j,:], return_squared=True)
            elif f == 'absolute_difference':
                C[i,j] = np.abs(np.sum(X[i,:] - Y[j,:]))
    
    return C

def compute_dtw(X:np.ndarray,Y:np.ndarray, show_graph:bool=False) -> Tuple[float, List[Tuple]]:
    """
    It computes the DTW score between two multivariate time series X and Y
    of the form [time_instants, num_features]

    Args:
        X (np.ndarray): First time series
        Y (np.ndarray): Second time series
        show_graph (bool, optional): Whether to show the graph or not. Defaults to False, 
        since graphs are very large.

    Returns:
        Tuple[float, List[Tuple]]: DTW score, shortest path. The latter is the sequence of nodes
    """
    n = X.shape[0]
    m = Y.shape[0]
    G, cost_matrix = construct_graph(X,Y,draw=show_graph)
    sh_path = nx.shortest_path(G, source=(0,0), target=(n-1,m-1), weight='weight')
    DTW_score = cost_matrix[(0,0)] + nx.path_weight(G, sh_path, weight='weight')
    # print(f"The shortest path is {sh_path} with DTW score of {DTW_score}")
    return DTW_score, sh_path
    
def construct_graph(X:np.ndarray,Y:np.ndarray, allowed_transitions:List[Tuple]=[(0,1), (1,0), (1,1)], draw:bool=False) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    It constructs the graph upon which we search the path with minimum weight.
    The graph is constructed in the form of the cost matrix C. This will be used to compute DTW score.
    Only allowed_transitions are considered. Hence there won't be all the feasible edges.
    Args:
        X (np.ndarray): First time series of the form [time_instants, num_predictors].
        Y (np.ndarray): Second time series of the form [time_instants, num_predictors].
        allowed_transitions (List[Tuple], optional): List of feasible edges. Defaults to [(0,1), (1,0), (1,1)] to compute DTW.
        draw (bool, optional): Whether to draw or not the graph. Defaults to False.

    Returns:
        Tuple[nx.DiGraph, np.ndarray]: The corresponding graph and the cost matrix.
        The latter has the form [time_instants_of_X, time_instants_of_Y]
    """
    
    n = X.shape[0]
    m = Y.shape[0]
    
    nodes = []
    for i in range(n):
        for j in range(m):
            nodes.append( (i,j) )
            
    edges = compute_allowed_edges(nodes, allowed_transitions)
    C = compute_cost_matrix(X, Y, f='squared_euclidean')
    weighted_edges = weight_edges(edges, C)
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(weighted_edges)
    if draw:
        pos = {i:(i[1], -i[0]) for i in G.nodes}
        nx.draw(G, pos, with_labels=True)
    return G, C
    
def compute_allowed_edges(nodes:List[Tuple], allowed_combinations:list=[(0,1), (1,0), (1,1)]) -> List[Tuple]:
    """
    Compute allowed edges from all possible combination of 2 nodes following the allowed set of combinations.

    Args:
        nodes (List[Tuple]): List of nodes from which we compute all possible combinations.
        allowed_combinations (list, optional): Set of allowed transitions. Defaults to [(0,1), (1,0), (1,1)].

    Returns:
        List[Tuple]: List of edges following the allowed_combinations list.
    """
    combs = list(combinations(nodes, 2))

    allowed_edges = []
    illegal_edges = []

    for combination in combs:
        c0 = combination[0]
        c1 = combination[1]
        res = (c1[0]-c0[0], c1[1]-c0[1])
        if res in allowed_combinations:
            allowed_edges.append(combination)
        else:
            illegal_edges.append(combination)
    return allowed_edges

def weight_edges(edges:List[Tuple], cost_matrix:np.ndarray) -> List[Tuple]:
    """
    Weight each edge with the corresponding transition cost from one node to another

    Args:
        edges (List[Tuple]): Set of edges.
        cost_matrix (np.ndarray): Cost matrix C

    Returns:
        List[Tuple]: Return the set of edges with the new attribute 'weight'
    """
    weighted_edges = []
    for i in edges:
        s = i[0] # source node
        t = i[1] # target node
        weighted_edges.append( (*i, cost_matrix[t]) )
    return weighted_edges
#-------------#