import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy.spatial import distance
from utils import base

def DTW(A:np.array, B: np.array) -> float:
    """
    Compute the DTW score between 2 time series.

    Args:
        A (np.array): First time series.
        B (np.array): Second time series.

    Returns:
        float: DTW score.
    """
    new_a = base.restore_time_series(A)
    new_b = base.restore_time_series(B)
    
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
# DTW implementation from scratch based on graphs. Not efficient.

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

def compute_dtw(X:np.ndarray,Y:np.ndarray, allowed_transitions:List[Tuple]=[(0,1), (1,0), (1,1)], show_graph:bool=False) -> Tuple[nx.DiGraph, np.ndarray, float, List[Tuple]]:
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
    G, cost_matrix = construct_graph(X,Y,allowed_transitions=allowed_transitions,draw=show_graph)
    sh_path = nx.shortest_path(G, source=(0,0), target=(n-1,m-1), weight='weight')
    DTW_score = cost_matrix[(0,0)] + nx.path_weight(G, sh_path, weight='weight')
    # print(f"The shortest path is {sh_path} with DTW score of {DTW_score}")
    return G, cost_matrix, DTW_score, sh_path
    
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