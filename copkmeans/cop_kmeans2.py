#!/usr/bin/env python
"""
cop_kmeans.py

An optimized implementation of COP-KMeans that supports must-link and cannot-link constraints.
This version uses SciPy’s optimized distance routines (via cdist) and supports both Euclidean and cosine distance metrics.

Usage:
    Run the file directly to execute a simple test.
"""

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

def normalize_vectors(arr):
    """
    Normalize each row vector in a 2D numpy array.
    
    Parameters:
        arr (np.array): Array of shape (n, d)
    
    Returns:
        np.array: The row-normalized array.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms

def tolerance(tol, dataset):
    """
    Compute a tolerance value based on the dataset's variance.
    
    Parameters:
        tol (float): The tolerance factor.
        dataset (np.array): Array of shape (n, d)
    
    Returns:
        float: Tolerance value.
    """
    n, d = dataset.shape
    averages = np.mean(dataset, axis=0)
    variances = np.mean((dataset - averages) ** 2, axis=0)
    return tol * np.sum(variances) / d

def closest_clusters(centers, datapoint, metric):
    """
    Compute the distances from a datapoint to each center using SciPy's cdist,
    then return the sorted center indices (closest first).
    
    Parameters:
        centers (np.array): Array of centers with shape (k, d)
        datapoint (np.array): A 1D vector of length d.
        metric (str): Distance metric ('euclidean' or 'cosine').
    
    Returns:
        indices (list): Sorted list of center indices (closest first)
        distances (list): Corresponding distances.
    """
    # cdist requires 2D arrays, so reshape datapoint to (1, d)
    distances = cdist(centers, datapoint.reshape(1, -1), metric=metric).flatten()
    indices = np.argsort(distances)
    return indices.tolist(), distances.tolist()

def initialize_centers(dataset, k, method):
    """
    Initialize centers from the dataset using either 'random' or 'kmpp' (k-means++).
    
    Parameters:
        dataset (np.array): Array of shape (n, d)
        k (int): Number of centers
        method (str): 'random' or 'kmpp'
    
    Returns:
        np.array: Array of shape (k, d) containing the initial centers.
    """
    n = dataset.shape[0]
    if method == 'random':
        indices = np.random.choice(n, size=k, replace=False)
        return dataset[indices]
    elif method == 'kmpp':
        centers = []
        idx = np.random.randint(n)
        centers.append(dataset[idx])
        # D holds the squared distances from each point to its nearest center so far.
        D = np.sum((dataset - centers[0]) ** 2, axis=1)
        for _ in range(1, k):
            total = D.sum()
            if total == 0:
                idx = np.random.randint(n)
            else:
                probs = D / total
                r = random.random()
                cumulative = np.cumsum(probs)
                idx = np.searchsorted(cumulative, r)
            centers.append(dataset[idx])
            new_distances = np.sum((dataset - dataset[idx]) ** 2, axis=1)
            D = np.minimum(D, new_distances)
        return np.array(centers)
    else:
        raise ValueError("Unknown initialization method: " + method)

def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    """
    Check if assigning the data point at data_index to cluster_index violates any constraints.
    
    Parameters:
        data_index (int): Index of the current data point.
        cluster_index (int): Proposed cluster assignment.
        clusters (list): Current cluster assignments (with -1 for unassigned).
        ml (dict): Must-link constraint graph (mapping data point index to a set of indices).
        cl (dict): Cannot-link constraint graph (mapping data point index to a set of indices).
    
    Returns:
        bool: True if any constraint is violated, else False.
    """
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True
    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True
    return False

def compute_centers(clusters, dataset, k, ml_info, distance_metric='euclidean'):
    """
    Recompute cluster centers given the current assignments. If the number of unique clusters
    is less than k, extra centers are added using must-link group information.
    
    Parameters:
        clusters (list): Cluster assignment for each data point.
        dataset (np.array): Data array of shape (n, d)
        k (int): Desired number of clusters.
        ml_info (tuple): (ml_groups, ml_scores, ml_centroids) computed by get_ml_info.
        distance_metric (str): 'euclidean' or 'cosine'
    
    Returns:
        (list, list): Updated cluster assignments and centers.
    """
    clusters = np.array(clusters)
    unique_clusters = np.unique(clusters)
    k_new = len(unique_clusters)
    # Remap cluster labels to 0,1,...,k_new-1
    id_map = {old: new for new, old in enumerate(unique_clusters)}
    clusters = np.array([id_map[x] for x in clusters])
    d = dataset.shape[1]
    centers = np.zeros((k, d))
    for new_cluster in range(k_new):
        indices = np.where(clusters == new_cluster)[0]
        if len(indices) > 0:
            center = np.mean(dataset[indices], axis=0)
            if distance_metric == 'cosine':
                norm_val = np.linalg.norm(center)
                if norm_val != 0:
                    center = center / norm_val
            centers[new_cluster] = center
    # If fewer than k clusters, supplement using must-link group centroids.
    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = []
        for group in ml_groups:
            score = 0.0
            for i in group:
                c = centers[clusters[i]]
                score += np.sum((dataset[i] - c) ** 2)
            current_scores.append(score)
        current_scores = np.array(current_scores)
        ml_scores = np.array(ml_scores)
        group_ids = np.argsort(current_scores - ml_scores)[::-1]
        for j in range(k - k_new):
            gid = group_ids[j]
            cid = k_new + j
            center = np.array(ml_centroids[gid])
            if distance_metric == 'cosine':
                norm_val = np.linalg.norm(center)
                if norm_val != 0:
                    center = center / norm_val
            centers[cid] = center
            for i in ml_groups[gid]:
                clusters[i] = cid
    return clusters.tolist(), centers.tolist()

def get_ml_info(ml, dataset):
    """
    Compute must-link group information, including groups, their scores, and centroids.
    
    Parameters:
        ml (dict): Must-link constraint graph.
        dataset (np.array): Array of shape (n, d)
    
    Returns:
        tuple: (groups, scores, centroids)
            - groups: List of lists of indices in each must-link group.
            - scores: Sum of squared distances (score) for each group.
            - centroids: Centroid of each group.
    """
    n = len(dataset)
    flags = [True] * n
    groups = []
    for i in range(n):
        if not flags[i]:
            continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False
    d = dataset.shape[1]
    centroids = []
    scores = []
    for group in groups:
        group_points = dataset[group]
        centroid = np.mean(group_points, axis=0)
        centroids.append(centroid.tolist())
        score = np.sum(np.sum((group_points - centroid) ** 2, axis=1))
        scores.append(score)
    return groups, scores, centroids

def transitive_closure(ml, cl, n):
    """
    Compute the transitive closure of must-link (ml) and cannot-link (cl) constraints.
    If an inconsistency is detected, an exception is raised.
    
    Parameters:
        ml (list of tuples): Must-link pairs.
        cl (list of tuples): Cannot-link pairs.
        n (int): Number of data points.
    
    Returns:
        (dict, dict): (ml_graph, cl_graph) as dictionaries mapping each index to a set of indices.
    """
    ml_graph = {i: set() for i in range(n)}
    cl_graph = {i: set() for i in range(n)}

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x in component:
                ml_graph[x].update(set(component) - {x})

    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception(f"Inconsistent constraints between {i} and {j}")
    return ml_graph, cl_graph

def cop_kmeans(dataset, k, ml=[], cl=[],
               initialization='kmpp', max_iter=300, tol=1e-4,
               distance_metric='cosine'):
    """
    Constrained K-Means (COP-KMeans) clustering with must-link and cannot-link constraints.
    Uses SciPy’s optimized distance computations and supports both Euclidean and cosine metrics.
    
    Parameters:
        dataset (list of lists): The dataset (each element is a d-dimensional vector).
        k (int): Desired number of clusters.
        ml (list of tuples): List of must-link pairs (each tuple is (i, j)).
        cl (list of tuples): List of cannot-link pairs (each tuple is (i, j)).
        initialization (str): Initialization method ('random' or 'kmpp').
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        distance_metric (str): 'euclidean' or 'cosine'.
    
    Returns:
        (list, list): A tuple containing the cluster assignments (list of length n) and centers (list of k centers).
        If no valid clustering is found, returns (None, None).
    """
    dataset_np = np.array(dataset, dtype=float)
    if distance_metric == 'cosine':
        dataset_np = normalize_vectors(dataset_np)
    n = len(dataset_np)
    # Compute the transitive closure of the constraints.
    ml_graph, cl_graph = transitive_closure(ml, cl, n)
    ml_info = get_ml_info(ml_graph, dataset_np)
    tol_val = tolerance(tol, dataset_np)
    # Initialize centers.
    centers = initialize_centers(dataset_np, k, initialization)
    if distance_metric == 'cosine':
        centers = normalize_vectors(centers)
    centers = np.array(centers)
    
    for iteration in range(max_iter):
        clusters_assignment = [-1] * n
        # Assign each point to a cluster.
        for i, point in enumerate(dataset_np):
            if clusters_assignment[i] != -1:
                continue
            indices, _ = closest_clusters(centers, point, metric=distance_metric)
            found = False
            for idx in indices:
                if not violate_constraints(i, idx, clusters_assignment, ml_graph, cl_graph):
                    found = True
                    clusters_assignment[i] = idx
                    # Propagate assignment along must-link constraints.
                    for j in ml_graph[i]:
                        clusters_assignment[j] = idx
                    break
            if not found:
                return None, None  # No valid clustering could be found.
        # Recompute centers based on current assignment.
        clusters_assignment, centers_list = compute_centers(clusters_assignment, dataset_np, k, ml_info, distance_metric=distance_metric)
        new_centers = np.array(centers_list)
        if distance_metric == 'cosine':
            new_centers = normalize_vectors(new_centers)
        shift = np.sum((centers - new_centers) ** 2)
        if shift <= tol_val:
            break
        centers = new_centers
    return clusters_assignment, centers.tolist()

# -----------------------------------------------------------------------------
# Testing the implementation

if __name__ == '__main__':
    import time
    # Generate a random dataset of 100 points in 2D.
    dataset = [[random.random() for _ in range(2)] for _ in range(100)]
    ml = []  # No must-link constraints for this test.
    cl = []  # No cannot-link constraints for this test.
    
    print("Testing COP-KMeans with Euclidean distance:")
    clusters, centers = cop_kmeans(dataset, 5, ml, cl, distance_metric='euclidean')
    print("Clusters:", clusters)
    print("Centers:", centers)
    
    print("\nTesting COP-KMeans with Cosine distance:")
    clusters, centers = cop_kmeans(dataset, 5, ml, cl, distance_metric='cosine')
    print("Clusters:", clusters)
    print("Centers:", centers)
    
    print("\nDone.")