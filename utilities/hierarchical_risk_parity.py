import math
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from typing import Any, Callable, List
from utilities.covariance_utilities import covariance_to_correlation
from utilities.hierarchical_clustering import hierarchical_clustering
from utilities.principal_component_analysis import detone


def correlation_to_hrp_distance(correlation: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to the Lopez de Prado HRP distance matrix
    ``d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))``.

    The result is a proper metric (see lab notes Question 3) and is the input
    expected by the HRP clustering step.

    Parameters:
        correlation (np.ndarray): Correlation matrix.

    Returns:
        np.ndarray: Distance matrix with the same shape as ``correlation``.
    """
    correlation = np.clip(correlation, -1.0, 1.0)
    # Template had sqrt(2*(1+rho)) which is WRONG.
    # Correct formula from the assignment notes: d_ij = sqrt(0.5*(1 - rho_ij))
    return np.sqrt(0.5 * (1.0 - correlation))


def flatten_list(lst: List[Any]) -> List[Any]:
    """
    Recursively flatten a nested list into a single list of elements.

    Parameters:
        lst (List[Any]): The nested list.

    Returns:
        List[Any]: A flattened list of elements.
    """
    if isinstance(lst, list):
        return [item for sub_lst in lst for item in flatten_list(sub_lst)]
    else:
        return [lst]


def is_nested(lst: List[Any]) -> bool:
    """
    Check if a list is nested (contains other lists as elements).

    Parameters:
        lst (List[Any]): The list to check.

    Returns:
        bool: True if the list is nested, False otherwise.
    """
    return any(isinstance(element, list) for element in lst)


def list_recursive_bisection(
    lst: List[Any],
    labels: List[Any] | None = None,
    cur_iter: int = 0,
    max_iter: int | None = None,
    start: int = 0,
    end: int | None = None,
) -> List[Any]:
    """
    Perform a recursive bisection of a list.

    At each recursion level the list segment [start, end) is split in half.
    Recursion stops when a segment has one element or when ``cur_iter``
    reaches ``max_iter`` (controlling the number of clusters produced).

    Parameters:
        lst (List[Any]): The full leaf-order list (not sliced — we use indices).
        labels (List[Any] | None): Unused, kept for API compatibility.
        cur_iter (int): Current recursion depth (starts at 0).
        max_iter (int | None): Stop splitting after this many levels.
            None means full recursion down to individual leaves.
        start (int): Start index of the current segment.
        end (int | None): End index (exclusive) of the current segment.

    Returns:
        list: A nested list representing the recursive bisection.
    """
    if end is None:
        end = len(lst)

    n = end - start

    # Base case: single element
    if n <= 1:
        return [lst[start]] if n == 1 else []

    # Stop early if we've exhausted the iteration budget
    if max_iter is not None and cur_iter >= max_iter:
        return lst[start:end]

    mid = start + n // 2

    left_branch = list_recursive_bisection(
        lst=lst, cur_iter=cur_iter + 1, max_iter=max_iter, start=start, end=mid
    )
    right_branch = list_recursive_bisection(
        lst=lst, cur_iter=cur_iter + 1, max_iter=max_iter, start=mid, end=end
    )

    return [left_branch, right_branch]


def recursive_bisection(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Return the nested cluster structure from a linkage matrix performing a recursive bisection.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """
    iter_num = None if clusters_num is None else math.log(clusters_num, 2)
    if iter_num is not None:
        if not iter_num.is_integer():
            raise ValueError(
                "The number of clusters must be a power of 2 for recursive bisection."
            )
        else:
            iter_num = int(iter_num)

    leaves_lst = sch.leaves_list(linkage_matrix.values)
    leaves_labels = (
        list(leaves_lst) if labels is None else [labels[leaf] for leaf in leaves_lst]
    )

    return list_recursive_bisection(
        lst=leaves_labels, max_iter=int(iter_num) if iter_num is not None else None
    )


def dendrogram_iteration(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Convert a linkage matrix to nested clusters according to the dendrogram structure.

    Unlike recursive bisection, this traversal replays the actual merge sequence
    recorded in the linkage matrix, so that the resulting nested list reflects the
    true hierarchy rather than a mechanical halving of the leaf order.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """
    Z = linkage_matrix.values
    n_leaves = len(Z) + 1

    # Initialise: each leaf is its own cluster (wrapped in a list for nesting)
    if labels is not None:
        active_clusters = {i: [labels[i]] for i in range(n_leaves)}
    else:
        active_clusters = {i: [i] for i in range(n_leaves)}

    # How many merges to replay
    if clusters_num is None or clusters_num <= 1:
        merges_to_perform = len(Z)
    else:
        merges_to_perform = n_leaves - clusters_num

    for i in range(merges_to_perform):
        idx1 = int(Z[i, 0])
        idx2 = int(Z[i, 1])
        new_cluster_id = n_leaves + i
        active_clusters[new_cluster_id] = [active_clusters[idx1], active_clusters[idx2]]
        del active_clusters[idx1]
        del active_clusters[idx2]

    # If full merge, return the single nested structure
    if clusters_num is None or clusters_num <= 1:
        return list(active_clusters.values())[0]
    else:
        return list(active_clusters.values())


def top_down_allocation(
    nested_clusters: List[Any], covariance: pd.DataFrame, get_cluster_var: Callable
) -> pd.Series:
    """
    Top-down allocation of weights to the clusters following Lopez de Prado's
    recursive split: at each bisection ``alpha = sigma2_R / (sigma2_L + sigma2_R)``
    is allocated to the left cluster, ``1 - alpha`` to the right one (more capital
    to the lower-variance branch). The variance of each cluster is computed via
    ``get_cluster_var``.

    Parameters:
        nested_clusters (List[Any]): The nested cluster structure produced by
            ``recursive_bisection`` or ``dendrogram_iteration``.
        covariance (pd.DataFrame): Covariance matrix.
        get_cluster_var (Callable): Function returning the variance of a cluster
            given the covariance matrix and the list of asset labels in the
            cluster.

    Returns:
        pd.Series: Weights summing to 1, indexed on the asset labels.
    """
    weights = pd.Series(1.0, index=flatten_list(nested_clusters))
    if not is_nested(nested_clusters):
        return weights
    else:
        cluster1 = flatten_list(nested_clusters[0])
        cluster2 = flatten_list(nested_clusters[1])

        cluster1_var = get_cluster_var(covariance=covariance, cluster=cluster1)
        cluster2_var = get_cluster_var(covariance=covariance, cluster=cluster2)

        # Inverse volatility between clusters
        alpha1 = cluster2_var / (cluster1_var + cluster2_var)
        alpha2 = 1 - alpha1

        weights[cluster1] *= alpha1 * top_down_allocation(
            nested_clusters[0], covariance.loc[cluster1, cluster1], get_cluster_var
        )
        weights[cluster2] *= alpha2 * top_down_allocation(
            nested_clusters[1], covariance.loc[cluster2, cluster2], get_cluster_var
        )

        return weights


def get_cluster_var_via_iv(covariance: pd.DataFrame, cluster: List[Any]):
    """Compute cluster variance using inverse-variance parity on the diagonal."""
    covariance = covariance.loc[cluster, cluster]
    iv_weights = 1.0 / np.diag(covariance)
    iv_weights /= iv_weights.sum()
    iv_weights = iv_weights.reshape(-1, 1)

    return (iv_weights.T @ covariance.values @ iv_weights)[0, 0]


def hierarchical_risk_parity(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    cluster_traverser: Callable = dendrogram_iteration,
    get_cluster_var: Callable = get_cluster_var_via_iv,
    perform_detoning: bool = False,
    plot_dendrogram: bool = False,
) -> pd.Series:
    """
    Compute Hierarchical Risk Parity weights for a single rebalance.

    Pipeline:
        1. Convert ``covariance`` to a correlation matrix.
        2. Optionally detone the correlation matrix by removing its first
           principal component.
        3. Build a linkage matrix on the HRP distance ``sqrt(0.5(1 - rho))``
           using the chosen linkage method and (second-stage) distance metric.
        4. Convert the linkage to a nested cluster structure via
           ``cluster_traverser`` (e.g. ``recursive_bisection`` or
           ``dendrogram_iteration``).
        5. Allocate weights top-down with inverse-variance splits.

    Parameters:
        covariance (pd.DataFrame): Covariance matrix indexed by asset labels.
        linkage_method (str): Linkage method passed to ``scipy.linkage``
            (``single``, ``ward``, ``complete``, ``average``, ...).
        distance_metric (str): Metric used by ``scipy.linkage`` to compute
            distances between rows of the input matrix (defaults to
            ``euclidean``, following Lopez de Prado).
        cluster_traverser (Callable): Function mapping a linkage matrix to a
            nested cluster structure. Defaults to ``dendrogram_iteration``.
        get_cluster_var (Callable): Function returning the variance of a
            cluster. Defaults to inverse-variance parity on the diagonal.
        perform_detoning (bool): Remove the first principal component from the
            correlation matrix before clustering. Defaults to False.
        plot_dendrogram (bool): Whether to plot the dendrogram while building it.

    Returns:
        pd.Series: HRP weights summing to 1, indexed on the asset labels.
    """
    correlation = covariance_to_correlation(covariance=covariance.values)
    if perform_detoning:
        correlation = detone(corr_matrix=correlation, components_num=1)

    distance = correlation_to_hrp_distance(correlation)
    linkage_matrix = hierarchical_clustering(
        matrix=pd.DataFrame(
            distance, index=covariance.index, columns=covariance.columns
        ),
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        plot_dendrogram=plot_dendrogram,
    )

    nested_clusters = cluster_traverser(
        linkage_matrix, labels=covariance.index.tolist()
    )

    return top_down_allocation(
        nested_clusters=nested_clusters,
        covariance=covariance,
        get_cluster_var=get_cluster_var,
    )
