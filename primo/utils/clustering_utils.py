#################################################################################
# PRIMO - The P&A Project Optimizer was produced by the National Energy
# Technology Laboratory (NETL).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# Standard libs
import logging
from itertools import combinations
from typing import List, Optional, Tuple, Union

# Installed libs
import networkx as nx
import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import BallTree

# User-defined libs
from primo.data_parser.default_data import EARTH_RADIUS
from primo.data_parser.well_data import WellData
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def distance_matrix(
    wd: WellData, weights: dict, list_wells: Optional[list] = None
) -> pd.DataFrame:
    """
    Generate a distance matrix based on the given features and
    associated weights for each pair of the given well candidates.

    Parameters
    ----------
    wd : WellData
        WellData object

    weights : dict
        Weights assigned to the features---distance, age, and
        depth when performing the clustering.

    list_wells : list, default = None
        If specified, returns the distance matrix only for the
        specified subset of wells

    Returns
    -------
    pd.DataFrame
        Distance matrix to be used for the agglomerative
        clustering method

    Raises
    ------
    ValueError
        1. if a spurious feature's weight is included apart from
            distance, age, and depth.
        2. if the sum of feature weights does not equal 1.
    """

    # If a feature is not provided, then set its weight to zero
    wt_dist = weights.pop("distance", 0)
    wt_age = weights.pop("age", 0)
    wt_depth = weights.pop("depth", 0)

    if len(weights) > 0:
        msg = (
            f"Received feature(s) {[*weights.keys()]} that are not "
            f"supported in the clustering step."
        )
        raise_exception(msg, ValueError)

    if not np.isclose(wt_dist + wt_depth + wt_age, 1, rtol=0.001):
        raise_exception("Feature weights do not add up to 1.", ValueError)

    # Construct the matrices only if the weights are non-zero
    # Converting list_wells to a list to handle non-list instances,
    # such as Pyomo Set, tuple, etc.
    data = wd.data if list_wells is None else wd.data.loc[list(list_wells)]
    cn = wd.column_names  # Column names

    coordinates = list(zip(data[cn.latitude], data[cn.longitude]))
    dist_matrix = wt_dist * (
        haversine_vector(coordinates, coordinates, unit=Unit.MILES, comb=True)
        if wt_dist > 0
        else 0
    )

    # Modifying the object in-place to save memory for large datasets
    dist_matrix += wt_age * (
        np.abs(np.subtract.outer(data[cn.age].to_numpy(), data[cn.age].to_numpy()))
        if wt_age > 0
        else 0
    )

    dist_matrix += wt_depth * (
        np.abs(np.subtract.outer(data[cn.depth].to_numpy(), data[cn.depth].to_numpy()))
        if wt_depth > 0
        else 0
    )

    return pd.DataFrame(dist_matrix, columns=data.index, index=data.index)


def build_ball_tree(
    lat_lon_pairs: List[Tuple[float, float]],
) -> Tuple[BallTree, np.ndarray]:
    """
    Build a Ball-Tree from a list of (latitude, longitude) pairs using the Haversine metric.

    Parameters
    ----------
    lat_lon_pairs : List[Tuple[float, float]]
        List of geographic coordinates (latitude, longitude) in degrees.

    Returns
    -------
    Tuple[Ball-Tree, np.ndarray]
        A tuple of the Ball-Tree object and the corresponding coordinates in radians.
    """
    coordinates_rad = np.radians(lat_lon_pairs)
    tree = BallTree(coordinates_rad, metric="haversine")
    return tree, coordinates_rad


def _well_clusters(wd: WellData) -> dict:
    """
    Returns well clusters

    Parameters
    ----------
    wd : WellData
        The WellData object to return well clusters.

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """
    col_names = wd.col_names
    return (
        wd.data.groupby(wd[col_names.cluster])
        .apply(lambda group: group.index.tolist(), include_groups=False)
        .to_dict()
    )


def check_existing_cluster(wd: WellData):
    """
    Checks if clustering has already been performed on the WellData object.

    Parameters
    ----------
    wd : WellData
        The WellData object to check for existing clusters.

    Returns
    -------
    Bool
        True if clustering exists, otherwise False.
    """
    if hasattr(wd.col_names, "cluster"):
        # Clustering has already been performed, so return the number of clusters.
        LOGGER.warning(
            "Found cluster attribute in the WellDataColumnNames object. "
            "Assuming that the data is already clustered. If the corresponding "
            "column does not correspond to clustering information, please use a "
            "different name for the attribute cluster while instantiating the "
            "WellDataColumnNames object."
        )
        return True
    return False


def perform_agglomerative_clustering(
    wd: WellData, threshold_distance: float = 10.0, list_wells: Optional[list] = None
):
    """
    Partitions the data into smaller clusters.

    Parameters
    ----------
    wd : WellData
        Object containing the information on all wells

    threshold_distance : float, default = 10.0
        Threshold distance for breaking clusters

    list_wells : list, default = None
        If specified, performs clustering only for the specified subset of wells

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """

    if check_existing_cluster(wd):
        return _well_clusters(wd)

    # Hard-coding the weights data since this should not be a tunable parameter
    # for users. Move to arguments if it is desired to make it tunable.
    # TODO: Need to scale each metric appropriately. Since good scaling
    # factors are not available right now, setting the weights of age and depth
    # as zero.
    weights = {"distance": 1, "age": 0, "depth": 0}

    distance_metric = distance_matrix(wd, weights, list_wells)
    clustered_data = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=threshold_distance / 2,
    ).fit(distance_metric)

    if list_wells is None:
        wd.add_new_column_ordered("cluster", "Clusters", clustered_data.labels_)
        return _well_clusters(wd)

    # Return the clusters as a dictionary without modifying the original WellData object
    cluster_labels = pd.Series(clustered_data.labels_, index=list_wells)
    return (
        cluster_labels.groupby(cluster_labels)
        .apply(lambda group: group.index.tolist())
        .to_dict()
    )


def get_pairwise_metrics(wd: WellData, list_wells: list) -> pd.DataFrame:
    """
    Returns pairwise metric values for all possible pairs of wells in
    `list_wells`.

    Parameters
    ----------
    wd : WellData
        Object containing well data

    list_wells : list
        List of wells for which pairwise metrics are needed to
        be calculated

    Returns
    -------
    pairwise_metrics : DataFrame
        DataFrame containing the pairwise metric values
    """
    well_pairs = list(combinations(list_wells, 2))
    pairwise_metrics = pd.DataFrame()
    if len(list_wells) == 1:
        metric_columns = ["dist_range"]
        if wd.column_names.age is not None:
            metric_columns.append("age_range")
        if wd.column_names.depth is not None:
            metric_columns.append("depth_range")
        return pd.DataFrame(columns=metric_columns, dtype=float)

    # Compute pairwise distances
    pairwise_metrics["dist_range"] = distance_matrix(
        wd, {"distance": 1}, list_wells
    ).stack()[well_pairs]

    # Compute pairwise age range
    if wd.column_names.age is not None:
        pairwise_metrics["age_range"] = distance_matrix(
            wd, {"age": 1}, list_wells
        ).stack()[well_pairs]

    # Compute pairwise depth range
    if wd.column_names.depth is not None:
        pairwise_metrics["depth_range"] = distance_matrix(
            wd, {"depth": 1}, list_wells
        ).stack()[well_pairs]

    return pairwise_metrics


# pylint: disable=too-many-positional-arguments
def perform_louvain_clustering(
    wd: WellData,
    threshold_distance: float,
    threshold_cluster_size: int,
    nearest_neighbors: int,
    seed: int = 4242,
    resolution: float = None,
    max_resolution: float = 10.0,
    list_wells: Optional[List] = None,
) -> dict:
    """
    Partitions the data into smaller clusters using the Louvain community detection method,
    limiting each well to connect only with its nearest_neighbors within a threshold distance.
    Dynamically adjusts the resolution parameter from 1 to max_resolution with 0.5 increments,
    to ensure no cluster exceeds the size threshold if resolution parameter is not provided.

    Parameters
    ----------
    wd : WellData
        Object containing well data

    threshold_distance : float
        Threshold distance (in miles) for breaking clusters

    threshold_cluster_size : int
        cluster size of the largest cluster

    nearest_neighbors : int
        nearest neighbors to consider when constructing graph for Louvain clustering

    seed : int
        random seed for Louvain communities function

    resolution : float
        resolution for Louvain communities function

    max_resolution : float
        maximum resolution while dynamically setting the resolution for Louvain communities function

    list_wells : list, optional
        If specified, performs clustering only for the specified subset of wells

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    if check_existing_cluster(wd):
        return _well_clusters(wd)

    data = wd.data if list_wells is None else wd.data.loc[list(list_wells)]

    coordinates = list(zip(data[wd.col_names.latitude], data[wd.col_names.longitude]))
    ball_tree, coords_rad = build_ball_tree(coordinates)

    # k = nearest_neighbors + 1 to include the well itself
    distances, indices = ball_tree.query(coords_rad, k=nearest_neighbors + 1)

    well_graph = nx.Graph()
    well_ids = data.index.tolist()
    well_graph.add_nodes_from(well_ids)
    # Add edges within distance threshold
    for i, neighbors in enumerate(indices):
        well_id_1 = well_ids[i]
        for j in range(1, len(neighbors)):  # Skip the point itself (index 0)
            well_id_2 = well_ids[neighbors[j]]
            if not well_graph.has_edge(well_id_1, well_id_2):
                distance = distances[i][j] * EARTH_RADIUS  # Convert to distance
                if distance <= threshold_distance:
                    well_graph.add_edge(
                        well_id_1, well_id_2, weight=threshold_distance - distance
                    )

    # Louvain clustering
    def _get_clusters(resolution: float):
        communities = nx.community.louvain_communities(
            well_graph, seed=seed, resolution=resolution
        )
        well_cluster_map = {
            well: cluster_id
            for cluster_id, community in enumerate(communities)
            for well in community
        }
        _cluster_list = [well_cluster_map[well] for well in well_ids]
        _max_cluster_size = max(len(community) for community in communities)
        LOGGER.debug(
            f"For resolution={resolution}, the maximum cluster size = {max_cluster_size}"
        )

        return _cluster_list, _max_cluster_size

    max_cluster_size = len(well_ids)  # Set a large number

    # If length of data is less than cluster_threshold, assign all wells to one cluster
    if max_cluster_size <= threshold_cluster_size:
        cluster_list = np.ones(len(data))
        LOGGER.warning(
            "Number of wells in the dataset is less than the threshold cluster size. "
            "Assigning all wells to the same cluster."
        )

    elif resolution is not None:
        # Using user-specified resolution for clustering
        cluster_list, max_cluster_size = _get_clusters(resolution=resolution)

    else:
        # Adaptively adjust resolution to control the cluster size
        resolution = 0.5  # Initial resolution
        while max_cluster_size > threshold_cluster_size:
            resolution += 0.5  # increase resolution--> starts from 1; increases in every iteration
            cluster_list, max_cluster_size = _get_clusters(resolution=resolution)
            if resolution == max_resolution:
                raise_exception("Could not reach desired cluster sizes", RuntimeError)

    if list_wells is None:

        wd.add_new_column_ordered("cluster", "Clusters", cluster_list)

        LOGGER.info(f"The resolution parameter is set to {resolution}.")
        LOGGER.info(
            f"There are {len(wd.data['Clusters'].unique())} clusters with "
            f"the largest cluster containing {max_cluster_size} wells."
        )

        return _well_clusters(wd)

    # Return the clusters as a dictionary without modifying the original WellData object
    cluster_labels = pd.Series(cluster_list, index=list_wells)
    return (
        cluster_labels.groupby(cluster_labels)
        .apply(lambda group: group.index.tolist())
        .to_dict()
    )


def perform_exhaustive_clustering(
    wd: WellData,
    threshold_distance: float,
    largest_cluster: int = 65,
) -> dict:
    """
    Create overlapping clusters around each well using threshold distance parameter.

    Parameters
    ----------
    wd : WellData
        Object containing well data

    threshold_distance : float
        Threshold distance (in miles) for breaking clusters

    largest_cluster : int
        Largest cluster size before reducing the number of clusters

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """
    # pylint: disable=too-many-locals
    if check_existing_cluster(wd):
        LOGGER.info("Returning existing clusters from cache.")
        return _well_clusters(wd)

    coordinates = list(
        zip(wd.data[wd.col_names.latitude], wd.data[wd.col_names.longitude])
    )
    ball_tree, coords_rad = build_ball_tree(coordinates)

    radius_radians = (threshold_distance / 2) / EARTH_RADIUS
    neighbors_indices = ball_tree.query_radius(coords_rad, r=radius_radians)

    well_ids = wd.data.index.tolist()

    # Initial clusters
    clusters = {
        well_ids[i]: [well_ids[j] for j in neighbors_indices[i]]
        for i in range(len(well_ids))
    }

    max_cluster_size = max(len(neigh) for neigh in clusters.values())
    LOGGER.info(f"Initial clustering completed with {len(clusters)} clusters.")
    LOGGER.info(f"Maximum cluster size is {max_cluster_size}.")

    # Early return if cluster size is manageable
    if max_cluster_size < largest_cluster:
        LOGGER.info(
            "Maximum cluster size < largest_cluster. Returning initial clusters."
        )
        return clusters

    # Sort clusters in descending order of cluster size.
    # If two clusters have the same size, use the center well ID to break ties (ascending).
    sorted_clusters = sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0]))

    selected_clusters = {}
    used_wells = set()

    for center_well, neighbor_wells in sorted_clusters:
        if center_well not in used_wells:
            selected_clusters[center_well] = neighbor_wells
            used_wells.update(neighbor_wells)

    LOGGER.info(f"{len(selected_clusters)} clusters selected using greedy approach.")

    # Add more clusters based on impact score of center well
    n_wells = len(well_ids)
    max_total_clusters = (n_wells + len(selected_clusters)) // 2
    additional_clusters_to_add = max_total_clusters - len(selected_clusters)

    # Sort clusters by impact score of center well, descending
    impact_series = wd.data[wd.col_names.priority_score]
    sorted_by_impact = sorted(
        clusters.items(),
        key=lambda x: (-impact_series.get(x[0], 0), x[0]),  # fallback 0 if missing
    )

    counter = 0
    for center_well, neighbor_wells in sorted_by_impact:
        if counter == additional_clusters_to_add:
            break
        if center_well not in selected_clusters:
            selected_clusters[center_well] = neighbor_wells
            counter += 1

    LOGGER.info(
        f"Added {counter} impact-prioritized clusters. Final total: {len(selected_clusters)}."
    )

    return selected_clusters


# pylint: disable=too-many-branches
def perform_boundary_clustering(
    wd: WellData,
    subcluster_method: Union[None, str] = None,
    max_boundary_cluster_size: int = 200,
    **clustering_kwargs,
) -> dict:
    """
    Cluster wells by a geographic boundary region. If the number of wells within the region
    exceeds the max_boundary_cluster_size, then sub-clustering is performed within the region
    using the specified subcluster_method.

    Parameters
    ----------
    wd : WellData
        Object containing well data

    subcluster_method : str, default = None
        Method used for sub-clustering the wells within geographic boundaries.
        Options are "Agglomerative" and "Louvain".

    max_boundary_cluster_size : int, default = 200
        Maximum number of wells in a cluster before sub-clustering is performed

    **clustering_kwargs:
        Additional keyword arguments to be passed to the sub-clustering method

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """

    if check_existing_cluster(wd):
        LOGGER.info("Returning existing clusters from cache.")
        return _well_clusters(wd)

    # check if the geographic boundary column exists
    if wd.col_names.geographic_boundary is None:
        raise_exception(
            "Could not find data to perform boundary clustering, please specify a column name "
            "for geographic_boundary in the WellDataColumnNames object.",
            ValueError,
        )

    # check if all wells have geographic boundary data
    if wd.data[wd.col_names.geographic_boundary].isna().values.any():
        raise_exception(
            "Some wells are missing geographic boundary data, cannot perform boundary clustering",
            ValueError,
        )

    # check if sub-clustering method is valid
    if subcluster_method is not None and subcluster_method not in [
        "Agglomerative",
        "Louvain",
    ]:
        raise_exception(
            f"Sub-clustering method {subcluster_method} is not supported."
            f"Please choose from 'Agglomerative' or 'Louvain'.",
            ValueError,
        )

    # list of unique regions
    unique_regions = wd.data[wd.col_names.geographic_boundary].unique()

    # dictionary to map well_id -> cluster number
    well_to_cluster = {}
    cluster_counter = 0

    for region in unique_regions:
        # list of all wells in the region
        well_ids = wd.data[wd.col_names.geographic_boundary][
            wd.data[wd.col_names.geographic_boundary] == region
        ].index.tolist()
        num_wells = len(well_ids)

        if num_wells <= max_boundary_cluster_size or subcluster_method is None:

            if num_wells > max_boundary_cluster_size:
                LOGGER.warning(
                    f"Ingnoring the max_boundary_cluster_size for region '{region}' since "
                    f"subcluster_method is None."
                )

            # assign all wells in this region to the same cluster
            for well_id in well_ids:
                well_to_cluster[well_id] = cluster_counter
            cluster_counter += 1
        else:
            # perform sub-clustering within this region using the specified method
            if subcluster_method == "Agglomerative":
                # sub_clusters maps cluster number -> list of well ids
                sub_clusters = perform_agglomerative_clustering(
                    wd,
                    list_wells=well_ids,
                    threshold_distance=clustering_kwargs.get("threshold_distance"),
                )
            else:
                sub_clusters = perform_louvain_clustering(
                    wd,
                    list_wells=well_ids,
                    **clustering_kwargs,
                )

            # assign sub-clusters to the main cluster number
            for sub_cluster_wells in sub_clusters.values():
                for well_id in sub_cluster_wells:
                    well_to_cluster[well_id] = cluster_counter
                cluster_counter += 1

    # put well_to_cluster in correct order for adding to the dataframe
    clusters = [well_to_cluster[well_id] for well_id in wd.data.index]

    wd.add_new_column_ordered("cluster", "Clusters", clusters)

    return _well_clusters(wd)
