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

from __future__ import annotations  # avoid circular imports for type hints

# Standard libs
import logging
from math import ceil, floor
from typing import TYPE_CHECKING, Optional, Tuple

# Installed libs
import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained  # pylint: disable=import-error

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils.clustering_utils import _well_clusters
from primo.utils.raise_exception import raise_exception

if TYPE_CHECKING:
    # User-defined libs
    from primo.opt_model.model_options import OptModelInputs

LOGGER = logging.getLogger(__name__)

_MAX_CLUSTER_SIZE_TARGET = 200
_MIN_CLUSTER_SIZE_TARGET = 10


def _estimate_min_clusters(model_options: OptModelInputs) -> int:
    """
    Step 1 of perform_heuristic_constrained_clustering - Estimate the lower
    bound on the number of clusters.

    Parameters
    ----------
    model_options : OptModelInputs
        Object containing the optimization model options

    Returns
    -------
    int
        Estimated minimum number of clusters
    """
    wd = model_options.config.well_data
    config = model_options.config

    if config.max_wells_in_project is None and config.max_cost_project is None:
        min_clusters = ceil(len(wd.data) / _MAX_CLUSTER_SIZE_TARGET)
        LOGGER.info(
            f"Minimum number of clusters is unconstrained. Using {min_clusters} as default."
        )
        return min_clusters

    average_well_cost = (
        wd.data[wd.col_names.cost_of_plugging].mean()
        if wd.col_names.cost_of_plugging is not None
        else None
    )
    enforced_max_cost_per_project = get_enforced_max_cost_per_project(
        config.max_cost_project,
        config.max_wells_in_project,
        config.mobilization_cost,
        average_well_cost,
    )
    min_clusters = ceil(config.total_budget / enforced_max_cost_per_project)
    LOGGER.info(f"Minimum number of clusters estimated to be {min_clusters}.")
    return min_clusters


def _estimate_min_project_size(model_options: OptModelInputs) -> int:
    """
    Step 2 of perform_heuristic_constrained_clustering - Estimate the minimum
    project size.

    There are three scenarios for selecting the minimum project size:
    1. No relevant constraints - use a default value of 1 to leave it unconstrained
    2. A hard constraint from either the minimum number of wells per project or a
       minimum cost per project when using mobilization costs
    3. A constraint from a minimum cost per project when using user-provided well
       costs - calculate worst case scenario

    Parameters
    ----------
    model_options : OptModelInputs
        Object containing the optimization model options

    Returns
    -------
    int
        Estimated minimum cluster size
    """
    wd = model_options.config.well_data
    config = model_options.config
    min_cluster_size_list = [1]

    if config.min_cost_project is not None:
        if config.mobilization_cost is None:
            # User-provided well costs, returns a worst-case scenario
            min_wells = min_project_cost_to_num_wells_cost_of_plugging(
                config.min_cost_project, wd.data[wd.col_names.cost_of_plugging]
            )
        else:
            min_wells = min_project_cost_to_num_wells_mobilization_cost(
                config.min_cost_project,
                config.mobilization_cost,
            )
        min_cluster_size_list.append(min_wells)

    if config.min_wells_in_project is not None:
        min_cluster_size_list.append(config.min_wells_in_project)

    min_size = max(min_cluster_size_list)
    LOGGER.info(f"Minimum cluster size estimated to be {min_size}.")
    return min_size


def _estimate_max_clusters(model_options: OptModelInputs, min_size: int) -> int:
    """
    Step 3 of perform_heuristic_constrained_clustering - Estimate the upper
    bound on the number of clusters.

    Parameters
    ----------
    model_options : OptModelInputs
        Object containing the optimization model options
    min_size : int
        Estimated minimum project size, as returned by _estimate_min_project_size

    Returns
    -------
    int
        Estimated maximum number of clusters
    """
    config = model_options.config
    total_wells = len(config.well_data.data)

    if config.min_cost_project is None and config.min_wells_in_project is None:
        max_clusters = floor(total_wells / _MIN_CLUSTER_SIZE_TARGET)
        LOGGER.info(
            f"Maximum number of clusters is unconstrained. Using {max_clusters} as default."
        )
        return max(max_clusters, 1)  # we don't want to return 0 clusters

    max_clusters = floor(total_wells / min_size)

    if max_clusters == 0:
        raise_exception(
            "The minimum project size or cost constraint is larger than the total number of wells."
            " Please relax the constraints and try again.",
            ValueError,
        )

    LOGGER.info(f"Maximum number of clusters estimated to be {max_clusters}.")
    return max_clusters


# pylint: disable=inconsistent-return-statements
def _get_initial_n_clusters(
    objective: str,
    min_clusters: int,
    max_clusters: int,
    target_clusters: Optional[int],
) -> int:
    """
    Step 4 of perform_heuristic_constrained_clustering - Choose the starting
    number of clusters to try for the given objective.

    Parameters
    ----------
    objective : str
        One of "center", "maximum", "minimum", "target"
    min_clusters : int
        Estimated minimum number of clusters
    max_clusters : int
        Estimated maximum number of clusters
    target_clusters : Optional[int]
        Target number of clusters, required if objective is "target"

    Returns
    -------
    int
        Starting number of clusters to try
    """
    if objective == "maximum":
        return max_clusters
    if objective == "minimum":
        return min_clusters
    if objective == "center":
        return (min_clusters + max_clusters) // 2
    if objective == "target":
        return target_clusters

    raise_exception(
        f"Unrecognized objective '{objective}'. Valid options are "
        "'center', 'maximum', 'minimum', 'target'.",
        ValueError,
    )


def _set_wd_cluster_column(wd: WellData, cluster_assignment: np.ndarray) -> None:
    """
    Overwrite (or set) the WellData object's cluster column with the given
    cluster assignment.

    Parameters
    ----------
    wd : WellData
        The WellData object to update
    cluster_assignment : np.ndarray
        Cluster label for each well, in the order of wd.data
    """
    if hasattr(wd.col_names, "cluster"):
        wd.data.drop(columns=["Clusters"], inplace=True)
        delattr(wd.col_names, "cluster")

    wd.add_new_column_ordered("cluster", "Clusters", cluster_assignment)


def _run_clustering_iteration(
    wd: WellData,
    n_clusters: int,
    min_size: int,
    size_max: float,
    model_options: OptModelInputs,
) -> Tuple[np.ndarray, bool, bool, bool]:
    """
    Run a single KMeansConstrained clustering attempt and evaluate it against
    the model's budget and minimum-size constraints.

    Parameters
    ----------
    wd : WellData
        The WellData object to cluster
    n_clusters : int
        Number of clusters to form
    min_size : int
        Minimum allowed cluster size
    size_max : float
        Maximum allowed cluster size
    model_options : OptModelInputs
        Object containing the optimization model options

    Returns
    -------
    Tuple[np.ndarray, bool, bool, bool]
        clustered_data, valid_budget, valid_min_constraints, valid_clustering
    """
    clustered_data = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=min_size,
        size_max=size_max,
        random_state=0,
    ).fit_predict(wd.data[[wd.col_names.latitude, wd.col_names.longitude]].values)

    _set_wd_cluster_column(wd, clustered_data)
    clusters = _well_clusters(wd)

    valid_budget = can_spend_full_budget(clusters, model_options)
    valid_min_constraints = are_minimum_constraints_satisfied(clusters, model_options)
    valid_clustering = valid_budget and valid_min_constraints

    return clustered_data, valid_budget, valid_min_constraints, valid_clustering


def _advance_n_clusters(
    n_clusters: int, valid_budget: bool, valid_min_constraints: bool, objective: str
) -> int:
    """
    Decide the next candidate number of clusters to try, based on which
    constraint failed (if any) for the current attempt.

    Parameters
    ----------
    n_clusters : int
        Number of clusters used in the current attempt
    valid_budget : bool
        Whether the current attempt satisfies the budget constraint
    valid_min_constraints : bool
        Whether the current attempt satisfies the minimum-size constraints
    objective : str
        One of "center", "maximum", "minimum", "target"

    Returns
    -------
    int
        Number of clusters to try next
    """
    if not valid_budget:
        return n_clusters + 1
    if not valid_min_constraints:
        return n_clusters - 1
    if objective == "maximum":
        return n_clusters + 1
    if objective == "minimum":
        return n_clusters - 1
    return n_clusters


# pylint: disable=inconsistent-return-statements
def _finish_search(
    wd: WellData, last_valid_cluster: Optional[np.ndarray], failure_message: str
) -> dict[int, list[int]]:
    """
    Conclude the iterative search: restore the best clustering found so far,
    or raise an error if no valid clustering was ever found.

    Parameters
    ----------
    wd : WellData
        The WellData object to update
    last_valid_cluster : Optional[np.ndarray]
        The best cluster assignment found so far, or None if none was found
    failure_message : str
        Message to raise if last_valid_cluster is None

    Returns
    -------
    dict[int, list[int]]
        Dictionary of lists of wells contained in each cluster
    """
    if last_valid_cluster is not None:
        _set_wd_cluster_column(wd, last_valid_cluster)
        return _well_clusters(wd)

    raise_exception(failure_message, RuntimeError)


def perform_heuristic_constrained_clustering(
    model_options: OptModelInputs,
    objective: str = "center",
    target_clusters: Optional[int] = None,
    max_iterations: int = 10,
    uniformity_parameter: float = 0,
) -> dict[int, list[int]]:
    """
    Cluster wells automatically considering the constraints of the optimization model.

    Parameters
    ----------
    model_options : OptModelInputs
        Object containing the optimization model options
    objective : str, optional
        Determines where in the feasible region of the number of clusters to search for a solution.
        Options are "center", "maximum", "minimum", "target". Default is "center".
    target_clusters : int, optional
        Target number of clusters to achieve if objective is set to "target". Default is None.
    max_iterations : int, optional
        Maximum number of iterations for the clustering process. Default is 10.
    uniformity_parameter : float, optional
        Parameter controlling the uniformity of cluster sizes. Default is 0.
        Must be between 0 and 1. A value closer to 1 encourages more uniform cluster sizes.

    Returns
    -------
    dict[int, list[int]]
        Dictionary of lists of wells contained in each cluster
    """
    min_clusters = _estimate_min_clusters(model_options)
    min_size = _estimate_min_project_size(model_options)
    max_clusters = _estimate_max_clusters(model_options, min_size)
    n_clusters = _get_initial_n_clusters(
        objective, min_clusters, max_clusters, target_clusters
    )

    wd = model_options.config.well_data
    state = {
        "iteration": 0,
        "valid_min_constraints": True,
        "last_valid_cluster": None,
        "tried": [],
    }

    while True:
        LOGGER.info(
            f"Iteration {state['iteration']}: Trying with {n_clusters} clusters"
        )

        # Determine the maximum cluster size based on the uniformity parameter
        size_max = (
            ceil(len(wd.data) / n_clusters) - len(wd.data)
        ) * uniformity_parameter + len(wd.data)

        # Determine if min_size needs to be relaxed
        if min_size * n_clusters > len(wd.data):
            min_size = floor(len(wd.data) / n_clusters)
            LOGGER.info(
                f"Relaxing minimum cluster size to {min_size} to satisfy total wells constraint."
            )
        elif state["iteration"] > 0 and not state["valid_min_constraints"]:
            min_size = floor(len(wd.data) / n_clusters)
            LOGGER.info(
                f"Increasing minimum cluster size to {min_size} to satisfy minimum constraints."
            )

        (
            clustered_data,
            valid_budget,
            state["valid_min_constraints"],
            valid_clustering,
        ) = _run_clustering_iteration(wd, n_clusters, min_size, size_max, model_options)
        LOGGER.info(
            f"Valid budget: {valid_budget}, "
            f"Valid min constraints: {state['valid_min_constraints']}, "
            f"Valid clustering: {valid_clustering}"
        )

        if not valid_budget and not state["valid_min_constraints"]:
            raise_exception(
                "Could not find a valid clustering that satisfies all constraints. "
                "Please relax the constraints and try again.",
                RuntimeError,
            )

        if valid_clustering:
            if objective in ["center", "target"]:
                # For these cases we want to find the first valid clustering
                return _well_clusters(wd)
            if objective in ["maximum", "minimum"]:
                state["last_valid_cluster"] = clustered_data

        if state["iteration"] == max_iterations:
            return _finish_search(
                wd,
                state["last_valid_cluster"],
                f"Could not find a valid clustering within {max_iterations} iterations.",
            )

        # Continue to the next iteration
        state["tried"].append(n_clusters)
        n_clusters = _advance_n_clusters(
            n_clusters, valid_budget, state["valid_min_constraints"], objective
        )

        if n_clusters in state["tried"] or n_clusters == 0:
            if state["last_valid_cluster"] is not None:
                LOGGER.info(
                    "Solution found, returning the last valid cluster configuration."
                )
            return _finish_search(
                wd,
                state["last_valid_cluster"],
                "Could not find a valid clustering that satisfies all constraints. "
                "Please relax the constraints and try again.",
            )

        state["iteration"] += 1


def get_enforced_max_cost_per_project(
    maximum_cost_per_project: Optional[float],
    maximum_wells_per_project: Optional[int],
    mobilization_cost: Optional[dict],
    average_cost_per_well: Optional[float] = None,
) -> float:
    """
    This function determines which of the "maximum cost per project" or "maximum number of wells
    per project" parameters is enforcing the constraint on maximum project cost.

    This should only be called if at least one of the parameters is specified.

    Parameters
    ----------
    maximum_cost_per_project : Optional[float]
        Maximum cost per project

    maximum_wells_per_project : Optional[int]
        Maximum number of wells

    mobilization_cost : Optional[dict]
        Dictionary mapping number of wells to total cost

    average_cost_per_well : Optional[float]
        Average cost per well, used if mobilization_cost is not provided

    Returns
    -------
    float
        Enforced maximum cost per project
    """
    if maximum_cost_per_project is None and maximum_wells_per_project is None:
        raise_exception(
            "At least one of maximum_cost_per_project or maximum_wells_per_project must be "
            "specified.",
            ValueError,
        )

    max_cost_list = []
    if maximum_cost_per_project is not None:
        max_cost_list.append(maximum_cost_per_project)
    if maximum_wells_per_project is not None:
        if mobilization_cost is not None:
            # handles the case where the max project size is larger than the total wells
            total_wells = max(mobilization_cost.keys())
            max_cost_list.append(
                mobilization_cost[min(maximum_wells_per_project, total_wells)]
            )
        else:
            max_cost_list.append(average_cost_per_well * maximum_wells_per_project)

    return min(max_cost_list)


def min_project_cost_to_num_wells_cost_of_plugging(
    min_project_cost: float,
    plugging_costs: pd.Series,
) -> int:
    """
    This function calculates the largest possible value for the minimum number of wells required to
    satisfy a given minimum project cost constraint, assuming the least expensive wells are
    selected.

    This function should only be called if plugging costs are present.

    Parameters
    ----------
    min_project_cost : float
        The minimum project cost

    plugging_costs : pd.Series
        The plugging costs for each well

    Returns
    -------
    int
        The upper bound on the minimum number of wells
    """

    # null values are possible if data checks are turned off
    if plugging_costs.isnull().any():
        raise_exception(
            "Missing entires found in plugging cost data. Please ensure all values are present.",
            ValueError,
        )

    # make sure the problem is feasible in the first place
    total_plugging_cost = plugging_costs.sum()
    if total_plugging_cost < min_project_cost:
        raise_exception(
            "Minimum project cost constraint is larger than the total plugging cost. "
            "Problem is infeasible.",
            ValueError,
        )

    # the upper bound uses the least expensive wells
    plugging_costs = plugging_costs.sort_values(ascending=True)

    total_cost = 0
    for i, cost in enumerate(plugging_costs):
        total_cost += cost
        if total_cost >= min_project_cost:
            ub = i + 1
            break

    # TODO: try finding a lower bound and starting from the middle to decrease iterations

    return ub


# pylint: disable=inconsistent-return-statements
def min_project_cost_to_num_wells_mobilization_cost(
    min_project_cost: float,
    mobilization_cost: dict,
) -> int:
    """
    This function finds the minimum number of wells required to meet the minimum project cost
    constraint given a specific mobilization cost. This should only be used if there is a
    minimum project costs constraint.

    Parameters
    ----------
    min_project_cost : float
        Minimum cost per project

    mobilization_cost : dict
        Dictionary mapping number of wells to total cost

    Returns
    -------
    int
        Minimum number of wells required to meet the project cost constraint
    """

    for num_wells, cost in mobilization_cost.items():
        if cost >= min_project_cost:
            return num_wells

    # the problem is infeasible so raise an error
    raise_exception(
        "Minimum project cost constraint is larger than the cost to plug all wells. "
        "Problem is infeasible.",
        ValueError,
    )


def can_spend_full_budget(clusters: dict, model_options: OptModelInputs) -> bool:
    """
    Checks if the given clusters satisfy the constraints specified in model_options.

    Parameters
    ----------
    clusters : dict
        A dictionary mapping cluster IDs to lists of wells in each cluster
    model_options : OptModelInputs
        The optimization model options

    Returns
    -------
    bool
        True if the clusters satisfy the constraints, False otherwise
    """

    # Extract the necessary configuration options
    wd = model_options.config.well_data
    config = model_options.config

    # null values are possible if data checks are turned off
    if (
        config.mobilization_cost is None
        and wd.data[wd.col_names.cost_of_plugging].isnull().any()
    ):
        raise_exception(
            "Missing entires found in plugging cost data. Please ensure all values are present.",
            ValueError,
        )

    # the number of wells in each cluster
    wells_per_cluster = {}
    for cluster, wells in clusters.items():
        wells_per_cluster[cluster] = len(wells)

    # make sure the full budget is available to spend
    total_max_cost = 0
    total_actual_cost = 0
    for cluster, count in wells_per_cluster.items():
        max_cost_list = []

        # max cost enforced directly
        if config.max_cost_project is not None:
            max_cost_list.append(config.max_cost_project)

        # max cost enforced by the maximum number of wells
        if config.max_wells_in_project is not None:
            if config.mobilization_cost is not None:
                # handles the case where the max project size is larger than the total wells
                max_wells = min(config.max_wells_in_project, len(wd.data))
                max_cost_list.append(config.mobilization_cost[max_wells])
            else:
                # make sure even in worst case scenario where only cheapest wells are selected
                plugging_costs = wd.data[wd.data[wd.col_names.cluster] == cluster][
                    wd.col_names.cost_of_plugging
                ].sort_values(ascending=True)
                max_cost_list.append(
                    plugging_costs.head(config.max_wells_in_project).sum()
                )

        # max cost enforced by the actual number of wells in the cluster
        if config.mobilization_cost is not None:
            max_cost_list.append(config.mobilization_cost[count])
            total_actual_cost += config.mobilization_cost[count]
        else:
            plugging_costs = wd.data[wd.data[wd.col_names.cluster] == cluster][
                wd.col_names.cost_of_plugging
            ]
            max_cost_list.append(plugging_costs.sum())
            total_actual_cost += plugging_costs.sum()

        total_max_cost += min(max_cost_list)

    # budget sufficient case - the entire budget can't be spent anyway
    if total_actual_cost <= config.total_budget:
        return True

    if total_max_cost < config.total_budget:
        return False

    return True


def are_minimum_constraints_satisfied(
    clusters: dict, model_options: OptModelInputs
) -> bool:
    """
    Checks if the given clusters satisfy the constraints specified in model_options.

    Parameters
    ----------
    clusters : dict
        A dictionary mapping cluster IDs to lists of wells in each cluster
    model_options : OptModelInputs
        The optimization model options

    Returns
    -------
    bool
        True if the clusters satisfy the constraints, False otherwise
    """

    # extract the necessary configuration options
    wd = model_options.config.well_data
    minimum_wells_per_project = model_options.config.min_wells_in_project
    minimum_cost_per_project = model_options.config.min_cost_project
    mobilization_cost = model_options.config.mobilization_cost

    # the number of wells in each cluster
    wells_per_cluster = {cluster: len(wells) for cluster, wells in clusters.items()}

    # check the minimum number of wells constraint
    if minimum_wells_per_project is not None:
        for cluster, count in wells_per_cluster.items():
            if count < minimum_wells_per_project:
                return False

    # check the minimum cost per project constraint
    if minimum_cost_per_project is not None:
        if mobilization_cost is not None:
            for cluster, count in wells_per_cluster.items():
                project_cost = mobilization_cost[count]
                if project_cost < minimum_cost_per_project:
                    return False
        else:
            # make sure the sum of the plugging costs of the wells in each cluster exceeds the
            # minimum project cost
            for cluster, count in wells_per_cluster.items():
                plugging_costs = wd.data[wd.data[wd.col_names.cluster] == cluster][
                    wd.col_names.cost_of_plugging
                ]
                if plugging_costs.sum() < minimum_cost_per_project:
                    return False

    return True
