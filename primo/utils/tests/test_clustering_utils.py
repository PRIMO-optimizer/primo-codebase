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
import os
from itertools import combinations
from pathlib import Path

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo import OptModelInputs
from primo.data_parser import WellDataColumnNames
from primo.data_parser.well_data import WellData
from primo.utils.clustering_utils import (
    distance_matrix,
    get_pairwise_metrics,
    perform_agglomerative_clustering,
    perform_boundary_clustering,
    perform_louvain_clustering,
)
from primo.utils.constrained_clustering_utils import (
    _estimate_max_clusters,
    _estimate_min_clusters,
    _estimate_min_project_size,
    are_minimum_constraints_satisfied,
    can_spend_full_budget,
    get_enforced_max_cost_per_project,
    min_project_cost_to_num_wells_cost_of_plugging,
    min_project_cost_to_num_wells_mobilization_cost,
)

# pylint: disable=redefined-outer-name


# Sample data for testing
@pytest.mark.parametrize(
    "well_data, weight, result, status",
    [  # Case1: Passed case
        (  # Well data
            [
                {
                    "Well API": "W1",
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                    "Op Name": "Owner 1",
                },
                {
                    "Well API": "W2",
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                    "Op Name": "Owner 2",
                },
            ],
            {"age": 0.5, "depth": 0.5},  # Weight
            [[0.0, 505.0], [505.0, 0.0]],  # Result
            True,  # Status
        ),
        # Case 3: Summation of feature weights is not 1
        (  # Well data
            [
                {
                    "Well API": "W1",
                    "Latitude": 42.0,
                    "Longitude": -70,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                    "Op Name": "Owner 1",
                },
                {
                    "Well API": "W2",
                    "Latitude": 43.0,
                    "Longitude": -74,
                    "Age [Years]": 30,
                    "Depth [ft]": 5000,
                    "Op Name": "Owner 2",
                },
            ],
            {"distance": 0.3, "age": 0.8, "depth": 0.5},  # Weight
            "Feature weights do not add up to 1",  # Result
            False,  # Status
        ),
        # Case 6: Spurious feature provided
        (  # Well data
            [
                {
                    "Well API": "W1",
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                    "Op Name": "Owner 1",
                },
                {
                    "Well API": "W2",
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                    "Op Name": "Owner 2",
                },
            ],
            {"distance": 0.5, "depth": 0.2, "ages": 0.3, "age": 0.3},  # Weight
            (
                "Received feature(s) [ages] that are not "
                "supported in the clustering step."
            ),  # Result
            False,  # Status
        ),
        # Add more test cases as needed
    ],
)
def test_distance_matrix(well_data, weight, result, status):
    """
    Tests for distance_matrix method
    """
    well_df = pd.DataFrame(well_data)
    well_cn = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Op Name",
    )
    wd = WellData(well_df, well_cn)
    result_arr = np.array(result)
    if status:
        assert np.allclose(
            distance_matrix(wd, weight), result_arr, rtol=1e-5, atol=1e-8
        )
    else:
        with pytest.raises(ValueError):
            _ = distance_matrix(wd, weight) == result


def test_perform_agglomerative_clustering(caplog):
    """
    Tests for perform_clustering method
    """
    # pylint: disable=duplicate-code
    warning_message = (
        "Found cluster attribute in the WellDataColumnNames object. "
        "Assuming that the data is already clustered. If the corresponding "
        "column does not correspond to clustering information, please use a "
        "different name for the attribute cluster while instantiating the "
        "WellDataColumnNames object."
    )
    filename = os.path.dirname(os.path.realpath(__file__))[:-12]  # Primo folder
    filename += "//data_parser//tests//random_well_data.csv"

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
    )

    wd = WellData(data=filename, column_names=col_names)
    assert "Clusters" not in wd
    assert not hasattr(col_names, "cluster")

    clusters = perform_agglomerative_clustering(wd, threshold_distance=20)
    num_clusters = len(set(clusters.keys()))
    assert "Clusters" in wd
    assert hasattr(col_names, "cluster")
    assert num_clusters == 16
    assert num_clusters == len(set(wd.data["Clusters"]))
    assert warning_message not in caplog.text

    # Capture the warning if the data has already been clustered
    clusters = perform_agglomerative_clustering(wd, threshold_distance=20)
    num_clusters = len(set(clusters.keys()))
    assert num_clusters == 16
    assert warning_message in caplog.text


def test_perform_louvain_clustering(caplog):
    """
    Tests for perform_clustering method
    """
    # pylint: disable=duplicate-code
    warning_message = (
        "Found cluster attribute in the WellDataColumnNames object. "
        "Assuming that the data is already clustered. If the corresponding "
        "column does not correspond to clustering information, please use a "
        "different name for the attribute cluster while instantiating the "
        "WellDataColumnNames object."
    )
    filename = os.path.dirname(os.path.realpath(__file__))[:-12]  # Primo folder
    filename += "//data_parser//tests//random_well_data.csv"

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
    )

    # Test the case where length of data is smaller than the max_cluster_threshold
    wd = WellData(data=filename, column_names=col_names)
    assert "Clusters" not in wd
    assert not hasattr(col_names, "cluster")

    clusters = perform_louvain_clustering(
        wd, threshold_distance=10, threshold_cluster_size=300, nearest_neighbors=10
    )
    num_clusters = len(set(clusters.keys()))
    assert "Clusters" in wd
    assert hasattr(col_names, "cluster")
    assert num_clusters == 1
    assert num_clusters == len(set(wd.data["Clusters"]))
    assert warning_message not in caplog.text

    # Test the case where length of data is greater than the max_cluster_threshold
    wd.data.drop(columns=["Clusters"], inplace=True)
    delattr(col_names, "cluster")
    assert "Clusters" not in wd
    assert not hasattr(col_names, "cluster")

    clusters = perform_louvain_clustering(
        wd, threshold_distance=10, threshold_cluster_size=100, nearest_neighbors=10
    )
    num_clusters = len(set(clusters.keys()))
    assert "Clusters" in wd
    assert hasattr(col_names, "cluster")
    assert num_clusters == 14
    assert num_clusters == len(set(wd.data["Clusters"]))

    # Capture the warning if the data has already been clustered
    clusters = perform_louvain_clustering(
        wd, threshold_distance=10, threshold_cluster_size=100, nearest_neighbors=10
    )
    num_clusters = len(set(clusters.keys()))
    assert num_clusters == 14
    assert warning_message in caplog.text


def test_get_pairwise_metrics():
    """Tests the get_pairwise_metrics function"""
    filename = os.path.dirname(os.path.realpath(__file__))[:-12]  # Primo folder
    filename += "//data_parser//tests//random_well_data.csv"

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
    )

    wd = WellData(data=filename, column_names=col_names)
    well_list = wd.data.head(4).index.to_list()  # Retaining only three wells

    pair_metrics = get_pairwise_metrics(wd, well_list)
    assert len(pair_metrics) == 6
    assert "dist_range" in pair_metrics.columns
    assert "age_range" in pair_metrics.columns
    assert "depth_range" in pair_metrics.columns
    assert list(pair_metrics.index) == list(combinations(well_list, 2))


def test_perform_boundary_clustering_no_subclustering():
    """Tests the perform_boundary_clustering function when no sub-clustering is necessary"""

    filename = (
        Path(__file__).resolve().parent.parent.parent
        / "data_parser"
        / "tests"
        / "random_well_data.csv"
    )

    data = pd.read_csv(filename)
    num_wells = len(data)
    data["County"] = ["County 1", "County 2", "County 3", "County 4", "County 5"] * (
        num_wells // 5
    )

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        geographic_boundary="County",
    )

    wd = WellData(data=data, column_names=col_names)

    clusters = perform_boundary_clustering(wd, threshold_cluster_size=num_wells)

    assert len(clusters.keys()) == 5
    assert "Clusters" in wd.data.columns
    assert len(wd.data["Clusters"].unique()) == 5


def test_perform_boundary_clustering_subclustering():
    """Tests the perform_boundary_clustering function when sub-clustering is necessary"""

    filename = (
        Path(__file__).resolve().parent.parent.parent
        / "data_parser"
        / "tests"
        / "random_well_data.csv"
    )

    data = pd.read_csv(filename)
    num_wells = len(data)
    data["County"] = ["County 1", "County 2", "County 3", "County 4", "County 5"] * (
        num_wells // 5
    )

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        geographic_boundary="County",
    )

    wd = WellData(data=data, column_names=col_names)

    clusters = perform_boundary_clustering(
        wd,
        subcluster_method="Agglomerative",
        max_boundary_cluster_size=1,  # all counties will require sub-clustering
        threshold_distance=200,
    )

    assert len(clusters.keys()) == 10
    assert "Clusters" in wd.data.columns
    assert len(wd.data["Clusters"].unique()) == 10

    for county in data["County"].unique():
        clusters_in_county = wd.data[wd.data["County"] == county]["Clusters"].unique()
        assert len(clusters_in_county) == 2
        # Check that the clusters in each county are distinct
        assert not set(clusters_in_county).intersection(
            set(wd.data[wd.data["County"] != county]["Clusters"].unique())
        )


@pytest.mark.parametrize(
    "max_cost_project, max_wells_project, result",
    [
        # enforced by maximum number of wells with a maximum cost per project
        (350, 3, 300),
        # enforced by maximum number of wells without a maximum cost per project
        (None, 4, 400),
        # enforced by maximum cost per project without a maximum number of wells
        (350, None, 350),
        # enforced by maximum cost per project with a maximum number of wells
        (350, 5, 350),
    ],
)
def test_get_enforced_max_cost_per_project(max_cost_project, max_wells_project, result):
    """Tests the get_enforced_max_cost_per_project function"""

    enforced_max_cost = get_enforced_max_cost_per_project(
        max_cost_project,
        max_wells_project,
        mobilization_cost={1: 100, 2: 200, 3: 300, 4: 400, 5: 500},
    )

    assert enforced_max_cost == result

    enforced_max_cost = get_enforced_max_cost_per_project(
        max_cost_project,
        max_wells_project,
        mobilization_cost=None,
        average_cost_per_well=100,
    )

    assert enforced_max_cost == result


def test_get_enforced_max_cost_per_project_no_constraints():
    """Tests the get_enforced_max_cost_per_project function when no constraints are provided"""

    with pytest.raises(ValueError):
        get_enforced_max_cost_per_project(
            None,
            None,
            mobilization_cost={1: 100, 2: 200, 3: 300, 4: 400, 5: 500},
        )

    with pytest.raises(ValueError):
        get_enforced_max_cost_per_project(
            None,
            None,
            mobilization_cost=None,
            average_cost_per_well=100,
        )


@pytest.mark.parametrize(
    "min_cost_project, plugging_costs, result",
    [
        # working case
        (350, pd.Series([100, 200, 300, 400, 500]), 3),
        # another working
        (10, pd.Series([1, 2, 2, 2, 2, 2, 2, 2, 2, 3]), 6),
        # working case
        (3, pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3),
        # failing case where the minimum project cost is larger than the total plugging cost
        (11, pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), None),
        # failing case where there are missing values in the plugging cost data
        (5, pd.Series([np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1]), None),
    ],
)
def test_min_project_cost_to_num_wells_cost_of_plugging(
    min_cost_project, plugging_costs, result
):
    """Tests the min_project_cost_to_num_wells_cost_of_plugging function"""

    if result is not None:
        assert (
            min_project_cost_to_num_wells_cost_of_plugging(
                min_cost_project, plugging_costs
            )
            == result
        )

    else:
        with pytest.raises(ValueError):
            min_project_cost_to_num_wells_cost_of_plugging(
                min_cost_project, plugging_costs
            )


@pytest.mark.parametrize(
    "min_cost_project, result",
    [
        # working case
        (300, 3),
        # working case between costs
        (350, 4),
        # failing case where the minimum project cost is larger than the cost to plug all wells
        (600, None),
    ],
)
def test_min_project_cost_to_num_wells_mobilization_cost(min_cost_project, result):
    """Tests the min_project_cost_to_num_wells_mobilization_cost function"""

    mobilization_cost = {1: 100, 2: 200, 3: 300, 4: 400, 5: 500}

    if result is not None:
        num_wells = min_project_cost_to_num_wells_mobilization_cost(
            min_cost_project, mobilization_cost
        )

        assert num_wells == result

    else:
        with pytest.raises(ValueError):
            min_project_cost_to_num_wells_mobilization_cost(
                min_cost_project, mobilization_cost
            )


@pytest.fixture
def make_well_data():
    """Factory fixture building a WellData from random_well_data.csv.

    tail: number of rows to keep from the end of the data (None keeps all).
    cost_of_plugging: value to fill a "Cost of Plugging" column with; the
        column (and WellDataColumnNames.cost_of_plugging) is omitted if None.
    """

    def _make_well_data(tail, cost_of_plugging):
        filename = os.path.dirname(os.path.realpath(__file__))[:-12]  # Primo folder
        filename += "//data_parser//tests//random_well_data.csv"

        data = pd.read_csv(filename)
        data["Priority Score"] = 100  # filling to avoid error

        if cost_of_plugging is not None:
            data["Cost of Plugging"] = cost_of_plugging

        col_names = WellDataColumnNames(
            well_id="API Well Number",
            latitude="x",
            longitude="y",
            operator_name="Operator Name",
            age="Age [Years]",
            depth="Depth [ft]",
            cost_of_plugging=(
                "Cost of Plugging" if cost_of_plugging is not None else None
            ),
            additional_columns={
                "priority_score": "Priority Score",
            },
        )

        if tail is not None:
            data = data.tail(tail)
        data = data.reset_index(drop=True)

        return WellData(data=data, column_names=col_names)

    return _make_well_data


@pytest.mark.parametrize(
    "kwargs, clusters, result",
    [
        # constrained by max project cost
        (
            {"max_cost_project": 500},
            {0: [1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10, 11, 12]},
            False,
        ),
        # budget sufficient case
        ({"max_cost_project": 500}, {0: [1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10]}, True),
        # one constraint, can spend budget
        (
            {"max_cost_project": 500},
            {0: [1, 2, 3, 4], 1: [5, 6, 7], 2: [8, 9, 10]},
            True,
        ),
        # constrained by max project size
        (
            {"max_cost_project": 600, "max_wells_in_project": 5},
            {0: [1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10, 11, 12]},
            False,
        ),
        # both constraints, can spend budget
        (
            {"max_cost_project": 600, "max_wells_in_project": 5},
            {0: [1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10], 2: [11, 12]},
            True,
        ),
        # number of clusters is sufficient, but constrained by tiny cluster
        (
            {"max_cost_project": 500},
            {0: [1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10, 11], 2: [12]},
            False,
        ),
    ],
)
def test_can_spend_full_budget(kwargs, clusters, result, make_well_data):
    """
    Tests the can_spend_full_budget function
    """
    wd = make_well_data(tail=12, cost_of_plugging=None)

    mobilization_cost = {
        1: 100,
        2: 200,
        3: 300,
        4: 400,
        5: 500,
        6: 600,
        7: 700,
        8: 800,
        9: 900,
        10: 1000,
        11: 1100,
        12: 1200,
    }

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=1100,
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
        **kwargs,
    )

    assert can_spend_full_budget(clusters, model_inputs) is result


@pytest.mark.parametrize(
    "kwargs, clusters, budget, result",
    [
        # constrained by max project cost
        (
            {"max_cost_project": 500},
            {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11]},
            1100,
            False,
        ),
        # budget sufficient case
        (
            {"max_cost_project": 500},
            {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10, 11]},
            1300,
            True,
        ),
        # one constraint, can spend budget
        (
            {"max_cost_project": 500},
            {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]},
            1100,
            True,
        ),
        # constrained by max project size
        (
            {"max_cost_project": 600, "max_wells_in_project": 5},
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9, 10, 11]},
            1100,
            False,
        ),
        # both constraints, can spend budget
        (
            {"max_cost_project": 600, "max_wells_in_project": 5},
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9], 2: [10, 11]},
            1100,
            True,
        ),
        # number of clusters is sufficient, but constrained by tiny cluster
        (
            {"max_cost_project": 500},
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9, 10], 2: [11]},
            1100,
            False,
        ),
    ],
)
def test_can_spend_full_budget_cost_of_plugging(
    kwargs, clusters, budget, result, make_well_data
):
    """
    Tests the can_spend_full_budget function with cost of plugging data
    """
    wd = make_well_data(tail=12, cost_of_plugging=100)

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=budget,
        cluster_mapping=clusters,
        mobilization_cost=None,
        objective_weight_impact=100,
        **kwargs,
    )

    assert can_spend_full_budget(clusters, model_inputs) is result


def test_can_spend_full_budget_cost_of_plugging_missing(make_well_data):
    """
    Tests can_spend_full_budget with cost of plugging data that includes a missing value
    """
    wd = make_well_data(tail=12, cost_of_plugging=100)

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=1100,
        mobilization_cost=None,
        objective_weight_impact=100,
        min_cost_project=350,
    )

    wd.data.loc[0, "Cost of Plugging"] = np.nan  # introduce a missing value

    clusters = {0: [0, 1, 2], 1: [3, 4, 5, 6, 7, 8, 9, 10, 11]}

    with pytest.raises(ValueError):
        can_spend_full_budget(clusters, model_inputs)


@pytest.mark.parametrize(
    "kwargs, clusters, result",
    [
        ({"min_wells_in_project": 3}, {0: [1, 2], 1: [3, 4, 5, 6, 7, 8, 9, 10]}, False),
        ({"min_wells_in_project": 3}, {0: [1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10]}, True),
        ({"min_cost_project": 350}, {0: [1, 2, 3], 1: [4, 5, 6, 7, 8, 9, 10]}, False),
        ({"min_cost_project": 350}, {0: [1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10]}, True),
    ],
)
def test_are_minimum_constraints_satisfied(kwargs, clusters, result, make_well_data):
    """
    Tests the are_minimum_constraints_satisfied function
    """
    wd = make_well_data(tail=12, cost_of_plugging=None)

    mobilization_cost = {
        1: 100,
        2: 200,
        3: 300,
        4: 400,
        5: 500,
        6: 600,
        7: 700,
        8: 800,
        9: 900,
        10: 1000,
    }

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=1100,
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
        **kwargs,
    )

    assert are_minimum_constraints_satisfied(clusters, model_inputs) is result


@pytest.mark.parametrize(
    "kwargs, clusters, result",
    [
        (
            {"min_cost_project": 350},
            {0: [0, 1, 2], 1: [3, 4, 5, 6, 7, 8, 9, 10, 11]},
            False,
        ),
        (
            {"min_cost_project": 350},
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9, 10, 11]},
            True,
        ),
    ],
)
def test_are_minimum_constraints_satisfied_cost_of_plugging(
    kwargs, clusters, result, make_well_data
):
    """
    Tests the are_minimum_constraints_satisfied function with cost of plugging data
    """
    wd = make_well_data(tail=12, cost_of_plugging=100)

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=1100,
        cluster_mapping=clusters,
        mobilization_cost=None,
        objective_weight_impact=100,
        **kwargs,
    )

    assert are_minimum_constraints_satisfied(clusters, model_inputs) is result


DEFAULT_CONFIG = {
    "uniformity": 0,
    "max_iter": 10,
    "target_clusters": 10,
    "max_wells_in_project": 50,
    "min_wells_in_project": 5,
    "budget": 5e6,
}


@pytest.mark.parametrize(
    "objective, config_override, result",
    [
        ("minimum", {}, 4),
        ("maximum", {"uniformity": 0.25}, 55),
        ("target", {"uniformity": 0.5}, 10),
        ("center", {"uniformity": 1}, 28),
        (
            "target",
            {"max_iter": 1, "target_clusters": 1, "max_wells_in_project": 20},
            None,
        ),
        (
            "center",
            {"min_wells_in_project": 35, "max_wells_in_project": 25, "budget": 13e6},
            None,
        ),
    ],
)
def test_perform_heuristic_constrained_clustering(
    objective, config_override, result, make_well_data
):
    """
    Tests for perform_heuristic_constrained_clustering method
    """
    config = {**DEFAULT_CONFIG, **config_override}
    wd = make_well_data(tail=None, cost_of_plugging=50000)

    if result is not None:
        opt_model_inputs = OptModelInputs(
            well_data=wd,
            total_budget=config["budget"],
            max_cost_project=None,
            min_cost_project=None,
            max_wells_in_project=config["max_wells_in_project"],
            min_wells_in_project=config["min_wells_in_project"],
            cluster_method="Constrained_Heuristic",
            constrained_clustering_objective=objective,
            target_number_of_clusters=config["target_clusters"],
            uniformity_parameter=config["uniformity"],
            objective_weight_impact=100,
        )

        clusters = opt_model_inputs.campaign_candidates
        assert len(clusters) == result

    else:
        with pytest.raises(RuntimeError):
            opt_model_inputs = OptModelInputs(
                well_data=wd,
                total_budget=config["budget"],
                max_cost_project=None,
                min_cost_project=None,
                max_wells_in_project=config["max_wells_in_project"],
                min_wells_in_project=config["min_wells_in_project"],
                cluster_method="Constrained_Heuristic",
                constrained_clustering_objective=objective,
                target_number_of_clusters=config["target_clusters"],
                uniformity_parameter=config["uniformity"],
                constrained_clustering_max_iterations=config["max_iter"],
                objective_weight_impact=100,
            )


@pytest.mark.parametrize(
    "max_wells_in_project, max_cost_project, result",
    [
        # no bounds
        (None, None, 2),
        # max cost only
        (None, 5000, 1),
        # max size only
        (50, None, 1),
        # both max cost and max size, constrained by max cost
        (10, 5000, 5),
        # both max cost and max size, constrained by max size
        (50, 1000, 5),
    ],
)
def test_estimate_min_clusters(
    max_wells_in_project, max_cost_project, result, make_well_data
):
    """
    Tests for _estimate_min_clusters method
    """
    wd = make_well_data(tail=None, cost_of_plugging=None)

    mobilization_cost = {i: 100 * i for i in range(1, len(wd.data) + 1)}

    opt_model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=5000,
        max_cost_project=max_cost_project,
        min_cost_project=None,
        max_wells_in_project=max_wells_in_project,
        min_wells_in_project=None,
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
    )

    assert _estimate_min_clusters(opt_model_inputs) == result


@pytest.mark.parametrize(
    "min_wells_in_project, min_cost_project, result",
    [
        # no bounds
        (None, None, 1),
        # min cost only
        (None, 500, 5),
        # min size only
        (5, None, 5),
        # both min cost and min size, constrained by min cost
        (4, 500, 5),
        # both min cost and min size, constrained by min size
        (5, 400, 5),
    ],
)
def test_estimate_min_project_size(
    min_wells_in_project, min_cost_project, result, make_well_data
):
    """
    Tests for _estimate_min_project_size method
    """
    wd = make_well_data(tail=None, cost_of_plugging=None)

    mobilization_cost = {i: 100 * i for i in range(1, len(wd.data) + 1)}

    opt_model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=5000,
        max_cost_project=None,
        min_cost_project=min_cost_project,
        max_wells_in_project=None,
        min_wells_in_project=min_wells_in_project,
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
    )

    assert _estimate_min_project_size(opt_model_inputs) == result

    # testing with cost of plugging
    wd = make_well_data(tail=None, cost_of_plugging=100)

    opt_model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=5000,
        max_cost_project=None,
        min_cost_project=min_cost_project,
        max_wells_in_project=None,
        min_wells_in_project=min_wells_in_project,
        mobilization_cost=None,
        objective_weight_impact=100,
    )

    assert _estimate_min_project_size(opt_model_inputs) == result


@pytest.mark.parametrize(
    "min_size, result",
    [
        # no minimum size
        (None, 27),
        # feasible minimum sizes
        (100, 2),
        (200, 1),
        # infeasible constraint, 275 wells total
        (300, None),
    ],
)
def test_estimate_max_clusters(min_size, result, make_well_data):
    """
    Tests for _estimate_max_clusters method
    """
    wd = make_well_data(tail=None, cost_of_plugging=None)

    mobilization_cost = {i: 100 * i for i in range(1, len(wd.data) + 1)}

    opt_model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=5000,
        max_cost_project=None,
        min_cost_project=None,
        max_wells_in_project=None,
        min_wells_in_project=min_size,
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
    )

    if result is None:
        with pytest.raises(ValueError):
            _estimate_max_clusters(opt_model_inputs, min_size)
    else:
        assert _estimate_max_clusters(opt_model_inputs, min_size) == result


def test_get_pairwise_metrics_single_well():
    """Tests pairwise metrics for a single-well project"""
    well_df = pd.DataFrame(
        [
            {
                "Well API": "1",
                "Latitude": 40.0,
                "Longitude": -70,
                "Age [Years]": 20,
                "Depth [ft]": 1000,
                "Op Name": "Owner 1",
            }
        ]
    )
    col_names = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Op Name",
    )
    wd = WellData(well_df, col_names)
    well_list = wd.data.head(1).index.to_list()
    pair_metrics = get_pairwise_metrics(wd, well_list)
    assert pair_metrics.empty
