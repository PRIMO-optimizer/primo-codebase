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

# Installed libs
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

# User-defined libs
from primo import OptModelInputs
from primo.data_parser import WellDataColumnNames
from primo.data_parser.well_data import WellData
from primo.utils.mip_clustering_utils import (
    MIPNonOverlappingClustering,
    MIPOverlappingClustering,
)

# pylint: disable=redefined-outer-name, protected-access

# Group A: wells 0-2 tightly clustered near (40.0, -80.0)
# Group B: wells 3-5 tightly clustered near (42.0, -80.0), ~138 miles from Group A
_GROUP_A_LAT = [40.000, 40.001, 40.002]
_GROUP_B_LAT = [42.000, 42.001, 42.002]


@pytest.fixture
def two_group_wd():
    """Factory fixture: 6-well WellData in two well-separated geographic groups"""

    def _make(cost_of_plugging=None):
        well_data = [
            {
                "Well ID": f"W{i}",
                "Latitude": lat,
                "Longitude": -80.0,
                "Age [Years]": 20,
                "Depth [ft]": 1000,
                "Op Name": f"Owner {i}",
                "Priority Score": 100,
            }
            for i, lat in enumerate(_GROUP_A_LAT + _GROUP_B_LAT)
        ]
        well_df = pd.DataFrame(well_data)
        if cost_of_plugging is not None:
            well_df["Cost of Plugging"] = cost_of_plugging

        well_cn = WellDataColumnNames(
            well_id="Well ID",
            latitude="Latitude",
            longitude="Longitude",
            age="Age [Years]",
            depth="Depth [ft]",
            operator_name="Op Name",
            cost_of_plugging=(
                "Cost of Plugging" if cost_of_plugging is not None else None
            ),
            additional_columns={"priority_score": "Priority Score"},
        )
        return WellData(well_df, well_cn)

    return _make


# ---------------------------------------------------------------------------
# A. get_exhaustive_clustering
# ---------------------------------------------------------------------------


def test_get_exhaustive_clustering_small_threshold(two_group_wd):
    """Groups should stay separated"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd)
    clusters, distances = mip.get_exhaustive_clustering(threshold_distance=5)

    assert set(clusters.keys()) == {0, 1, 2, 3, 4, 5}
    for well in (0, 1, 2):
        assert set(clusters[well]) == {0, 1, 2}
    for well in (3, 4, 5):
        assert set(clusters[well]) == {3, 4, 5}
        assert distances[well][well] == 0


def test_get_exhaustive_clustering_large_threshold(two_group_wd):
    """Groups should merge"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd)
    clusters, _ = mip.get_exhaustive_clustering(threshold_distance=200)

    for nearby_wells in clusters.values():
        assert set(nearby_wells) == {0, 1, 2, 3, 4, 5}


def test_get_exhaustive_clustering_max_num_wells(two_group_wd):
    """Clusters should be truncated to the max_num_wells limit"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd, max_num_wells=2)
    clusters, _ = mip.get_exhaustive_clustering(threshold_distance=5)

    for nearby_wells in clusters.values():
        assert len(nearby_wells) <= 2


# ---------------------------------------------------------------------------
# B. build_simple_model (no solving)
# ---------------------------------------------------------------------------


def test_build_simple_model(two_group_wd):
    """Tests that model has the expected wells and binary variables"""
    wd = two_group_wd()
    mip = MIPOverlappingClustering(wd=wd)
    m = mip.build_model(threshold_distance=5)

    assert set(m.wells_set) == {0, 1, 2, 3, 4, 5}
    for well in m.wells_set:
        assert m.select_cluster[well].domain is pyo.Binary


@pytest.mark.parametrize(
    "min_num_wells, min_project_cost",
    [
        (4, None),  # min_num_wells constraint active
        (None, 350),  # min_project_cost constraint active
    ],
)
def test_build_simple_model_fixed_clusters(
    two_group_wd, min_num_wells, min_project_cost
):
    """Restrictive constraints result in all clusters fixed to 0"""
    wd = two_group_wd(cost_of_plugging=100)
    mip = MIPOverlappingClustering(
        wd=wd, min_num_wells=min_num_wells, min_project_cost=min_project_cost
    )
    m = mip.build_model(threshold_distance=5)

    # Every cluster only has 3 members (within-group), so all get fixed to 0
    for well in m.wells_set:
        assert m.select_cluster[well].fixed
        assert m.select_cluster[well].value == 0


@pytest.mark.parametrize(
    "min_num_wells, min_project_cost",
    [
        (4, None),  # min_num_wells constraint active
        (None, 350),  # min_project_cost constraint active
    ],
)
def test_build_simple_model_fixed_clusters_mob_cost(
    two_group_wd, min_num_wells, min_project_cost
):
    """Tests build_simple_model where restrictive constraints result in all clusters fixed to 0"""
    wd = two_group_wd()
    costs = {i + 1: 100 * (i + 1) for i in range(len(wd))}
    mip = MIPOverlappingClustering(
        wd=wd,
        min_num_wells=min_num_wells,
        min_project_cost=min_project_cost,
        mobilization_cost=costs,
    )
    m = mip.build_model(threshold_distance=5)

    # Every cluster only has 3 members (within-group), so all get fixed to 0
    for well in m.wells_set:
        assert m.select_cluster[well].fixed
        assert m.select_cluster[well].value == 0


@pytest.mark.parametrize(
    "min_num_wells, min_project_cost",
    [
        (3, None),  # min_num_wells constraint active
        (None, 250),  # min_project_cost constraint active
    ],
)
def test_build_simple_model_unfixed_clusters(
    two_group_wd, min_num_wells, min_project_cost
):
    """Tests build_simple_model where constraints do not result in any clusters being fixed to 0"""
    wd = two_group_wd(cost_of_plugging=100)
    mip = MIPOverlappingClustering(
        wd=wd, min_num_wells=min_num_wells, min_project_cost=min_project_cost
    )
    m = mip.build_model(threshold_distance=5)

    for well in m.wells_set:
        assert not m.select_cluster[well].fixed


def test_build_simple_model_fixed_num_clusters(two_group_wd):
    """Tests build_simple_model with fix_num_clusters argument"""
    wd = two_group_wd()
    mip = MIPOverlappingClustering(wd=wd)
    # Invalid objective_type is never validated because of the early return
    m = mip.build_model(threshold_distance=5, fix_num_clusters=2)

    assert hasattr(m, "fixing_num_clusters")
    assert m.obj.sense == pyo.minimize


@pytest.mark.parametrize(
    "objective_type", ["num_clusters", "cluster_size", "cluster_radius"]
)
def test_build_simple_model_valid_objective_types(two_group_wd, objective_type):
    """Tests build_simple_model with valid objective_type values"""
    wd = two_group_wd()
    mip = MIPOverlappingClustering(wd=wd)
    mip.objective_type = objective_type
    m = mip.build_model(threshold_distance=5)

    assert m.obj.sense == pyo.minimize


def test_build_simple_model_invalid_objective_type(two_group_wd):
    """Tests build_simple_model with invalid objective_type value"""
    wd = two_group_wd()
    mip = MIPOverlappingClustering(wd=wd)

    with pytest.raises(
        ValueError, match="Unrecognized value fake_objective for objective_type"
    ):
        mip.objective_type = "fake_objective"


# ---------------------------------------------------------------------------
# C. build_detailed_model (no solving)
# ---------------------------------------------------------------------------


def test_build_detailed_model_baseline(two_group_wd):
    """Tests build_detailed_model with no optional arguments"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd)
    m = mip.build_model(threshold_distance=5)

    assert set(m.wells_set) == set(wd.data.index)
    assert set(m.clusters_set) == {0, 1, 2, 3, 4, 5}
    assert not hasattr(m, "ensure_enough_clusters_exist")


def test_build_detailed_model_min_num_wells(two_group_wd):
    """Tests build_detailed_model with min_num_wells argument"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd, min_num_wells=2)
    m = mip.build_model(threshold_distance=5)

    blk = m.cluster[0]
    assert len(blk.essential_wells_set) == 2
    assert len(blk.optional_wells_set) == 1
    assert hasattr(blk, "ordering_of_selected_wells")


def test_build_detailed_model_disabled_clusters(two_group_wd):
    """Tests build_detailed_model with min_num_wells argument that is too large"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(
        wd=wd, min_num_wells=5
    )  # groups only have 3 wells
    m = mip.build_model(threshold_distance=5)

    for i in m.clusters_set:
        blk = m.cluster[i]
        assert blk.select_cluster.fixed
        assert blk.select_cluster.value == 0
        assert set(blk.essential_wells_set) == {i}
        assert set(blk.optional_wells_set) == set()
        assert pyo.value(blk.cluster_radius) == 0


def test_build_detailed_model_ensure_enough_clusters_exist_1(two_group_wd):
    """Tests build_detailed_model with max_project_cost and total_budget arguments"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd, max_project_cost=100, total_budget=250)
    m = mip.build_model(threshold_distance=5)

    assert hasattr(m, "ensure_enough_clusters_exist")
    assert m.ensure_enough_clusters_exist.lower == pytest.approx(np.ceil(250 / 100))


def test_build_detailed_model_ensure_enough_clusters_exist_2(two_group_wd):
    """Tests build_detailed_model with max_num_wells and total_budget arguments using cost of
    plugging"""
    wd = two_group_wd(cost_of_plugging=50)
    mip = MIPNonOverlappingClustering(
        wd=wd, max_num_wells_in_project=3, total_budget=250
    )
    m = mip.build_model(threshold_distance=5)

    assert hasattr(m, "ensure_enough_clusters_exist")
    assert m.ensure_enough_clusters_exist.lower == pytest.approx(np.ceil(250 / 150))


def test_build_detailed_model_ensure_enough_clusters_exist_3(two_group_wd):
    """Tests build_detailed_model with max_num_wells and total_budget arguments using mobilization
    cost"""
    wd = two_group_wd()
    costs = {i + 1: 50 * (i + 1) for i in range(len(wd))}
    mip = MIPNonOverlappingClustering(
        wd=wd, max_num_wells_in_project=3, total_budget=250, mobilization_cost=costs
    )
    m = mip.build_model(threshold_distance=5)

    assert hasattr(m, "ensure_enough_clusters_exist")
    assert m.ensure_enough_clusters_exist.lower == pytest.approx(np.ceil(250 / 150))


def test_build_detailed_model_fix_num_clusters(two_group_wd):
    """Tests build_detailed_model with fix_num_clusters argument"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd)
    mip.objective_type = "num_clusters"
    m = mip.build_model(threshold_distance=5, fix_num_clusters=2)

    assert hasattr(m, "constrain_num_clusters")
    assert m.constrain_num_clusters.upper == 2
    # pylint: disable=no-member
    assert m.constrain_num_clusters.lower == 2


def test_build_detailed_model_invalid_objective_type(two_group_wd):
    """Tests build_detailed_model with invalid objective_type value"""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd)

    with pytest.raises(
        ValueError, match="Unrecognized value fake_objective for objective_type"
    ):
        mip.objective_type = "fake_objective"


# ---------------------------------------------------------------------------
# D. Error handling for missing cost information
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["model_type", "trigger_variable"],
    [
        ("MIP_Overlapping", "min_project_cost"),
        ("MIP_NonOverlapping", "min_project_cost"),
        ("MIP_NonOverlapping", "max_num_wells"),
    ],
)
def test_missing_cost_error_handling(two_group_wd, model_type, trigger_variable):
    """Tests various situations where costs haven't been provided to the model"""
    wd = two_group_wd()
    match = "Well costs are required but have not been provided"

    if model_type == "MIP_Overlapping":
        mip = MIPOverlappingClustering(wd=wd)
    else:
        mip = MIPNonOverlappingClustering(wd=wd)

    if trigger_variable == "min_project_cost":
        mip.min_project_cost = 50
    else:
        mip.max_num_wells_in_project = 6
        mip.total_budget = 400

    with pytest.raises(ValueError, match=match):
        _ = mip.perform_clustering(threshold_distance=5)


# ---------------------------------------------------------------------------
# E. perform_clustering (real solves)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type, threshold_distance",
    [
        ("overlapping", 5),
        ("overlapping", 200),
        ("non-overlapping", 5),
        ("non-overlapping", 200),
    ],
)
def test_perform_clustering(two_group_wd, model_type, threshold_distance):
    """Threshold distances below ~138 miles should give 1 cluster and
    distances above should give 2 clusters"""
    wd = two_group_wd()
    mip = (
        MIPNonOverlappingClustering(wd=wd)
        if model_type == "non-overlapping"
        else MIPOverlappingClustering(wd=wd)
    )
    mip.objective_type = "num_clusters"
    clusters = mip.perform_clustering(threshold_distance=threshold_distance)

    if threshold_distance == 5:
        assert len(clusters) == 2
    else:
        assert len(clusters) == 1

    all_wells = [w for wells in clusters.values() for w in wells]
    assert sorted(all_wells) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("model_type", ["overlapping", "non-overlapping"])
def test_fixing_num_clusters(two_group_wd, model_type):
    """Threshold distances below ~138 miles should give 1 cluster and
    distances above should give 2 clusters"""
    wd = two_group_wd()
    mip = (
        MIPNonOverlappingClustering(wd=wd)
        if model_type == "non-overlapping"
        else MIPOverlappingClustering(wd=wd)
    )
    mip.objective_type = "num_clusters"
    clusters, _ = mip.find_threshold_distance(
        start_value=5, increment=5, num_clusters=2
    )

    assert mip.objective_type == "cluster_radius"

    all_wells = [w for wells in clusters.values() for w in wells]
    assert sorted(list(set(all_wells))) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# F. find_threshold_distance (real solves)
# ---------------------------------------------------------------------------


def test_find_threshold_distance(two_group_wd):
    """Tests find_threshold_distance. First attempt at 100 fails, second attempt at 150 succeeds.
    The two groups need to merge to satisfy the min_num_wells=4 constraint."""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd, min_num_wells=4)

    clusters, td = mip.find_threshold_distance(start_value=100, increment=50)

    assert td == 150
    assert len(clusters) == 1


def test_find_threshold_distance_fail(two_group_wd):
    """Tests find_threshold_distance. Problem is infeasible since there are only 6 wells
    and min_num_wells=7."""
    wd = two_group_wd()
    mip = MIPNonOverlappingClustering(wd=wd, min_num_wells=7)

    with pytest.raises(RuntimeError):
        mip.find_threshold_distance(start_value=100, increment=50)


# ---------------------------------------------------------------------------
# G. Accessing clustering through OptModelInputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["MIP_NonOverlapping", "MIP_Overlapping"])
def test_mip_clustering_through_model_options(two_group_wd, model_type):
    """Tests accessing MIP clustering through the OptModelInputs class using various options"""
    wd = two_group_wd(cost_of_plugging=100)

    threshold_radius = 10

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=1000,
        threshold_distance=threshold_radius * 2,
        objective_weight_impact=100,
        cluster_method=model_type,
    )

    if model_type == "MIP_Overlapping":
        assert model_inputs.has_overlapping_clusters is True
        assert "Clusters" not in wd.data.columns
    else:
        assert model_inputs.has_overlapping_clusters is False
        assert "Clusters" in wd.data.columns


def test_mip_clustering_through_model_options_td_update(two_group_wd):
    """Tests that the threshold distance is updated in the config after MIP clustering"""
    wd = two_group_wd(cost_of_plugging=100)

    threshold_radius = 100

    model_inputs = OptModelInputs(
        well_data=wd,
        total_budget=1000,
        min_wells_in_project=4,
        threshold_distance=threshold_radius * 2,
        threshold_distance_increment=50,
        objective_weight_impact=100,
        cluster_method="MIP_NonOverlapping",
    )

    assert model_inputs.config.threshold_distance / 2 == 150
