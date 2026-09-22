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
# pylint: disable=too-many-lines

# Standard libs
import copy
import logging
import pathlib
from itertools import combinations

# Installed libs
import numpy as np
import pyomo.environ as pe
import pytest

# User-defined libs
from primo.data_parser import EfficiencyMetrics, ImpactMetrics, WellDataColumnNames
from primo.data_parser.well_data import WellData

# pylint: disable=no-name-in-module
from primo.opt_model.cluster_block import IndexedClusterBlock
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.model_with_clustering import PluggingCampaignModel
from primo.opt_model.result_parser import Campaign, Project
from primo.opt_model.tests.test_efficiency_model import (
    get_column_names_fixture as efficiency_get_column_names_fixture,
)
from primo.utils.clustering_utils import distance_matrix
from primo.utils.config_utils import (
    OverrideAddInfo,
    OverrideRemoveLockInfo,
    OverrideSelections,
)
from primo.utils.override_utils import OverrideCampaign, ReOptimizationData

LOGGER = logging.getLogger(__name__)


# pylint: disable=duplicate-code
# pylint: disable=too-many-lines


@pytest.fixture(name="get_efficiency_column_names", scope="function")
def get_efficiency_column_names_fixture():
    """Reuse the efficiency-model test fixture in this test module."""
    return efficiency_get_column_names_fixture.__wrapped__()


# pylint: disable=missing-function-docstring
@pytest.fixture(name="get_column_names", scope="function")
def get_column_names_fixture():
    """
    Pytest fixture to set up the impact metric, assign
    column names, and read the test data.
    """

    # Define impact metrics by creating an instance of ImpactMetrics class
    im_metrics = ImpactMetrics()

    # Specify weights
    im_metrics.set_weight(
        primary_metrics={
            "well_history": 35,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_age": 15,
            "well_count": 10,
        },
        submetrics={
            "well_history": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )

    # Construct an object to store column names
    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        hospitals="Number of Nearby Hospitals",
        schools="Number of Nearby Schools",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        # These are user-specific columns
        elevation_delta="Elevation Delta [m]",
        dist_to_road="Distance to Road [miles]",
    )

    current_file = pathlib.Path(__file__).resolve()
    # primo folder is 2 levels up the current folder
    data_file = str(
        current_file.parents[2].joinpath("demo", "Example_Workflow_data.csv")
    )
    return im_metrics, col_names, data_file


@pytest.mark.parametrize(
    "cluster_method, num_projects",
    [
        ("Louvain", [5, 6]),
        ("Agglomerative", [4, 5]),
    ],
)
def test_opt_model_inputs(get_column_names, cluster_method, num_projects):
    """
    Test that the optimization model is constructed and solved correctly.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches

    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Catch inputs missing error
    with pytest.raises(
        ValueError,
        match=("Mobilization cost is an essential input for the optimization model."),
    ):
        opt_mdl_inputs = OptModelInputs(well_data=wd, total_budget=3250000)

    # Catch priority score missing error
    with pytest.raises(
        ValueError,
        match=(
            "Unable to find priority scores in the WellData object. Compute the scores "
            "using the compute_priority_scores method."
        ),
    ):
        opt_mdl_inputs = OptModelInputs(
            well_data=wd_gas,
            total_budget=3250000,  # 3.25 million USD
            mobilization_cost=mobilization_cost,
        )

    # Compute priority scores
    # Test the model and options
    wd_gas.compute_priority_scores()

    assert "Clusters" not in wd_gas

    if cluster_method == "Louvain":
        threshold_distance = 10
    else:
        threshold_distance = 20
    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3250000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=threshold_distance,
        max_wells_per_owner=1,
        min_budget_usage=50,
        penalize_unused_budget=True,
        cluster_method=cluster_method,
        objective_weight_impact=100,
    )

    # Ensure that clustering is performed internally
    assert "Clusters" in wd_gas

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    opt_mdl = opt_mdl_inputs.optimization_model

    assert hasattr(opt_mdl_inputs, "config")
    assert "Clusters" in wd_gas  # Column is added after clustering
    assert hasattr(opt_mdl_inputs, "campaign_candidates")
    assert hasattr(opt_mdl_inputs, "owner_well_count")

    assert opt_mdl_inputs.get_max_cost_project is None
    assert opt_mdl_inputs.get_total_budget == 3.25

    scaled_mobilization_cost = {1: 0.12, 2: 0.21, 3: 0.28, 4: 0.35}
    for n_wells in range(5, len(wd_gas.data) + 1):
        scaled_mobilization_cost[n_wells] = n_wells * 0.084

    get_mobilization_cost = opt_mdl_inputs.get_mobilization_cost
    for well, cost in scaled_mobilization_cost.items():
        assert np.isclose(get_mobilization_cost[well], cost)

    assert isinstance(opt_mdl, PluggingCampaignModel)
    assert isinstance(opt_campaign, Campaign)
    project_keys = list(opt_campaign.projects.keys())
    example_key = project_keys[0]  # Pick the first available key
    assert isinstance(opt_campaign.projects[example_key], Project)

    # TODO: Confirm degeneracy
    assert len(opt_campaign.projects) in num_projects

    # Test the structure of the optimization model
    num_clusters = len(set(wd_gas["Clusters"]))
    assert hasattr(opt_mdl, "cluster")
    assert len(opt_mdl.cluster) == num_clusters
    assert isinstance(opt_mdl.cluster, IndexedClusterBlock)
    assert hasattr(opt_mdl, "max_well_owner_constraint")
    assert hasattr(opt_mdl, "total_priority_score")

    # Check if the scaling factor for unused budget variable is correctly built
    # pylint: disable=protected-access
    scaling_factor, _, budget_sufficient = opt_mdl._slack_variable_scaling()
    assert np.isclose(scaling_factor, 955.6699386511185)
    assert not budget_sufficient

    # Check if all the cluster sets are defined
    assert hasattr(opt_mdl.cluster[1], "set_wells")
    assert hasattr(opt_mdl.cluster[1], "set_well_pairs_remove")

    # Check if all the required variables are defined
    assert not opt_mdl.cluster[1].select_cluster.is_indexed()
    assert opt_mdl.cluster[1].select_cluster.is_binary()
    assert opt_mdl.cluster[1].select_well.is_indexed()
    for j in opt_mdl.cluster[1].select_well:
        assert opt_mdl.cluster[1].select_well[j].domain == pe.Binary
    assert opt_mdl.cluster[1].num_wells_var.is_indexed()
    for j in opt_mdl.cluster[1].num_wells_var:
        assert opt_mdl.cluster[1].num_wells_var[j].domain == pe.Binary
    assert not opt_mdl.cluster[1].plugging_cost.is_indexed()
    assert opt_mdl.cluster[1].plugging_cost.domain == pe.NonNegativeReals
    assert opt_mdl.cluster[1].num_wells_chosen.domain == pe.NonNegativeReals
    # pylint: disable=no-member
    assert opt_mdl.unused_budget.domain == pe.NonNegativeReals

    # Check if upper bound of the unused budget is defined correctly
    assert opt_mdl.unused_budget.upper is not None

    # Check if the required expressions are defined
    assert hasattr(opt_mdl.cluster[1], "cluster_impact_score")

    # Check if the required constraints are defined
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_chosen")
    assert hasattr(opt_mdl.cluster[1], "calculate_plugging_cost")
    assert hasattr(opt_mdl.cluster[1], "campaign_length")
    assert hasattr(opt_mdl.cluster[1], "num_well_uniqueness")
    assert not hasattr(opt_mdl.cluster[1], "ordering_num_wells_vars")
    if cluster_method == "Louvain":
        assert hasattr(opt_mdl.cluster[1], "skip_distant_well_cuts")

        # Add test to ensure is_distant_pair is zero under Louvain clustering
        well_pair_remove_list = opt_mdl.cluster[1].set_well_pairs_remove
        for well_pair in well_pair_remove_list:
            w1, w2 = well_pair
            assert opt_mdl.cluster[1].is_distant_pair[w1, w2].value == pytest.approx(
                0, rel=1e-4
            )

    # Test activate and deactivate methods
    opt_mdl.cluster[1].deactivate()
    assert opt_mdl.cluster[1].select_cluster.value == 0
    assert opt_mdl.cluster[1].num_wells_chosen.value == 0
    assert opt_mdl.cluster[1].plugging_cost.value == 0
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].num_wells_chosen.is_fixed()
    assert opt_mdl.cluster[1].plugging_cost.is_fixed()

    opt_mdl.cluster[1].activate()
    assert not opt_mdl.cluster[1].select_cluster.is_fixed()
    assert not opt_mdl.cluster[1].num_wells_chosen.is_fixed()
    assert not opt_mdl.cluster[1].plugging_cost.is_fixed()

    # Test fix and unfix methods
    opt_mdl.cluster[1].fix(0)
    # since no arguments are specified only cluster variable is fixed
    # at its incumbent value, which is zero based on earlier operations
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].select_cluster.value == 0
    for j in opt_mdl.cluster[1].select_well:
        assert not opt_mdl.cluster[1].select_well[j].is_fixed()

    opt_mdl.cluster[1].unfix()

    # fix method with only cluster argument
    opt_mdl.cluster[1].fix(cluster=1)
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].select_cluster.value == 1
    for j in opt_mdl.cluster[1].select_well:
        assert not opt_mdl.cluster[1].select_well[j].is_fixed()

    opt_mdl.cluster[1].unfix()

    # fix method with both cluster and well arguments
    opt_mdl.cluster[1].fix(
        cluster=1,
        wells={i: 1 for i in opt_mdl.cluster[1].set_wells},
    )
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].select_cluster.value == 1
    for j in opt_mdl.cluster[1].select_well:
        assert opt_mdl.cluster[1].select_well[j].is_fixed()
        assert opt_mdl.cluster[1].select_well[j].value == 1

    # test override-related items are zeros or not used
    assert opt_mdl.excess_budget.value == pytest.approx(0, rel=1e-4)
    for owner in opt_mdl.excess_owc:
        assert opt_mdl.excess_owc[owner].value == pytest.approx(0, rel=1e-4)
    excess_campaign_length = opt_mdl_inputs.optimization_model.excess_campaign_length()
    excess_campaign_cost = opt_mdl_inputs.optimization_model.excess_campaign_cost()
    assert excess_campaign_length == pytest.approx(0, rel=1e-4)
    assert excess_campaign_cost == pytest.approx(0, rel=1e-4)
    assert opt_mdl.total_constraint_penalty == pytest.approx(0, rel=1e-4)
    assert not hasattr(opt_mdl, "set_clusters_reassign")
    assert not hasattr(opt_mdl, "num_distant_well_pairs")
    assert not hasattr(opt_mdl, "excess_distant_pairs")
    assert not hasattr(opt_mdl, "min_project_size")
    assert not hasattr(opt_mdl, "max_project_size")
    assert hasattr(opt_mdl.cluster[1], "is_distant_pair")
    assert not hasattr(opt_mdl, "budget_status")
    assert not hasattr(opt_mdl, "excess_budget_constraint")
    assert not hasattr(opt_mdl, "unused_budget_constraint")

    if cluster_method == "Louvain":
        assert opt_mdl.cluster[1].set_well_pairs_remove == [
            (31, 925),
            (38, 925),
            (38, 932),
            (38, 939),
            (39, 925),
            (925, 934),
            (934, 939),
        ]
        for well_pair in opt_mdl.cluster[1].set_well_pairs_remove:
            i, j = well_pair
            assert str(opt_mdl.cluster[1].skip_distant_well_cuts[(i, j)].expr) == (
                f"cluster[1].select_well[{i}] + cluster[1].select_well[{j}]  <=  "
                f"cluster[1].select_cluster + cluster[1].is_distant_pair[{i},{j}]"
            )
    else:
        assert opt_mdl.cluster[1].set_well_pairs_remove == []

    cluster_1_wells = list(opt_mdl.cluster[1].set_wells)
    cluster_1_distances = distance_matrix(
        opt_mdl_inputs.config.well_data,
        {"distance": 1},
        cluster_1_wells,
    )
    expected_remove_pairs = {
        (w1, w2)
        for w1, w2 in combinations(cluster_1_wells, 2)
        if cluster_1_distances.loc[w1, w2] > opt_mdl_inputs.config.threshold_distance
    }
    actual_remove_pairs = set(opt_mdl.cluster[1].set_well_pairs_remove)
    assert actual_remove_pairs == expected_remove_pairs

    assert not opt_mdl.cluster[1].min_well_violation.fixed
    assert opt_mdl.cluster[1].min_well_violation.value == pytest.approx(0, rel=1e-4)
    assert not opt_mdl.cluster[1].max_well_violation.fixed
    assert opt_mdl.cluster[1].max_well_violation.value == pytest.approx(0, rel=1e-4)


def test_unused_budget_variable_scaling(get_column_names):
    """
    Test the optimization model when there is enough budget for plugging all wells.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=325000000,  # 325 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        min_budget_usage=50,
        penalize_unused_budget=True,
        objective_weight_impact=100,
    )

    opt_mdl = opt_mdl_inputs.build_optimization_model()

    # Check if the scaling factor for budget slack variable is correctly built
    # pylint: disable=protected-access
    scaling_factor, _, budget_sufficient = opt_mdl._slack_variable_scaling()
    assert np.isclose(scaling_factor, 105.71767887503083)
    assert budget_sufficient

    # Check if the upper bound of the unused budget is set
    assert opt_mdl.unused_budget.upper is None


# pylint: disable=too-many-locals, protected-access
def test_override_re_optimization(get_column_names):
    # pylint: disable=too-many-statements
    """
    Test that the optimization model is constructed and solved correctly
    with feasible solution when an override choice is made.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        objective_weight_impact=100,
    )

    override_data = ReOptimizationData(
        {13: 0, 19: 1},
        {
            1: {851: 0, 858: 0},
            11: {80: 1},
            6: {600: 1},
            19: {21: 1, 83: 1, 182: 1, 280: 1, 981: 1},
            10: {734: 1, 647: 1},
            40: {601: 1},
        },
        {10: [647]},
        True,
    )

    well_add_existing_cluster = {6: [600], 11: [80], 10: [734], 40: [601], 24: [647]}
    well_add_new_cluster = {11: [80], 6: [600], 10: [734, 647], 40: [601]}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    remove_widget_return = OverrideRemoveLockInfo([], {})
    lock_widget_return = OverrideRemoveLockInfo([], {})
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )
    initial_opt_mdl_inputs = copy.deepcopy(opt_mdl_inputs)

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    assert hasattr(opt_mdl_inputs, "update_cluster")
    assert 13 in opt_campaign.projects

    # Update the model input based on the override selection
    opt_mdl_inputs.update_cluster(override_selections)

    for cluster in opt_mdl_inputs.campaign_candidates:
        if cluster != 10:
            assert set(opt_mdl_inputs.campaign_candidates[cluster]) == set(
                initial_opt_mdl_inputs.campaign_candidates[cluster]
            )
    assert 24 not in opt_mdl_inputs.campaign_candidates
    assert opt_mdl_inputs.owner_well_count == initial_opt_mdl_inputs.owner_well_count

    # Build the new optimization model based on the override selection
    or_opt_mdl = opt_mdl_inputs.build_optimization_model(override_data)
    or_opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    opt_mdl = opt_mdl_inputs.optimization_model

    assert hasattr(or_opt_mdl, "fix_var")

    # Ensure clusters and wells are fixed based on the override selection
    assert or_opt_mdl.cluster[13].select_cluster.is_fixed()
    assert or_opt_mdl.cluster[13].select_cluster.value == pytest.approx(0, rel=1e-4)
    for j in or_opt_mdl.cluster[13].select_well:
        assert not or_opt_mdl.cluster[13].select_well[j].is_fixed()

    assert or_opt_mdl.cluster[19].select_cluster.is_fixed()
    assert or_opt_mdl.cluster[19].select_cluster.value == pytest.approx(1, rel=1e-4)
    for j in override_data.re_optimize_well_dict[19]:
        assert or_opt_mdl.cluster[19].select_well[j].is_fixed()
        assert (
            or_opt_mdl.cluster[19].select_well[j].value
            == override_data.re_optimize_well_dict[19][j]
        )

    assert not or_opt_mdl.cluster[1].select_cluster.is_fixed()

    # Test the re-optimization results
    assert 13 not in or_opt_campaign.projects
    assert 80 in or_opt_campaign.projects[11].well_data.data.index
    assert 600 in or_opt_campaign.projects[6].well_data.data.index
    assert 851 not in or_opt_campaign.projects[1].well_data.data.index
    assert set(or_opt_campaign.clusters_dict[19]) == {21, 280, 182, 83, 981}

    # test override-related items are zeros or not used
    assert opt_mdl.excess_budget.value == pytest.approx(0, rel=1e-4)
    for owner in opt_mdl.excess_owc:
        assert opt_mdl.excess_owc[owner].value == pytest.approx(
            0, rel=1e-4
        )  # pylint: disable=no-member
    excess_campaign_length = opt_mdl_inputs.optimization_model.excess_campaign_length()
    assert excess_campaign_length == pytest.approx(0, rel=1e-4)
    assert opt_mdl.total_constraint_penalty() == pytest.approx(
        3067.7005030700907, rel=1e-4
    )
    # pylint: disable=protected-access
    budget_scaling_factor, violation_scaling_factor, _ = (
        opt_mdl._slack_variable_scaling()
    )
    assert opt_mdl.override_slack_scaling.value == violation_scaling_factor
    assert opt_mdl.excess_budget_scaling.value == 100 * budget_scaling_factor
    assert hasattr(opt_mdl, "set_clusters_reassign")
    assert hasattr(opt_mdl, "num_distant_well_pairs")
    assert hasattr(opt_mdl, "excess_distant_pairs")
    assert hasattr(opt_mdl.cluster[1], "is_distant_pair")
    assert not hasattr(opt_mdl, "min_project_size")
    assert not hasattr(opt_mdl, "max_project_size")
    assert hasattr(opt_mdl, "budget_status")
    assert hasattr(opt_mdl, "excess_budget_constraint")
    assert hasattr(opt_mdl, "unused_budget_constraint")
    assert opt_mdl.cluster[1].set_well_pairs_remove == []
    assert not opt_mdl.cluster[1].min_well_violation.fixed
    assert opt_mdl.cluster[1].min_well_violation.value == pytest.approx(0, rel=1e-4)
    assert not opt_mdl.cluster[1].max_well_violation.fixed
    assert opt_mdl.cluster[1].max_well_violation.value == pytest.approx(0, rel=1e-4)
    assert opt_mdl.budget_status.value == pytest.approx(  # pylint: disable=no-member
        0, rel=1e-4
    )
    assert opt_mdl.unused_budget.value > 0  # pylint: disable=no-member
    assert opt_mdl.excess_budget.value == pytest.approx(  # pylint: disable=no-member
        0, 1e-4
    )
    assert not opt_mdl.cluster[1].min_cost_project_violation.fixed
    assert opt_mdl.cluster[1].min_cost_project_violation.value == pytest.approx(
        0, rel=1e-4
    )
    assert not opt_mdl.cluster[1].max_cost_project_violation.fixed
    assert opt_mdl.cluster[1].max_cost_project_violation.value == pytest.approx(
        0, rel=1e-4
    )


# pylint: disable=too-many-locals
def test_re_cluster(get_column_names):
    """
    Test the re_cluster function to ensure that the optimization model
    inputs are accurately updated based on the override choice.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        objective_weight_impact=100,
    )

    well_add_existing_cluster = {
        6: [600],
        11: [80],
        19: [981],
        40: [601],
    }
    well_add_new_cluster = {11: [80], 6: [600], 10: [981], 40: [601]}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    remove_widget_return = OverrideRemoveLockInfo([], {})
    lock_widget_return = OverrideRemoveLockInfo([], {})
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    assert hasattr(opt_mdl_inputs, "update_cluster")
    assert 13 in opt_campaign.projects

    # Update the model input based on the override selection
    opt_mdl_inputs.update_cluster(override_selections)

    assert 981 not in opt_mdl_inputs.campaign_candidates[19]
    assert 981 in opt_mdl_inputs.campaign_candidates[10]
    assert 981 in opt_mdl_inputs.owner_well_count["Owner 104"]


def test_dictionary_instantiation(get_column_names):
    """
    Test using a dictionary to instantiate the OptModelInputs object
    and avoid re-clustering
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    gas_oil_wells_replica = wd.get_gas_oil_wells
    wd_gas_replica = gas_oil_wells_replica["gas"]
    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()
    wd_gas_replica.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        objective_weight_impact=100,
    )

    assert "Clusters" not in wd_gas_replica

    clustering_dictionary = opt_mdl_inputs.campaign_candidates

    OptModelInputs(
        cluster_mapping=clustering_dictionary,
        well_data=wd_gas_replica,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        objective_weight_impact=100,
    )

    assert "Clusters" in wd_gas_replica


def test_pairwise_efficiency_scaling(get_efficiency_column_names):
    """
    Age and depth range scaling should use the full dataset range even when
    campaign candidates do not contain the min/max wells together.
    """
    eff_metrics, im_metrics, col_names, filename = get_efficiency_column_names

    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "age_range": 50,
            "depth_range": 40,
            "dist_range": 10,
        }
    )

    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )
    wd_gas = wd.get_gas_oil_wells["gas"]
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,
        mobilization_cost={1: 120000},
        threshold_distance=20,
        objective_weight_impact=0,
    )

    opt_mdl_inputs.compute_efficiency_scaling_factors()

    expected_age_range = (
        wd_gas[wd_gas.column_names.age].max() - wd_gas[wd_gas.column_names.age].min()
    )
    expected_depth_range = (
        wd_gas[wd_gas.column_names.depth].max()
        - wd_gas[wd_gas.column_names.depth].min()
    )
    expected_dist_range = opt_mdl_inputs.config.threshold_distance

    assert opt_mdl_inputs.config.max_age_range == pytest.approx(expected_age_range)
    assert opt_mdl_inputs.config.max_depth_range == pytest.approx(expected_depth_range)
    assert opt_mdl_inputs.config.max_dist_range == pytest.approx(expected_dist_range)


def test_override_re_optimization_violation(get_column_names):
    # pylint: disable = too-many-statements
    """
    Test that the optimization model is constructed and solved correctly
    with solution that violate specified constraints.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        objective_weight_impact=100,
        max_wells_in_project=5,
        min_wells_in_project=2,
        max_cost_project=500000,
        min_cost_project=240000,
    )

    override_data = ReOptimizationData(
        {1: 1},
        {
            9: {91: 0, 204: 0},
            11: {125: 1},
            19: {86: 1},
            31: {79: 1},
            13: {56: 1},
            1: {807: 1, 858: 1, 860: 1, 876: 1, 901: 1, 912: 1},
        },
        {31: [79]},
        True,
    )

    well_add_existing_cluster = {11: [125], 19: [86], 3: [79], 13: [56], 1: [807]}
    well_add_new_cluster = {11: [125], 19: [86], 31: [79], 13: [56], 1: [807]}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    remove_widget_return = OverrideRemoveLockInfo([], {})
    lock_widget_return = OverrideRemoveLockInfo([], {})
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )
    initial_opt_mdl_inputs = copy.deepcopy(opt_mdl_inputs)

    opt_mdl_inputs.build_optimization_model()
    opt_mdl_inputs.solve_model(solver="highs")

    assert (
        opt_mdl_inputs.campaign_candidates == initial_opt_mdl_inputs.campaign_candidates
    )
    assert opt_mdl_inputs.owner_well_count == initial_opt_mdl_inputs.owner_well_count
    assert hasattr(opt_mdl_inputs.optimization_model.cluster[1], "min_project_size")
    assert hasattr(opt_mdl_inputs.optimization_model.cluster[1], "max_project_size")
    assert opt_mdl_inputs.optimization_model.cluster[1].min_well_violation.fixed
    assert opt_mdl_inputs.optimization_model.cluster[1].max_well_violation.fixed
    assert opt_mdl_inputs.optimization_model.cluster[1].min_cost_project_violation.fixed
    assert opt_mdl_inputs.optimization_model.cluster[1].max_cost_project_violation.fixed

    # Update the model input based on the override selection
    assert hasattr(opt_mdl_inputs, "update_cluster")
    opt_mdl_inputs.update_cluster(override_selections)

    # Build the new optimization model based on the override selection
    or_opt_mdl = opt_mdl_inputs.build_optimization_model(override_data)
    or_opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    opt_mdl = opt_mdl_inputs.optimization_model

    assert hasattr(or_opt_mdl, "fix_var")

    # Test the re-optimization results
    for project in or_opt_campaign.projects:
        assert 91 not in or_opt_campaign.projects[project].well_data.data.index
        assert 204 not in or_opt_campaign.projects[project].well_data.data.index
    assert 807 in or_opt_campaign.projects[1].well_data.data.index
    assert set(or_opt_campaign.clusters_dict[19]) == {21, 83, 223, 280, 86}

    # test override-related items are zeros or not used
    assert opt_mdl.excess_budget.value == pytest.approx(0, rel=1e-4)
    owner_violation = sum(
        opt_mdl.excess_owc[owner].value for owner in opt_mdl.excess_owc
    )
    assert owner_violation == pytest.approx(1, rel=1 - 4)
    excess_campaign_length = opt_mdl_inputs.optimization_model.excess_campaign_length()
    assert excess_campaign_length == pytest.approx(1, rel=1e-4)

    # pylint: disable=protected-access
    budget_scaling_factor, violation_scaling_factor, _ = (
        opt_mdl._slack_variable_scaling()
    )
    assert opt_mdl.override_slack_scaling.value == violation_scaling_factor
    assert opt_mdl.excess_budget_scaling.value == 100 * budget_scaling_factor
    assert opt_mdl.total_constraint_penalty() == pytest.approx(
        6517.668981600631, rel=1e-4
    )

    assert hasattr(opt_mdl, "set_clusters_reassign")
    assert hasattr(opt_mdl, "num_distant_well_pairs")
    assert hasattr(opt_mdl, "excess_distant_pairs")
    assert hasattr(opt_mdl.cluster[1], "is_distant_pair")
    assert hasattr(opt_mdl.cluster[1], "min_project_size")
    assert hasattr(opt_mdl.cluster[1], "max_project_size")
    assert hasattr(opt_mdl.cluster[1], "set_lower_bound_project_cost")
    assert hasattr(opt_mdl.cluster[1], "set_upper_bound_project_cost")
    assert opt_mdl.cluster[31].set_well_pairs_remove == []
    for cluster in opt_mdl.cluster:
        assert not opt_mdl.cluster[cluster].min_well_violation.fixed
        assert opt_mdl.cluster[cluster].min_well_violation.value == pytest.approx(
            0, rel=1e-2
        )
    assert not opt_mdl.cluster[1].max_well_violation.fixed
    assert opt_mdl.cluster[1].max_well_violation.value == pytest.approx(1, rel=1e-4)
    for cluster in opt_mdl.cluster:
        assert not opt_mdl.cluster[cluster].min_cost_project_violation.fixed
        assert not opt_mdl.cluster[cluster].max_cost_project_violation.fixed
    assert opt_mdl.cluster[31].min_cost_project_violation.value == pytest.approx(
        0.0, rel=1e-4
    )
    assert opt_mdl.cluster[1].max_cost_project_violation.value == pytest.approx(
        0.004, rel=1e-5
    )


def test_override_re_optimization_no_selection(get_column_names):
    # pylint: disable=too-many-statements
    """
    Test that the optimization model is constructed and solved
    to the original solution when an override choice is not made but the
    re-optimization process is executed.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        max_wells_per_owner=1,
        objective_weight_impact=100,
    )

    override_data = ReOptimizationData(
        {},
        {},
        {},
        False,
    )

    well_add_existing_cluster = {}
    well_add_new_cluster = {}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    remove_widget_return = OverrideRemoveLockInfo([], {})
    lock_widget_return = OverrideRemoveLockInfo([], {})
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )
    initial_opt_mdl_inputs = copy.deepcopy(opt_mdl_inputs)

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    assert hasattr(opt_mdl_inputs, "update_cluster")
    assert 13 in opt_campaign.projects

    # Update the model input
    opt_mdl_inputs.update_cluster(override_selections)
    print(f"{opt_mdl_inputs.campaign_candidates}")
    print(f"{initial_opt_mdl_inputs.campaign_candidates}")

    for cluster in opt_mdl_inputs.campaign_candidates:
        assert set(opt_mdl_inputs.campaign_candidates[cluster]) == set(
            initial_opt_mdl_inputs.campaign_candidates[cluster]
        )

    assert opt_mdl_inputs.owner_well_count == initial_opt_mdl_inputs.owner_well_count

    # Build the new optimization model
    or_opt_mdl = opt_mdl_inputs.build_optimization_model(override_data)
    or_opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    opt_mdl = opt_mdl_inputs.optimization_model

    assert hasattr(or_opt_mdl, "fix_var")

    # Ensure clusters and wells are not fixed
    assert not or_opt_mdl.cluster[13].select_cluster.is_fixed()
    assert not or_opt_mdl.cluster[1].select_cluster.is_fixed()

    # Test the re-optimization results
    assert set(or_opt_campaign.projects.keys()) == {1, 11, 13, 19}
    for project in opt_campaign.clusters_dict:
        assert set(opt_campaign.clusters_dict[project]) == set(
            or_opt_campaign.clusters_dict[project]
        )

    # test override-related items are zeros or not used
    assert opt_mdl.excess_budget.value == pytest.approx(0, rel=1e-4)
    for owner in opt_mdl.excess_owc:
        assert opt_mdl.excess_owc[owner].value == pytest.approx(
            0, rel=1e-4
        )  # pylint: disable=no-member
    excess_campaign_length = opt_mdl_inputs.optimization_model.excess_campaign_length()
    assert excess_campaign_length == pytest.approx(0, rel=1e-4)
    assert opt_mdl.total_constraint_penalty == pytest.approx(0, rel=1e-4)
    assert opt_mdl.override_slack_scaling.value == pytest.approx(0, rel=1e-4)
    assert opt_mdl.excess_budget_scaling.value == pytest.approx(0, rel=1e-4)
    assert not hasattr(opt_mdl, "set_clusters_reassign")
    assert not hasattr(opt_mdl, "num_distant_well_pairs")
    assert not hasattr(opt_mdl, "excess_distant_pairs")
    assert hasattr(opt_mdl.cluster[1], "is_distant_pair")
    assert not hasattr(opt_mdl, "min_project_size")
    assert not hasattr(opt_mdl, "max_project_size")
    assert opt_mdl.cluster[1].set_well_pairs_remove == []
    assert not opt_mdl.cluster[1].min_well_violation.fixed
    assert opt_mdl.cluster[1].min_well_violation.value == pytest.approx(0, rel=1e-4)
    assert not opt_mdl.cluster[1].max_well_violation.fixed
    assert opt_mdl.cluster[1].max_well_violation.value == pytest.approx(0, rel=1e-4)


def test_resolve_efficiency_score_mismatch(get_efficiency_column_names):
    """
    Test that the resolve helper fixes the selected decision variables and
    restores the correct efficiency score after a model score mismatch.
    """
    eff_metrics, im_metrics, col_names, filename = get_efficiency_column_names

    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )
    wd_gas = wd.get_gas_oil_wells["gas"]
    wd_gas.compute_priority_scores()
    wd_gas = wd_gas.get_high_priority_wells(num_wells=100)

    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3250000,
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        objective_weight_impact=99,
        max_wells_in_project=5,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(
        solver="highs", mip_gap=5e-2, stream_output=False
    )
    opt_mdl = opt_mdl_inputs.optimization_model

    first_project_id = next(iter(opt_campaign.projects))
    expected_depth_range_score = opt_campaign.projects[
        first_project_id
    ].efficiency_metric_scores["depth_range"]

    selected_cluster_values = {
        cluster_id: int(opt_mdl.cluster[cluster_id].select_cluster.value >= 0.95)
        for cluster_id in opt_mdl.set_clusters
    }
    selected_well_values = {
        cluster_id: {
            well: int(opt_mdl.cluster[cluster_id].select_well[well].value >= 0.95)
            for well in opt_mdl.cluster[cluster_id].set_wells
        }
        for cluster_id in opt_mdl.set_clusters
    }

    opt_mdl.cluster[first_project_id].efficiency_model.depth_range.score.set_value(
        expected_depth_range_score + 1
    )

    resolved_campaign = (
        opt_mdl_inputs._get_optimal_campaign_with_resolve_on_efficiency_mismatch(
            "highs",
            {"solver": "highs", "mip_gap": 5e-2, "stream_output": False},
            resolve_mip_gap=1e-4,
        )
    )

    assert resolved_campaign.clusters_dict == opt_campaign.clusters_dict
    assert resolved_campaign.projects[first_project_id].efficiency_metric_scores[
        "depth_range"
    ] == pytest.approx(expected_depth_range_score, rel=1e-3)
    assert opt_mdl.cluster[
        first_project_id
    ].efficiency_model.depth_range.score.value == (
        pytest.approx(expected_depth_range_score, rel=1e-3)
    )

    for cluster_id in opt_mdl.set_clusters:
        blk = opt_mdl.cluster[cluster_id]
        assert blk.select_cluster.is_fixed()
        assert blk.select_cluster.value == selected_cluster_values[cluster_id]

        for well in blk.set_wells:
            if selected_cluster_values[cluster_id] == 1:
                assert blk.select_well[well].is_fixed()
                assert (
                    blk.select_well[well].value
                    == selected_well_values[cluster_id][well]
                )
            else:
                assert not blk.select_well[well].is_fixed()


def find_key(campaign, well):
    """
    A function that identifies a well's cluster based on its index.
    """
    return next(
        (cluster for cluster, well_list in campaign.items() if well in well_list), None
    )


@pytest.fixture(name="get_mdl_input_exhaustive", scope="function")
def get_mdl_input_exhaustive_fixture(get_column_names):
    """
    Pytest fixture to set up the impact metric, assign
    column names, and read the test data.
    """
    im_metrics, col_names, filename = get_column_names

    eff_metrics = EfficiencyMetrics()

    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 20,
            "num_unique_owners": 30,
            "elevation_delta": 20,
            "age_range": 10,
            "depth_range": 20,
        }
    )

    # Create the well data object
    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 1200, 2: 2100, 3: 2800, 4: 3500}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 840

    # Test the model and options
    wd_gas.compute_priority_scores()
    selected_well = [
        860,
        901,
        864,
        858,
        912,
        829,
        495,
        843,
        67,
        21,
        767,
        903,
        897,
        812,
        833,
        180,
        104,
        817,
        118,
        83,
        851,
        773,
        649,
        716,
    ]
    wd_gas = wd_gas._construct_sub_data(selected_well)

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3.21e9,
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
        cluster_method="Exhaustive",
    )

    return opt_mdl_inputs, eff_metrics


# pylint: disable=too-many-statements,unused-variable
def test_override_re_optimization_exhaustive(get_mdl_input_exhaustive):
    """
    Test that the optimization model is constructed and solved correctly
    when Exhaustive clustering method is used.
    """

    override_data = ReOptimizationData(
        {812: 1, 817: 1, 833: 1},
        {
            21: {83: 0, 67: 1, 495: 1},
            903: {903: 0},
            104: {104: 0},
            180: {180: 0},
            67: {67: 0},
            495: {495: 0},
            773: {773: 0},
            767: {767: 0, 180: 1, 104: 1},
            812: {649: 1, 901: 1, 812: 1},
            716: {767: 1},
            817: {83: 1, 829: 1, 817: 1},
            833: {858: 1, 833: 1},
        },
        {812: [649], 767: [104, 180], 716: [767], 21: [67, 495], 817: [83]},
        True,
    )

    well_add_existing_cluster = {
        649: [649],
        180: [180],
        104: [104],
        767: [767],
        67: [67],
        495: [495],
        21: [83],
    }
    well_add_new_cluster = {
        812: [649],
        767: [180, 104],
        716: [767],
        21: [67, 495],
        817: [83],
    }

    lock_clusters = [817, 812, 833]
    lock_wells = {817: [829, 817], 812: [901, 812], 833: [858, 833]}

    remove_clusters = [104, 180, 67, 495, 773]
    remove_wells = {
        21: [83],
        903: [903],
        104: [104],
        180: [180],
        67: [67],
        495: [495],
        773: [773],
        767: [767],
    }

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(lock_clusters, lock_wells)
    remove_widget_return = OverrideRemoveLockInfo(remove_clusters, remove_wells)
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )

    opt_mdl_inputs, _ = get_mdl_input_exhaustive

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs", mip_gap=0.2)

    # assert is_distant_pair is zero in the original model

    assert hasattr(opt_mdl_inputs, "update_cluster_exhaustive")
    assert 21 in opt_campaign.projects

    # Update the model input based on the override selection
    opt_mdl_inputs.update_cluster(override_selections)

    # Build the new optimization model based on the override selection
    or_opt_mdl = opt_mdl_inputs.build_optimization_model(override_data)
    or_opt_campaign = opt_mdl_inputs.solve_model(solver="highs", mip_gap=0.2)
    opt_mdl = opt_mdl_inputs.optimization_model

    assert hasattr(or_opt_mdl, "fix_var")

    # Ensure the clusters are correctly updated based on the override selection
    assert set(opt_mdl_inputs.campaign_candidates[104]) == {104, 180}
    assert set(opt_mdl_inputs.campaign_candidates[180]) == {104, 180}
    assert set(opt_mdl_inputs.campaign_candidates[67]) == {67, 495}
    assert set(opt_mdl_inputs.campaign_candidates[495]) == {67, 495}
    assert set(opt_mdl_inputs.campaign_candidates[767]) == {767, 104, 180}
    assert set(opt_mdl_inputs.campaign_candidates[716]) == {21, 716, 767, 67, 495}
    assert set(opt_mdl_inputs.campaign_candidates[21]) == {21, 716, 767, 67, 495}
    assert 21 not in opt_mdl_inputs.campaign_candidates[83]
    assert set(opt_mdl_inputs.campaign_candidates[83]) == set(
        opt_mdl_inputs.campaign_candidates[817]
    )

    for cluster, well_list in opt_mdl_inputs.campaign_candidates.items():
        if cluster != 649:
            if not 901 in well_list or not 812 in well_list:
                assert 649 not in well_list
        else:
            assert 649 in well_list

        if 901 in well_list and 812 in well_list:
            assert 649 in well_list

    # Ensure wells are fixed based on the override selection
    assert not or_opt_mdl.cluster[67].select_cluster.is_fixed()
    assert or_opt_mdl.cluster[67].select_cluster.value == pytest.approx(0, rel=1e-4)
    assert or_opt_mdl.cluster[67].select_well[67].is_fixed()
    assert not or_opt_mdl.cluster[67].select_well[495].is_fixed()

    assert not or_opt_mdl.cluster[180].select_cluster.is_fixed()
    assert or_opt_mdl.cluster[180].select_cluster.value == pytest.approx(1, rel=1e-4)
    for j in override_data.re_optimize_well_dict[180]:
        assert not or_opt_mdl.cluster[180].select_well[j].is_fixed()
        assert or_opt_mdl.cluster[180].select_well[j].value == pytest.approx(
            1, rel=1e-4
        )

    assert or_opt_mdl.cluster[767].select_well[767].is_fixed()
    assert not or_opt_mdl.cluster[21].select_well[21].is_fixed()
    for j in override_data.re_optimize_well_dict[716]:
        assert not or_opt_mdl.cluster[716].select_well[j].is_fixed()

    assert not or_opt_mdl.cluster[817].select_cluster.is_fixed()
    for j in override_data.re_optimize_well_dict[817]:
        assert not or_opt_mdl.cluster[817].select_well[j].is_fixed()

    # Test the re-optimization results
    campaign_list = or_opt_campaign.clusters_dict
    assert find_key(campaign_list, 180) == find_key(campaign_list, 104)
    assert find_key(campaign_list, 83) == find_key(campaign_list, 817)
    assert find_key(campaign_list, 829) == find_key(campaign_list, 817)
    assert find_key(campaign_list, 649) == find_key(campaign_list, 812)
    assert find_key(campaign_list, 812) == find_key(campaign_list, 901)

    # test override-related items are zeros or not used
    assert opt_mdl.excess_budget.value == pytest.approx(0, rel=1e-4)
    for owner in opt_mdl.excess_owc:
        assert opt_mdl.excess_owc[owner].value == pytest.approx(
            0, rel=1e-4
        )  # pylint: disable=no-member
    assert opt_mdl.total_constraint_penalty() == pytest.approx(0, rel=1e-4)
    # pylint: disable=protected-access
    budget_scaling_factor, violation_scaling_factor, _ = (
        opt_mdl._slack_variable_scaling()
    )
    assert opt_mdl.override_slack_scaling.value == violation_scaling_factor
    assert opt_mdl.excess_budget_scaling.value == 100 * budget_scaling_factor
    assert hasattr(opt_mdl, "set_clusters_reassign")
    assert hasattr(opt_mdl, "num_distant_well_pairs")
    assert hasattr(opt_mdl, "excess_distant_pairs")
    assert hasattr(opt_mdl.cluster[21], "is_distant_pair")
    assert not hasattr(opt_mdl, "min_project_size")
    assert not hasattr(opt_mdl, "max_project_size")
    assert hasattr(opt_mdl, "budget_status")
    assert hasattr(opt_mdl, "excess_budget_constraint")
    assert hasattr(opt_mdl, "unused_budget_constraint")
    assert opt_mdl.cluster[21].set_well_pairs_remove == {
        (21, 495),
        (716, 67),
        (716, 495),
        (767, 495),
        (67, 495),
        (495, 67),
    }
    assert opt_mdl.budget_status.value == pytest.approx(  # pylint: disable=no-member
        0, rel=1e-4
    )
    assert opt_mdl.unused_budget.value > 0  # pylint: disable=no-member
    assert opt_mdl.excess_budget.value == pytest.approx(  # pylint: disable=no-member
        0, 1e-4
    )


# pylint: disable=too-many-statements,unused-variable
def test_override_re_optimization_anchor_well_handling(get_mdl_input_exhaustive):
    """
    Test that the _check_missing_well function when Exhaustive clustering
    method is used.
    """

    override_data = ReOptimizationData(
        {},
        {
            118: {21: 1},
            21: {21: 0},
            67: {67: 0},
            833: {858: 0},
            180: {858: 1},
        },
        {118: [21], 180: [858]},
        True,
    )

    well_add_existing_cluster = {21: [21], 833: [858]}
    well_add_new_cluster = {118: [21], 180: [858]}

    lock_clusters = []
    lock_wells = {}

    remove_clusters = []
    remove_wells = {21: [21], 67: [67], 833: [858]}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(lock_clusters, lock_wells)
    remove_widget_return = OverrideRemoveLockInfo(remove_clusters, remove_wells)
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )

    opt_mdl_inputs, eff_metrics = get_mdl_input_exhaustive
    opt_mdl_inputs.config.max_age_range = 118
    opt_mdl_inputs.config.max_depth_range = 10591
    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs", mip_gap=0.2)

    # Remove clusters to test the _check_missing_well function
    # If an anchor well - 21 - is reassigned
    del opt_mdl_inputs.campaign_candidates[83]
    assert 83 not in opt_mdl_inputs.campaign_candidates

    # If a non-anchor well - 858 - is reassigned
    del opt_mdl_inputs.campaign_candidates[901]
    assert 901 not in opt_mdl_inputs.campaign_candidates
    for cluster, well_list in opt_mdl_inputs.campaign_candidates.items():
        if cluster != 858:
            for well in well_list:
                if well == 901:
                    opt_mdl_inputs.campaign_candidates[cluster].remove(well)

    campaign = OverrideCampaign(
        override_selections=override_selections,
        opt_inputs=opt_mdl_inputs,
        opt_campaign=opt_campaign.clusters_dict,
        eff_metrics=eff_metrics,
    )
    override_campaign = campaign.recalculate()

    # check if empty cluster is removed
    assert 67 not in override_campaign.clusters_dict

    opt_mdl_inputs.update_cluster(override_selections)
    assert 83 in opt_mdl_inputs.campaign_candidates
    assert set(opt_mdl_inputs.campaign_candidates[83]) == {83, 716}

    assert 901 in opt_mdl_inputs.campaign_candidates
    assert set(opt_mdl_inputs.campaign_candidates[901]) == {
        833,
        843,
        851,
        860,
        864,
        897,
        901,
    }


def test_pairwise_efficiency_scaling_single_well_candidate(
    get_efficiency_column_names,
):
    """Pairwise efficiency scaling supports a single-well candidate."""
    _, im_metrics, col_names, filename = get_efficiency_column_names

    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "age_range": 50,
            "depth_range": 40,
            "dist_range": 10,
        }
    )

    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )
    wd_gas = wd.get_gas_oil_wells["gas"]
    wd_gas.compute_priority_scores()
    wd_gas = wd_gas.get_high_priority_wells(num_wells=3)
    well_list = wd_gas.data.index.to_list()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        cluster_mapping={1: [well_list[0]], 2: well_list[1:]},
        total_budget=3210000,
        mobilization_cost={1: 120000, 2: 210000, 3: 280000},
        threshold_distance=20,
        objective_weight_impact=0,
    )

    opt_mdl_inputs.compute_efficiency_scaling_factors()

    expected_columns = ["dist_range", "age_range", "depth_range"]
    assert opt_mdl_inputs.pairwise_metrics[1].empty
    assert list(opt_mdl_inputs.pairwise_metrics[1].columns) == expected_columns
    assert len(opt_mdl_inputs.pairwise_metrics[2]) == 1

    campaign = Campaign(
        wd=wd_gas,
        clusters_dict=opt_mdl_inputs.campaign_candidates,
        plugging_cost={1: 0, 2: 0},
        opt_model_inputs=opt_mdl_inputs,
    )
    assert campaign.projects[1].efficiency_metric_scores == {
        "age_range": 0,
        "depth_range": 0,
        "dist_range": 0,
    }


# pylint: disable=too-many-statements,unused-variable
def test_override_re_optimization_exhaustive_remove_wells(get_mdl_input_exhaustive):
    """
    Test that the optimization model is constructed and solved correctly
    when Exhaustive clustering method is used.
    """

    well_add_existing_cluster = {
        21: [83],
    }
    well_add_new_cluster = {}

    lock_clusters = []
    lock_wells = {}

    remove_clusters = []
    remove_wells = {
        21: [83],
    }

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(lock_clusters, lock_wells)
    remove_widget_return = OverrideRemoveLockInfo(remove_clusters, remove_wells)
    override_selections = OverrideSelections(
        remove_widget_return=remove_widget_return,
        add_widget_return=add_widget_return,
        lock_widget_return=lock_widget_return,
    )

    opt_mdl_inputs, _ = get_mdl_input_exhaustive

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs", mip_gap=0.2)

    # assert is_distant_pair is zero in the original model

    assert hasattr(opt_mdl_inputs, "update_cluster_exhaustive")
    assert 21 in opt_campaign.projects

    # Update the model input based on the override selection
    opt_mdl_inputs.update_cluster(override_selections)
