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
import pytest

# User-defined libs
from primo.data_parser import EfficiencyMetrics
from primo.data_parser.well_data import WellData
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.result_parser import Campaign
from primo.opt_model.tests.test_model_options import (  # pylint: disable=unused-import
    get_column_names_fixture,
)
from primo.utils.config_utils import (
    OverrideAddInfo,
    OverrideRemoveLockInfo,
    OverrideSelections,
)
from primo.utils.override_utils import AssessFeasibility, OverrideCampaign
from primo.utils.tests.test_config_utils import (  # pylint: disable=unused-import
    efficiency_metrics_fixture,
    get_model_fixture,
)


@pytest.fixture(name="get_model_no_max_owc")
def get_model_fixture_no_max_owc(get_column_names, eff_metric):
    """
    Pytest fixture for constructing an optimization model and obtain
    the optimization results.
    """
    # pylint: disable=duplicate-code
    im_metrics, col_names, filename = get_column_names
    eff_metrics = eff_metric

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
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    wd_gas.compute_priority_scores()

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=20,
        objective_weight_impact=100,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    return opt_campaign, opt_mdl_inputs, eff_metrics


@pytest.fixture(name="or_infeasible_selection")
def or_infeasible_selection_fixture():
    """
    Pytest fixture for constructing an override selection return which
    will lead to infeasible P&A projects.
    """

    project_remove = [13]
    well_remove = {1: [851, 858]}
    well_add_existing_cluster = {
        1: [851],
        6: [600],
        11: [80],
        10: [734],
        40: [601],
    }
    well_add_new_cluster = {11: [851, 80], 6: [600], 10: [734], 40: [601]}
    project_lock = [19]
    well_lock = {19: [21, 83, 182, 280, 981]}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_override_campaign(or_infeasible_selection, get_model):
    """
    Test the override campaign class when the new project is infeasible
    after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_infeasible_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert isinstance(or_camp_class.remove, OverrideRemoveLockInfo)
    assert isinstance(or_camp_class.add, OverrideAddInfo)
    assert isinstance(or_camp_class.lock, OverrideRemoveLockInfo)
    assert isinstance(or_camp_class.opt_inputs, OptModelInputs)
    assert or_camp_class.eff_metrics == eff_metrics
    assert isinstance(or_camp_class.eff_metrics, EfficiencyMetrics)
    assert hasattr(or_camp_class.remove, "cluster")
    assert hasattr(or_camp_class.remove, "well")

    # Update the optimal campaign based on the override selection
    assert 13 not in or_camp_class.new_campaign
    assert 851 not in or_camp_class.new_campaign[1]
    assert 6 in or_camp_class.new_campaign
    assert or_camp_class.new_campaign[6] == [600]
    assert or_camp_class.new_campaign[19] == [21, 83, 182, 280, 981]

    assert isinstance(or_camp_class.feasibility.wd, WellData)
    assert 210 not in or_camp_class.feasibility.wd.data.index
    assert len(or_camp_class.feasibility.plug_list) == 37

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() > 0
    assert or_camp_class.feasibility.assess_owner_well_count()
    assert or_camp_class.feasibility.assess_distances()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 4
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], str)
    assert isinstance(violation_info_dict[key_list[2]], pd.DataFrame)
    assert isinstance(violation_info_dict[key_list[3]], pd.DataFrame)

    override_campaign = or_camp_class.recalculate()
    assert isinstance(override_campaign, Campaign)
    assert hasattr(override_campaign, "set_efficiency_weights")
    assert hasattr(override_campaign, "compute_efficiency_scores")
    project = override_campaign.projects[1]
    assert np.isclose(project.impact_score, 68.88, rtol=1e-2, atol=1e-2)
    assert np.isclose(project.efficiency_score, 20.802, rtol=1e-2, atol=1e-2)
    assert len(override_campaign.wd) == len(opt_mdl_inputs.config.well_data)

    override_campaign_dict = or_camp_class.recalculate_scores()
    assert isinstance(override_campaign_dict, dict)
    assert 13 not in override_campaign_dict
    assert 63.55375464985779 == pytest.approx(override_campaign_dict[11][0], rel=1e-2)
    assert 19.69807578343402 == pytest.approx(override_campaign_dict[11][1], rel=1e-2)

    override_info = or_camp_class.re_optimize_data()
    assert override_info.re_optimize_cluster_dict == {19: 1}
    assert override_info.re_optimize_well_dict == {
        1: {851: 0, 858: 0},
        11: {851: 1, 80: 1},
        6: {600: 1},
        19: {21: 1, 83: 1, 182: 1, 280: 1, 981: 1},
        10: {734: 1},
        40: {601: 1},
    }
    assert override_info.reassign == {11: [851]}
    assert override_info.override_status


@pytest.fixture(name="or_feasible_selection")
def or_feasible_selection_fixture():
    """
    Pytest fixture for constructing an override selection return which
    will lead to feasible P&A projects.
    """
    project_remove = [19]
    well_remove = {}
    well_add_existing_cluster = {}
    well_add_new_cluster = {}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_feasible_override_campaign(or_feasible_selection, get_model):
    """
    Test the override campaign class when the new project is feasible
    after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_feasible_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert not or_camp_class.feasibility.assess_max_cost_project()
    assert not or_camp_class.feasibility.assess_min_cost_project()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 1
    assert violation_info_dict["Project Status:"] == "FEASIBLE"


def test_override_campaign_no_owc(or_feasible_selection, get_model_no_max_owc):
    """
    Test to make sure that when a user leaves
    max_owner_well_count ==None that we skip that
    override check by setting the return to an empty Dict.
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model_no_max_owc
    or_selection = or_feasible_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert not or_camp_class.feasibility.assess_owner_well_count()


@pytest.fixture(name="or_infeasible_owc_selection")
def or_infeasible_owc_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the owner well count constraint.
    """
    project_remove = [13]
    well_remove = {}
    well_add_existing_cluster = {19: [86]}
    well_add_new_cluster = {19: [86]}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_owc(or_infeasible_owc_selection, get_model):
    """
    Test the override campaign class where the new projects violate
    the owner well count constraint after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_infeasible_owc_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)
    violated_operators = violation_info_dict[key_list[1]]
    assert violated_operators["Owner"].loc[violated_operators.index[0]] == "Owner 7"
    assert violated_operators["Number of wells"].loc[violated_operators.index[0]] == 2
    assert violated_operators["Wells"].loc[violated_operators.index[0]] == [
        "33912",
        "69687",
    ]


@pytest.fixture(name="or_feasible_distance_selection")
def or_infeasible_distance_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the distance constraint.
    """
    project_remove = [13]
    well_remove = {1: [851]}
    well_add_existing_cluster = {1: [851, 807]}
    well_add_new_cluster = {11: [851], 36: [807]}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_distance(or_feasible_distance_selection, get_model):
    """
    Test the override campaign class where (1) the new projects violate
    the distance constraint after the override step (2) A well is not removed
    from the recommended projects before being reassigned to another project
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_feasible_distance_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    # Test if there is any duplication in the plug_list
    assert 807 in or_camp_class.feasibility.plug_list
    assert or_camp_class.feasibility.plug_list.count(807) == 1

    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert or_camp_class.feasibility.assess_distances()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)
    violated_operators = violation_info_dict[key_list[1]].head(1)
    assert violated_operators["Project"].loc[violated_operators.index[0]] == 11
    assert violated_operators["Well 1"].loc[violated_operators.index[0]] == "62199"
    assert violated_operators["Well 2"].loc[violated_operators.index[0]] == "58876"
    assert np.isclose(
        violated_operators["Distance between Well 1 and 2 [Miles]"].loc[
            violated_operators.index[0]
        ],
        98.30642202632235,
    )


@pytest.fixture(name="or_non_selection")
def or_non_selection_fixture():
    """
    Pytest fixture for constructing an override selection return where
    no change is made.
    """
    project_remove = []
    well_remove = {}
    well_add_existing_cluster = {}
    well_add_new_cluster = {}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_non_override(or_non_selection, get_model):
    """
    Test the scenario where no override selection is made
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_non_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    # Test if the plug_list is empty
    assert or_camp_class.feasibility.plug_list[0] == 897

    override_info = or_camp_class.re_optimize_data()
    assert not override_info.override_status


@pytest.fixture(name="get_model_max_num_well_in_project")
def get_model_fixture_max_num_well_in_project(get_column_names, eff_metric):
    """
    Pytest fixture for constructing an optimization model and obtain
    the optimization results where there are constraint on the max
    and min number of wells in the project but no distance constraint.
    """
    # pylint: disable=duplicate-code
    im_metrics, col_names, filename = get_column_names
    eff_metrics = eff_metric
    min_wells_in_project = 2
    max_wells_in_project = 5

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
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    wd_gas.compute_priority_scores()

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
        threshold_distance=20,
        min_wells_in_project=min_wells_in_project,
        max_wells_in_project=max_wells_in_project,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    return opt_campaign, opt_mdl_inputs, eff_metrics


@pytest.fixture(name="or_infeasible_max_num_well_in_project_selection")
def or_infeasible_max_num_well_in_project_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the maximum number of well
    in a project constraint.
    """
    project_remove = []
    well_remove = {19: [21]}
    well_add_existing_cluster = {13: [214]}
    well_add_new_cluster = {13: [214]}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_max_num_well_in_project(
    or_infeasible_max_num_well_in_project_selection, get_model_max_num_well_in_project
):
    """
    Test the override campaign class where the new projects violate
    the maximum number of wells in the project constraint after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model_max_num_well_in_project
    or_selection = or_infeasible_max_num_well_in_project_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()

    # Test that assess_distances() returns an empty dictionary when no
    # distance threshold is defined

    assert not or_camp_class.feasibility.assess_distances()
    assert not or_camp_class.feasibility.assess_min_num_well_in_project()
    assert or_camp_class.feasibility.assess_max_num_well_in_project()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)
    violated_max_num_wells_in_project = violation_info_dict[key_list[1]].head(1)
    assert (
        violated_max_num_wells_in_project["Project"].loc[
            violated_max_num_wells_in_project.index[0]
        ]
        == 13
    )
    assert (
        violated_max_num_wells_in_project["# of Wells"].loc[
            violated_max_num_wells_in_project.index[0]
        ]
        == 6
    )


@pytest.fixture(name="get_model_min_num_well_in_project")
def get_model_fixture_min_num_well_in_project(get_column_names, eff_metric):
    """
    Pytest fixture for constructing an optimization model and obtain
    the optimization results where there are constraint on the max
    and min number of wells in the project but no distance constraint.
    """
    # pylint: disable=duplicate-code
    im_metrics, col_names, filename = get_column_names
    eff_metrics = eff_metric
    min_wells_in_project = 5
    max_wells_in_project = 5

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
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    wd_gas.compute_priority_scores()

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
        threshold_distance=20,
        min_wells_in_project=min_wells_in_project,
        max_wells_in_project=max_wells_in_project,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    return opt_campaign, opt_mdl_inputs, eff_metrics


@pytest.fixture(name="or_infeasible_min_num_well_in_project_selection")
def or_infeasible_min_num_well_in_project_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the minimum number of wells
    in a project constraint.
    """
    project_remove = []
    well_remove = {
        1: [876],
    }
    well_add_existing_cluster = {}
    well_add_new_cluster = {}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_min_num_well_in_project(
    or_infeasible_min_num_well_in_project_selection, get_model_min_num_well_in_project
):
    """
    Test the override campaign class where the new projects violate
    the minimum number of wells in the project constraint after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model_min_num_well_in_project
    or_selection = or_infeasible_min_num_well_in_project_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_min_num_well_in_project()
    assert not or_camp_class.feasibility.assess_max_num_well_in_project()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)

    violated_min_num_wells_in_project = violation_info_dict[key_list[1]].head(1)
    assert (
        violated_min_num_wells_in_project["Project"].loc[
            violated_min_num_wells_in_project.index[0]
        ]
        == 1
    )
    assert (
        violated_min_num_wells_in_project["# of Wells"].loc[
            violated_min_num_wells_in_project.index[0]
        ]
        == 4
    )


@pytest.fixture(name="get_model_cost_project")
def get_model_fixture_cost_project(get_column_names, eff_metric):
    """
    Pytest fixture for constructing an optimization model and obtain
    the optimization results where there are constraint on the max
    and min budget per project but no distance constraint.
    """
    # pylint: disable=duplicate-code
    im_metrics, col_names, filename = get_column_names
    eff_metrics = eff_metric
    max_cost_project = 500000
    min_cost_project = 240000

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
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    wd_gas.compute_priority_scores()

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        objective_weight_impact=100,
        threshold_distance=20,
        max_cost_project=max_cost_project,
        min_cost_project=min_cost_project,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    return opt_campaign, opt_mdl_inputs, eff_metrics


@pytest.fixture(name="or_infeasible_max_cost_project_selection")
def or_infeasible_max_cost_project_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the maximum budget
    per project constraint.
    """
    project_remove = []
    well_remove = {1: [876]}
    well_add_existing_cluster = {11: [419]}
    well_add_new_cluster = {11: [419]}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_max_cost_project(
    or_infeasible_max_cost_project_selection, get_model_cost_project
):
    """
    Test the override campaign class when the new project is infeasible
    after the override step which violates the maximum budget per project
    constraint
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model_cost_project
    or_selection = or_infeasible_max_cost_project_selection

    or_camp_class = OverrideCampaign(
        or_selection,
        opt_mdl_inputs,
        opt_campaign.clusters_dict,
        eff_metrics,
    )

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_max_cost_project()
    assert not or_camp_class.feasibility.assess_min_cost_project()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)

    violated_max_cost_project = violation_info_dict[key_list[1]].head(1)
    assert (
        violated_max_cost_project["Project"].loc[violated_max_cost_project.index[0]]
        == 11
    )
    assert (
        pytest.approx(
            violated_max_cost_project["Cost"].loc[violated_max_cost_project.index[0]],
            1e-3,
        )
        == 504000.0
    )


@pytest.fixture(name="or_infeasible_min_cost_project_selection")
def or_infeasible_min_cost_project_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the minimum budget
    per project constraint.
    """
    project_remove = []
    well_remove = {9: [204, 91]}
    well_add_existing_cluster = {}
    well_add_new_cluster = {}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_min_cost_project(
    or_infeasible_min_cost_project_selection, get_model_cost_project
):
    """
    Test the override campaign class when the new project is infeasible
    after the override step which violates the minimum budget per project
    constraint
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model_cost_project
    or_selection = or_infeasible_min_cost_project_selection

    or_camp_class = OverrideCampaign(
        or_selection,
        opt_mdl_inputs,
        opt_campaign.clusters_dict,
        eff_metrics,
    )

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert not or_camp_class.feasibility.assess_max_cost_project()
    assert or_camp_class.feasibility.assess_min_cost_project()

    violation_info_dict = or_camp_class.violation_info
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "CONSTRAINT(S) VIOLATED"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)

    violated_min_cost_project = violation_info_dict[key_list[1]].head(1)
    assert (
        violated_min_cost_project["Project"].loc[violated_min_cost_project.index[0]]
        == 9
    )
    assert (
        pytest.approx(
            violated_min_cost_project["Cost"].loc[violated_min_cost_project.index[0]],
            1e-3,
        )
        == 120000.0
    )
