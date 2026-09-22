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
import pathlib

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser import EfficiencyMetrics, WellDataColumnNames
from primo.data_parser.default_data import SORTING_METRICS
from primo.data_parser.input_config import ScenarioType
from primo.data_parser.well_data import WellData
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.project_comparison import ProjectComparison, WellProject
from primo.opt_model.result_parser import Campaign, Project

# pylint: disable=unused-import
from primo.opt_model.tests.test_efficiency_model import get_column_names_fixture

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-many-locals, too-many-statements
def test_project_comparison_basic(get_column_names):
    """
    Tests ProjectComparison class
    """
    # Step 0: Build and solve the model
    eff_metrics, im_metrics, col_names, filename = get_column_names

    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )

    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]
    wd_gas.compute_priority_scores()

    mobilization_cost = {i: i * 84000 for i in range(1, len(wd_gas) + 1)}
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=1500000,
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        objective_weight_impact=100,
        max_wells_in_project=20,
    )
    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs", mip_gap=1e-2)
    project_list = list(opt_campaign.projects.values())

    # Check sorting for all allowed metrics

    for metric in SORTING_METRICS:
        project_comp = ProjectComparison(project_list, sorting_metric=metric)
        sorted_projects = project_comp.projects

        values = [getattr(p, metric) for p in sorted_projects]
        assert values == pytest.approx(
            sorted(values, reverse=True)
        ), f"Sorting failed for {metric}"

    # Test get_top_k_projects and get_project test
    project_comp = ProjectComparison(project_list, sorting_metric="total_impact_score")
    sorted_projects = project_comp.projects

    project_comp_top_3 = project_comp.get_top_k_projects(k=3)
    assert len(project_comp_top_3.projects) == 3
    sorted_scores = [p.total_impact_score for p in project_comp_top_3.projects]
    assert sorted_scores == pytest.approx(sorted(sorted_scores, reverse=True))

    first_project = project_list[0]
    assert first_project.project_id == sorted_projects[0].project_id

    fetched = project_comp.get_project(first_project.project_id)
    assert fetched == first_project

    # Change sorting metric and verify new sort order
    project_comp.sorting_metric = "plugging_cost"
    new_sorted = project_comp.projects
    new_values = [p.plugging_cost for p in new_sorted]
    assert new_values == pytest.approx(sorted(new_values, reverse=True))

    # Remove a project and check project IDs before/after
    project_ids_before = [p.project_id for p in project_comp.projects]
    project_comp.remove_project(first_project.project_id)
    project_ids_after = [p.project_id for p in project_comp.projects]

    assert first_project.project_id not in project_ids_after
    assert len(project_ids_before) - 1 == len(project_ids_after)

    # Add the project back
    project_comp.add_project(first_project)
    project_ids_final = [p.project_id for p in project_comp.projects]
    assert first_project.project_id in project_ids_final
    assert sorted(project_ids_final) == sorted(project_ids_before)

    # Add an empty project to test functionality
    empty_project = Project(
        wd=wd_gas,
        index=[],
        plugging_cost=0,
        project_id="empty_001",
    )

    project_comp.add_project(empty_project)
    assert any(p.project_id == "empty_001" for p in project_comp.projects)

    # Should still be able to sort without error
    project_comp.sorting_metric = "total_impact_score"
    sorted_with_empty = project_comp.projects
    ids = [p.project_id for p in sorted_with_empty]
    assert "empty_001" in ids

    # test different impact metrics in project
    # should return error message

    eff_metrics_2 = EfficiencyMetrics()

    # Set weights for the metrics
    eff_metrics_2.set_weight(
        primary_metrics={
            "num_wells": 10,
            "elevation_delta": 20,
            "age_range": 15,  # changed from 10 to 15
            "depth_range": 15,  # changed from 20 to 15
            "dist_range": 20,
            "dist_to_road": 15,
            "population_density": 0,
            "record_completeness": 0,
            "num_unique_owners": 5,
        }
    )

    wd_2 = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics_2,
    )

    gas_oil_wells = wd_2.get_gas_oil_wells
    wd_gas_2 = gas_oil_wells["gas"]
    wd_gas_2.compute_priority_scores()

    project_2 = Project(
        wd=wd_gas_2,
        index=[2, 3, 6, 7, 15],
        plugging_cost=0.42,
        project_id="project_2",
    )

    project_comp.add_project(project_2)
    assert any(p.project_id == "project_2" for p in project_comp.projects)

    with pytest.raises(
        ValueError,
        match=("Projects use different efficiency metric weights."),
    ):
        project_comp.sorting_metric = "total_impact_score"
        project_comp.sort()


@pytest.fixture(name="get_column_names_comparison", scope="function")
def get_column_names_comparison_fixture():
    """
    Pytest fixture to set up the efficiency metric, assign
    column names, and read the test data.
    """

    eff_metrics = EfficiencyMetrics()

    # Set weights for the metrics
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 20,
            "num_unique_owners": 30,
            "elevation_delta": 20,
            "age_range": 10,
            "depth_range": 20,
        }
    )

    # Construct an object to store column names
    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="y",
        longitude="x",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        elevation_delta="Elevation Delta [m]",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        additional_columns={
            "priority_score": "Priority Score",
        },
    )

    current_dir = pathlib.Path(__file__).resolve()
    well_filename = str(
        current_dir.parents[2].joinpath("demo", "Example_Project_Comparison_data.csv")
    )
    project_filename = str(
        current_dir.parents[2].joinpath(
            "demo", "Example_Project_Comparison_project_data.xlsx"
        )
    )

    return eff_metrics, col_names, well_filename, project_filename


def test_project_comparison_only(get_column_names_comparison):
    """Test the scenario where users choose to compare projects
    when both budget and mobilization costs are provided."""

    eff_metrics, col_names, well_filename, project_filename = (
        get_column_names_comparison
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )

    wd = WellData(
        data=well_filename,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )

    well_project = WellProject(
        project_data=project_filename,
        well_data=wd,
        column_names=col_names,
        well_column="Well",
        project_column="Project",
    )

    assert len(well_project.project_dict) == 4
    assert len(well_project.project_map) == 5
    assert len(well_project.project_dict[1]) == 22
    well_index_select = wd.data[
        wd.data[wd.column_names.well_id] == str(49439)
    ].index.item()
    assert well_index_select in well_project.project_dict[1]
    well_index_not_select = wd.data[
        wd.data[wd.column_names.well_id] == str(99999)
    ].index.item()

    assert well_index_not_select in well_project.project_map[0]
    assert well_index_not_select not in well_project.project_dict

    wd.compute_priority_scores()
    assert hasattr(col_names, "priority_score_normalized")

    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    opt_mdl_inputs = OptModelInputs(
        well_data=wd,
        total_budget=3.2e6,
        mobilization_cost=mobilization_cost,
        cluster_mapping=well_project.project_map,
        scenario_type=scenario_type,
    )

    campaign = well_project.campaign_creation(opt_model_inputs=opt_mdl_inputs)

    assert isinstance(campaign, Campaign)
    assert len(campaign) == 4
    assert isinstance(campaign.projects[1], Project)
    project_length = campaign.projects[13].num_wells
    assert project_length == 4
    assert campaign.projects[13].plugging_cost == pytest.approx(
        mobilization_cost[project_length], 1e-4
    )
    assert campaign.projects[13].total_impact_score == pytest.approx(242, 1e-4)
    assert campaign.projects[13].impact_score == pytest.approx(60.5, 1e-4)
    assert campaign.projects[13].efficiency_score == pytest.approx(
        37.247446220294336, 1e-4
    )


def test_project_comparison_only_no_budget_mob_cost(get_column_names_comparison):
    """Test the scenario where users choose to compare projects
    when budget and mobilization costs are not provided."""

    eff_metrics, col_names, well_filename, project_filename = (
        get_column_names_comparison
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )

    wd = WellData(
        data=well_filename,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )

    well_project = WellProject(
        project_data=project_filename,
        well_data=wd,
        column_names=col_names,
        well_column="Well",
        project_column="Project",
    )

    assert not hasattr(col_names, "priority_score_normalized")
    wd.compute_priority_scores()
    assert hasattr(col_names, "priority_score_normalized")

    opt_mdl_inputs = OptModelInputs(
        well_data=wd,
        cluster_mapping=well_project.project_map,
        scenario_type=scenario_type,
    )

    campaign = well_project.campaign_creation(opt_model_inputs=opt_mdl_inputs)

    assert isinstance(campaign, Campaign)
    assert len(campaign) == 4
    assert isinstance(campaign.projects[1], Project)
    assert campaign.projects[13].plugging_cost == "N/A"
    assert campaign.projects[13].total_impact_score == pytest.approx(242, 1e-4)
    assert campaign.projects[13].impact_score == pytest.approx(60.5, 1e-4)
    assert campaign.projects[13].efficiency_score == pytest.approx(
        37.247446220294336, 1e-4
    )
    project_sort = ProjectComparison(
        projects=list(campaign.projects.values()),
        sorting_metric="total_impact_score",
    )
    project_str = (
        "Total Wells: 38Total Plugging Cost: not available"
        "Project IDs (Sorted): [1, 11, 13, 19]"
    )
    assert str(project_sort) == project_str


def test_project_comparison_only_data_issue(get_column_names_comparison):
    """Test the scenario where users choose to compare projects
    when both budget and mobilization costs are provided."""

    eff_metrics, col_names, well_filename, _ = get_column_names_comparison

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )

    wd = WellData(
        data=well_filename,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )

    current_dir = pathlib.Path(__file__).resolve().parent
    project_filename = str(current_dir / "project_data_missing.csv")

    with pytest.raises(
        ValueError,
        match=(
            r"The following wells do not have an associated project number: "
            r"\['82654', '84951', '47058'\]\.\n"
            r"The following projects have missing well information: \['1', '13'\]\.\n"
            r"The following wells are assigned to multiple projects: "
            r"\['91176', '47058', '23065', '48446', '58876', '53255'\]\.\n"
            r"The following wells have non-integer project number: "
            r"\['84538', '73758'\]\.\n"
            r"Please check your input file\."
        ),
    ):
        WellProject(
            project_data=project_filename,
            well_data=wd,
            column_names=col_names,
            well_column="Well",
            project_column="Project",
        )


def test_project_comparison_only_data_consistency(get_column_names_comparison):
    """Test the scenario where users choose to compare projects
    when both budget and mobilization costs are provided."""

    eff_metrics, col_names, well_filename, _ = get_column_names_comparison

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )

    wd = WellData(
        data=well_filename,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )

    current_dir = pathlib.Path(__file__).resolve().parent
    project_filename = str(current_dir / "project_data_unexpected.xlsx")

    with pytest.raises(
        ValueError,
        match=(
            r"The following wells are not in the well data input: \['100001'\]\."
            r"Please check your input file\."
        ),
    ):
        WellProject(
            project_data=project_filename,
            well_data=wd,
            column_names=col_names,
            well_column="Well",
            project_column="Project",
        )


def test_project_comparison_only_wrong_file_type(get_column_names_comparison):
    """Test the scenario where users choose to compare projects
    when both budget and mobilization costs are provided."""

    eff_metrics, col_names, well_filename, _ = get_column_names_comparison

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )

    wd = WellData(
        data=well_filename,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )

    current_dir = pathlib.Path(__file__).resolve().parent
    project_filename = str(current_dir / "project_data.txt")

    with pytest.raises(
        TypeError,
        match=(
            r"Unsupported input file format. Only .xlsx, .xls, and .csv are supported."
        ),
    ):
        WellProject(
            project_data=project_filename,
            well_data=wd,
            column_names=col_names,
            well_column="Well",
            project_column="Project",
        )


def test_project_comparison_only_cost_of_plugging(get_column_names_comparison):
    """Test the scenario where users choose to compare projects
    when users provide the cost of plugging information."""

    eff_metrics, col_names, well_filename, project_filename = (
        get_column_names_comparison
    )

    # Add cost of plugging information to the well data
    wd_df = pd.read_csv(well_filename, dtype={col_names.well_id: str})
    cost_values = [200000, 240222, 280444, 320666, 360888]
    wd_df["Cost of Plugging [USD]"] = np.resize(cost_values, len(wd_df))

    # Add cost_of_plugging to column name
    col_names.cost_of_plugging = "Cost of Plugging [USD]"

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )

    wd = WellData(
        data=wd_df,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )

    well_project = WellProject(
        project_data=project_filename,
        well_data=wd,
        column_names=col_names,
        well_column="Well",
        project_column="Project",
    )

    opt_mdl_inputs = OptModelInputs(
        well_data=wd,
        total_budget=3.2e6,
        cluster_mapping=well_project.project_map,
        scenario_type=scenario_type,
        discount_cost_factor=1,
    )

    campaign = well_project.campaign_creation(opt_model_inputs=opt_mdl_inputs)

    assert isinstance(campaign, Campaign)
    assert len(campaign) == 4
    assert isinstance(campaign.projects[1], Project)
    project_length = campaign.projects[13].num_wells
    assert project_length == 4
    project_wells = well_project.project_dict[13]
    project_well_index = wd.data.loc[project_wells]
    beta = opt_mdl_inputs.config.discount_cost_factor
    cost_scale = (project_length**beta) / project_length
    plugging_cost = (
        project_well_index[wd.col_names.cost_of_plugging].sum()
    ) * cost_scale

    assert campaign.projects[13].plugging_cost == pytest.approx(plugging_cost, 1e-4)


def test_project_map_omits_empty_unselected_project(get_column_names_comparison):
    """Project map should not add project 0 when every well is selected."""
    eff_metrics, col_names, well_filename, _ = get_column_names_comparison

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=False, project_comparison=True
    )
    well_df = pd.read_csv(well_filename, dtype={col_names.well_id: str}).head(10)
    wd = WellData(
        data=well_df,
        column_names=col_names,
        impact_metrics=None,
        efficiency_metrics=eff_metrics,
        scenario_type=scenario_type,
    )
    project_data = pd.DataFrame(
        {
            "Project": range(1, len(wd) + 1),
            "Well": wd.data[col_names.well_id].to_list(),
        }
    )

    well_project = WellProject(
        project_data=project_data,
        well_data=wd,
        column_names=col_names,
        well_column="Well",
        project_column="Project",
    )

    assert 0 not in well_project.project_map
    assert all(len(wells) == 1 for wells in well_project.project_map.values())
