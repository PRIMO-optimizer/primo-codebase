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
# pylint: disable=too-many-lines
# Standard libs
import logging

# Installed libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser import WellDataColumnNames
from primo.data_parser.metric_data import EfficiencyMetrics, ImpactMetrics
from primo.data_parser.well_data import WellData
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.result_parser import Campaign, export_data_to_excel
from primo.utils.override_utils import find_nearby_efficient_project_id
from primo.utils.raise_exception import MissingDataError

MOBILIZATION_COST = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
for n_wells in range(5, 10 + 1):
    MOBILIZATION_COST[n_wells] = n_wells * 84000

TOTAL_BUDGET = 1e6

# pylint: disable=missing-function-docstring


def _build_campaign(wd: WellData):
    """Helper function to avoid code duplication"""
    return Campaign(
        wd,
        {2: [0, 1], 3: [2, 3], 4: [4, 5]},
        {2: 10, 3: 15, 4: 20},
        OptModelInputs(
            mobilization_cost=MOBILIZATION_COST,
            total_budget=TOTAL_BUDGET,
            well_data=wd,
            objective_weight_impact=100,
        ),
    )


def _build_campaign_with_upc(wd: WellData):
    """Helper function to build a campaign with user plugging cost"""
    return Campaign(
        wd,
        {2: [0, 1], 3: [2, 3], 4: [4, 5]},
        {2: 10, 3: 15, 4: 20},
        OptModelInputs(
            mobilization_cost=None,
            total_budget=TOTAL_BUDGET,
            well_data=wd,
            objective_weight_impact=100,
            discount_cost_factor=1,
        ),
    )


@pytest.fixture(name="get_eff_metrics", scope="function")
def get_eff_metrics_fixture():
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

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


@pytest.fixture(name="get_eff_metrics_simple", scope="function")
def get_eff_metrics_fixture_simple():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 20,
            "num_unique_owners": 30,
            "elevation_delta": 0,
            "age_range": 30,
            "depth_range": 20,
        }
    )

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


@pytest.fixture(name="get_impact_metrics", scope="function")
def get_impact_metrics_fixture():
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

    im_metrics.check_validity()
    im_metrics.delete_metric("other_losses")
    im_metrics.delete_metric("five_year_production_volume")
    im_metrics.delete_metric("well_integrity")
    im_metrics.delete_metric("environment")

    # Submetrics can also be deleted in a similar manner
    im_metrics.delete_submetric("buildings_near")
    im_metrics.delete_submetric("buildings_far")
    return im_metrics


@pytest.fixture(name="get_impact_metrics_custom_basis", scope="function")
def get_impact_metrics_custom_basis_fixture():
    im_metrics = ImpactMetrics(basis=200)

    # Specify weights
    im_metrics.set_weight(
        primary_metrics={
            "well_history": 70,
            "sensitive_receptors": 40,
            "ann_production_volume": 40,
            "well_age": 30,
            "well_count": 20,
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

    im_metrics.check_validity()
    im_metrics.delete_metric("other_losses")
    im_metrics.delete_metric("five_year_production_volume")
    im_metrics.delete_metric("well_integrity")
    im_metrics.delete_metric("environment")

    # Submetrics can also be deleted in a similar manner
    im_metrics.delete_submetric("buildings_near")
    im_metrics.delete_submetric("buildings_far")
    return im_metrics


@pytest.fixture(name="get_impact_metrics_custom_basis_less_than_100", scope="function")
def get_impact_metrics_custom_basis_less_than_100_fixture():
    im_metrics = ImpactMetrics(basis=50)

    # Specify weights
    im_metrics.set_weight(
        primary_metrics={
            "well_history": 10,
            "sensitive_receptors": 10,
            "ann_production_volume": 10,
            "well_age": 10,
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

    im_metrics.check_validity()
    im_metrics.delete_metric("other_losses")
    im_metrics.delete_metric("five_year_production_volume")
    im_metrics.delete_metric("well_integrity")
    im_metrics.delete_metric("environment")

    # Submetrics can also be deleted in a similar manner
    im_metrics.delete_submetric("buildings_near")
    im_metrics.delete_submetric("buildings_far")
    return im_metrics


@pytest.fixture(name="get_well_data_cols", scope="function")
def get_well_data_cols_fixture():
    return WellDataColumnNames(
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


@pytest.fixture(name="get_well_data_cols_with_plugging_cost", scope="function")
def get_well_data_cols_with_plugging_cost_fixture(get_well_data_cols):
    get_well_data_cols.cost_of_plugging = "Plugging Cost"
    return get_well_data_cols


@pytest.fixture(name="get_data", scope="function")
def get_data_fixture():
    return {
        "API Well Number": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Leak [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Violation [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Incident [Yes/No]": {0: "Yes", 1: "Yes", 2: "No", 3: "No", 4: "Yes", 5: "Yes"},
        "Compliance [Yes/No]": {0: "No", 1: "Yes", 2: "No", 3: "Yes", 4: "No", 5: "No"},
        "Oil [bbl/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Gas [Mcf/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Age [Years]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Depth [ft]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Elevation Delta [m]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Distance to Road [miles]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Operator Name": {
            0: "Owner 56",
            1: "Owner 136",
            2: "Owner 137",
            3: "Owner 190",
            4: "Owner 196",
            5: "Owner 196",
        },
        "x": {0: 0.99982, 1: 0.99995, 2: 1.51754, 3: 1.51776, 4: 1.51964, 5: 1.51931},
        "y": {0: 1.95117, 1: 1.9572, 2: 1.9584, 3: 1.95746, 4: 1.95678, 5: 1.95674},
        "Number of Nearby Hospitals": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
        "Number of Nearby Schools": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
    }


@pytest.fixture(name="get_data_with_plugging_cost", scope="function")
def get_data_with_plugging_cost_fixture(get_data):
    get_data["Plugging Cost"] = {i: 10000 for i in range(6)}
    return get_data


# pylint: disable=duplicate-code
@pytest.fixture(name="get_well_data", scope="function")
def get_well_data_fixture(
    get_eff_metrics, get_impact_metrics, get_well_data_cols, get_data
):
    im_metrics = get_impact_metrics
    col_names = get_well_data_cols
    data = get_data

    return WellData(
        pd.DataFrame(data),
        col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=get_eff_metrics,
    )


@pytest.fixture(name="get_well_data_with_plugging_cost", scope="function")
def get_well_data_with_plugging_cost_fixture(
    get_eff_metrics,
    get_impact_metrics,
    get_well_data_cols_with_plugging_cost,
    get_data_with_plugging_cost,
):
    return WellData(
        pd.DataFrame(get_data_with_plugging_cost),
        get_well_data_cols_with_plugging_cost,
        impact_metrics=get_impact_metrics,
        efficiency_metrics=get_eff_metrics,
    )


# pylint: disable=duplicate-code
@pytest.fixture(name="get_well_data_custom_basis", scope="function")
def get_well_data_custom_basis_fixture(
    get_eff_metrics, get_impact_metrics_custom_basis, get_well_data_cols, get_data
):
    im_metrics = get_impact_metrics_custom_basis
    col_names = get_well_data_cols
    data = get_data

    return WellData(
        pd.DataFrame(data),
        col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=get_eff_metrics,
    )


@pytest.fixture(name="get_well_data_custom_basis_less_than_100", scope="function")
def get_well_data_custom_basis_less_than_100fixture(
    get_eff_metrics,
    get_impact_metrics_custom_basis_less_than_100,
    get_well_data_cols,
    get_data,
):
    im_metrics = get_impact_metrics_custom_basis_less_than_100
    col_names = get_well_data_cols
    data = get_data

    return WellData(
        pd.DataFrame(data),
        col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=get_eff_metrics,
    )


@pytest.fixture(name="get_campaign", scope="function")
def get_campaign_fixture(get_well_data):
    well_data = get_well_data
    well_data.compute_priority_scores()

    return _build_campaign(well_data)


@pytest.fixture(name="get_campaign_with_plugging_cost", scope="function")
def get_campaign_with_plugging_cost_fixture(get_well_data_with_plugging_cost):
    well_data = get_well_data_with_plugging_cost
    well_data.compute_priority_scores()

    return _build_campaign_with_upc(well_data)


@pytest.fixture(name="get_campaign_custom_basis_greater_than_100", scope="function")
def get_campaign_custom_basis_greater_than_100_fixture(get_well_data_custom_basis):
    well_data = get_well_data_custom_basis
    well_data.add_new_column_ordered(
        "priority_score",
        "Priority Score [0-200]",
        [0, 50, 150, 200, 40, 100],
    )
    well_data.add_new_column_ordered(
        "priority_score_normalized",
        "Priority Score [0-100]",
        [0, 25, 75, 100, 20, 50],
    )

    return _build_campaign(well_data)


@pytest.fixture(name="get_campaign_custom_basis_less_than_100", scope="function")
def get_campaign_custom_basis_less_than_100_fixture(
    get_well_data_custom_basis_less_than_100,
):
    well_data = get_well_data_custom_basis_less_than_100
    well_data.add_new_column_ordered(
        "priority_score",
        "Priority Score [0-50]",
        [0, 50, 25, 12, 7, 29],
    )
    well_data.add_new_column_ordered(
        "priority_score_normalized",
        "Priority Score [0-100]",
        [0, 100, 50, 24, 14, 58],
    )
    return _build_campaign(well_data)


@pytest.fixture(
    name="get_campaign_user_priority_scores_greater_than_100", scope="function"
)
def get_campaign_user_priority_scores_greater_than_100_fixture(get_well_data):
    well_data = get_well_data
    well_data.add_new_column_ordered(
        "priority_score",
        "User supplied priority scores",
        [0, 50, 150, 200, 40, 150],
    )
    well_data.add_new_column_ordered(
        "priority_score_normalized",
        "Priority Score [0-100]",
        [0, 25, 75, 100, 20, 75],
    )

    return _build_campaign(well_data)


@pytest.fixture(
    name="get_campaign_user_priority_scores_less_than_100", scope="function"
)
def get_campaign_user_priority_scores_less_than_100_fixture(get_well_data):
    well_data = get_well_data
    well_data.add_new_column_ordered(
        "priority_score",
        "User supplied priority scores",
        [0, 50, 41, 20, 40, 18],
    )
    well_data.add_new_column_ordered(
        "priority_score_normalized",
        "Priority Score [0-100]",
        [0, 100, 82, 40, 80, 36],
    )
    return _build_campaign(well_data)


@pytest.fixture(name="get_campaign_ranking_scores_equal_to_100", scope="function")
def get_campaign_ranking_scores_equal_to_100_fixture(get_well_data):
    well_data = get_well_data
    well_data.add_new_column_ordered(
        "priority_score",
        "Priority Score [0-100]",
        [0, 50, 41, 20, 40, 18],
    )
    return _build_campaign(well_data)


@pytest.fixture(
    name="get_campaign_custom_basis_greater_than_100_scores_less_than_100",
    scope="function",
)
def get_campaign_custom_basis_greater_than_100_scores_less_than_100fixture(
    get_well_data_custom_basis,
):
    well_data = get_well_data_custom_basis
    well_data.add_new_column_ordered(
        "priority_score",
        "Priority Score [0-200]",
        [0, 50, 50, 50, 40, 50],
    )
    well_data.add_new_column_ordered(
        "priority_score_normalized",
        "Priority Score [0-100]",
        [0, 25, 25, 25, 20, 25],
    )

    return _build_campaign(well_data)


# pylint: disable=duplicate-code
@pytest.fixture(name="get_minimal_campaign", scope="function")
def get_minimal_campaign_fixture(get_eff_metrics):
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

    im_metrics.check_validity()

    im_metrics.delete_metric("other_losses")
    im_metrics.delete_metric("five_year_production_volume")
    im_metrics.delete_metric("well_integrity")
    im_metrics.delete_metric("environment")

    # Submetrics can also be deleted in a similar manner
    im_metrics.delete_submetric("buildings_near")
    im_metrics.delete_submetric("buildings_far")

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
        elevation_delta="Elevation Delta",
        # These are user-specific columns
    )

    data = {
        "API Well Number": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Leak [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Violation [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Incident [Yes/No]": {0: "Yes", 1: "Yes", 2: "No", 3: "No", 4: "Yes", 5: "Yes"},
        "Compliance [Yes/No]": {0: "No", 1: "Yes", 2: "No", 3: "Yes", 4: "No", 5: "No"},
        "Oil [bbl/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Gas [Mcf/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Age [Years]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Depth [ft]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Operator Name": {
            0: "Owner 56",
            1: "Owner 136",
            2: "Owner 137",
            3: "Owner 190",
            4: "Owner 196",
            5: "Owner 196",
        },
        "x": {0: 0.99982, 1: 0.99995, 2: 1.51754, 3: 1.51776, 4: 1.51964, 5: 1.51931},
        "y": {0: 1.95117, 1: 1.9572, 2: 1.9584, 3: 1.95746, 4: 1.95678, 5: 1.95674},
        "Number of Nearby Hospitals": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
        "Number of Nearby Schools": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
        "Elevation Delta": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
    }

    well_data = WellData(
        pd.DataFrame(data),
        col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=get_eff_metrics,
    )

    well_data.compute_priority_scores()

    return Campaign(
        well_data,
        {1: [0, 1], 2: [2, 3], 3: [4]},
        {1: 10, 2: 15, 3: 20},
        OptModelInputs(
            mobilization_cost=MOBILIZATION_COST,
            total_budget=TOTAL_BUDGET,
            well_data=well_data,
        ),
    )


@pytest.fixture(name="get_project", scope="function")
def get_project_fixture(get_campaign):
    return get_campaign.projects[2]


@pytest.fixture(name="get_eff_metrics_accessibility", scope="function")
def get_eff_metrics_accessibility_fixture():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 10,
            "num_unique_owners": 30,
            "elevation_delta": 20,
            "age_range": 10,
            "depth_range": 20,
            "dist_to_road": 10,
        }
    )

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


def test_check_column_exists(get_project):
    get_project.column_names.hospitals = None
    with pytest.raises(
        MissingDataError,
        match="hospitals data is not in the input well data",
    ):
        print(get_project.num_wells_near_hospitals)


# test Project Class
def test_project_attributes(get_project):
    project = get_project
    for index in project:
        assert index in project.well_data.data.index
    assert len(project.well_data.data) == 2
    assert project.project_id == 2
    assert project.num_wells == 2
    assert project.plugging_cost == 10e6
    assert project.efficiency_score == pytest.approx(61.4333333)
    assert project.num_wells_near_hospitals == 2
    assert project.num_wells_near_schools == 2
    assert project.average_age == 1.5
    assert project.age_range == 1
    assert project.average_depth == 1.5
    assert project.depth_range == 1
    assert project.elevation_delta == 2
    assert project.centroid == (0.999885, 1.954185)
    assert project.dist_to_road == 2
    assert project.num_unique_owners == 2
    assert project.impact_score == 38.25
    delattr(project.column_names, "priority_score")
    with pytest.raises(AttributeError):
        project.impact_score += 2.0


def test_project_attributes_minimal(get_minimal_campaign):
    project = get_minimal_campaign.projects[1]

    # checking for missing attributes
    with pytest.raises(
        MissingDataError,
        match="dist_to_road data is not in the input well data",
    ):
        print(project.dist_to_road)


def test_max_val_col(get_project):
    project = get_project
    assert project.get_max_val_col(project.column_names.age) == 2


def test_get_well_info_data_frame(get_project):
    project = get_project
    well_data = project.get_well_info_dataframe()
    assert (
        "Violation [Yes/No]" == project.column_names.violation
        and "Violation [Yes/No]" not in well_data.columns
    )
    assert len(well_data) == 2
    assert all(i in [1, 2] for i in well_data["Age [Years]"].values)


def test_compute_accessibility_score(get_campaign, get_eff_metrics_accessibility):
    # this can only be called after the efficiency scores have been assigned
    get_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_eff_metrics_accessibility
    )
    # Creating a new campaign object, because overwriting efficiency
    # metrics is not allowed.
    new_campaign = _build_campaign(get_campaign.wd)
    project = new_campaign.projects[2]
    assert project.accessibility_score == (
        30,
        pytest.approx(20),
    )
    delattr(project, "elevation_delta_eff_score_0_20")
    delattr(project, "dist_to_road_eff_score_0_10")
    assert project.accessibility_score is None


def test_compute_accessibility_score_2(get_campaign):
    # this can only be called after the efficiency scores have been assigned
    project = get_campaign.projects[2]
    assert project.accessibility_score == (20, pytest.approx(13.333333))


def test_project_str(get_project):
    output = (
        "======================================================================"
        "\nProject ID: 2 \n\n"
        "Number of wells in the project\t\t: 2\n"
        "Estimated project cost\t\t\t: $10,000,000\n"
        "Sum of impact scores of all wells\t: 76.50\n"
        "Average impact Score [0-100]\t\t: 38.25\n"
        "Efficiency Score [0-100]\t\t: 61.43\n"
        "\nNumber of wells that are near a hospital: 2"
        "\nNumber of wells that are near a school\t: 2\n"
        "======================================================================"
    )
    assert str(get_project) == output


# test Campaign Class
def test_campaign_attributes(get_campaign):
    campaign = get_campaign
    assert len(campaign.projects) == 3
    assert campaign.num_projects == 3
    assert campaign.total_plugging_cost == 45e6
    assert len(campaign.wd) == 6
    # already tested the project string function
    msg = (
        "Optimal campaign has 3 projects and the cost is $45,000,000."
        "\n\nCampaign Summary:\n\n"
    )
    data = campaign.get_campaign_summary()
    data["Plugging Cost [$]"] = data["Plugging Cost [$]"].apply(
        lambda cost: f"{round(cost):,}"
    )
    assert str(campaign) == msg + data.to_string(index=False)
    assert campaign.efficiency_calculator.efficiency_weights is not None


@pytest.mark.parametrize(
    ("campaign_fixture", "use_user_plugging_cost"),
    [
        ("get_campaign", False),
        ("get_campaign_with_plugging_cost", True),
    ],
)
def test_update_campaign(campaign_fixture, use_user_plugging_cost, request, caplog):
    get_campaign = request.getfixturevalue(campaign_fixture)

    campaign = get_campaign.update_campaign(remove_wells_dict={2: [1]})
    assert 0 not in campaign.clusters_dict[2]

    if use_user_plugging_cost:
        assert campaign.projects[2].plugging_cost == 10000
    else:
        assert campaign.projects[2].plugging_cost == MOBILIZATION_COST[1]

    with pytest.raises(ValueError, match="Project ID 1 is not in the campaign."):
        get_campaign.update_campaign(remove_wells_dict={1: [3]})

    with pytest.raises(ValueError, match="Project ID 1 is not in the campaign."):
        get_campaign.update_campaign(add_wells_dict={1: [4]})

    with pytest.raises(ValueError, match="Well ID 3 is not in project 2."):
        get_campaign.update_campaign(remove_wells_dict={2: [3]})

    campaign = campaign.update_campaign(add_wells_dict={3: [1]})
    assert 0 in campaign.clusters_dict[3]

    if use_user_plugging_cost:
        assert campaign.projects[3].plugging_cost == 30000
    else:
        assert campaign.projects[3].plugging_cost == MOBILIZATION_COST[3]

    with pytest.raises(ValueError, match="Well ID 1 is already in project 2."):
        get_campaign.update_campaign(add_wells_dict={2: [1]})

    campaign = get_campaign.update_campaign(add_wells_dict={2: [3]})
    assert 2 in campaign.clusters_dict[2]
    assert 2 not in campaign.clusters_dict[3]

    if use_user_plugging_cost:
        assert campaign.projects[2].plugging_cost == 30000
        assert campaign.projects[3].plugging_cost == 10000
    else:
        assert campaign.projects[2].plugging_cost == MOBILIZATION_COST[3]
        assert campaign.projects[3].plugging_cost == MOBILIZATION_COST[1]

    with caplog.at_level(logging.INFO):
        get_campaign.update_campaign(remove_wells_dict={2: [1, 2]})

    assert (
        "Project 2 has no wells after campaign update. Removing project." in caplog.text
    )

    with pytest.raises(
        ValueError, match="Updated campaign must contain at least one project."
    ):
        get_campaign.update_campaign(
            remove_wells_dict={2: [1, 2], 3: [3, 4], 4: [5, 6]}
        )


@pytest.mark.parametrize(
    "well_id, project_ids, expected_project_id",
    [
        (1, [3, 4], 3),
        (2, [3, 4], 3),
        (3, [2, 4], 2),
        (4, [2, 4], 4),
        (5, [2, 3], 3),
        (6, [2, 3], 3),
    ],
)
def test_find_nearby_efficient_project_id(
    get_campaign, well_id, project_ids, expected_project_id
):
    assert (
        find_nearby_efficient_project_id(well_id, project_ids, get_campaign)
        == expected_project_id
    )


def test_get_max_value_across_all_projects(get_campaign):
    assert get_campaign.get_max_value_across_all_projects("average_depth") == 5.5
    assert get_campaign.get_max_value_across_all_projects("num_wells_near_schools") == 2
    with pytest.raises(AttributeError):
        get_campaign.get_max_value_across_all_projects("Dev is a good dev")


def test_get_min_value_across_all_projects(get_campaign):
    assert get_campaign.get_min_value_across_all_projects("average_depth") == 1.5
    assert get_campaign.get_min_value_across_all_projects("num_wells_near_schools") == 2
    with pytest.raises(AttributeError):
        get_campaign.get_min_value_across_all_projects("Dev is a good dev")


# errors checked in test_check_col_in_data
def test_get_max_value_across_all_wells(get_campaign):
    assert get_campaign.get_max_value_across_all_wells("Depth [ft]") == 6
    assert get_campaign.get_max_value_across_all_wells("Age [Years]") == 6


def test_get_min_value_across_all_wells(get_campaign):
    assert get_campaign.get_min_value_across_all_wells("Depth [ft]") == 1
    assert get_campaign.get_min_value_across_all_wells("Age [Years]") == 1


# for now leaving the plotting out of the tests
def test_get_project_well_information(get_campaign):
    info = get_campaign.get_project_well_information()
    assert all(i in [2, 3, 4] for i in info.keys())
    # already tested well_info_dataframe


def test_get_efficiency_score_project(get_campaign):
    # Testing the case where efficiency metrics are not specified
    # by default
    get_campaign.wd.config.efficiency_metrics = None
    # Creating a new campaign object, because overwriting efficiency
    # metrics is not allowed.
    new_campaign = _build_campaign(get_campaign.wd)
    project = new_campaign.projects[2]
    assert new_campaign.get_efficiency_score_project(2) == 0
    assert project.efficiency_metric_scores == {}
    assert project.max_efficiency_metric_scores == {}


def test_get_impact_score_project(get_campaign):
    assert get_campaign.get_impact_score_project(2) == 38.25


def test_efficiency_metrics_column_headers(get_campaign):
    # pylint: disable=protected-access
    headers = get_campaign._efficiency_metrics_column_headers()
    assert headers == {
        "age_range": "Age Range Score [0-10]",
        "depth_range": "Depth Range Score [0-20]",
        "elevation_delta": "Elevation Delta Score [0-20]",
        "num_unique_owners": "Num Unique Owners Score [0-30]",
        "num_wells": "Num Wells Score [0-20]",
    }


# pylint: disable=duplicate-code
def test_campaign_summary(get_campaign):
    summary = get_campaign.get_campaign_summary()
    assert all(
        i
        in [
            "Project ID",
            "Number of Wells",
            "Plugging Cost [$]",
            "Impact Score [0-100]",
            "Efficiency Score [0-100]",
        ]
        for i in summary.columns
    )
    assert list(summary["Project ID"].values) == [2, 3, 4]
    assert list(summary["Plugging Cost [$]"]) == [10e6, 15e6, 20e6]
    assert summary["Impact Score [0-100]"].values[0] == 38.25
    assert len(summary) == 3


# test the Efficiency Calculator
def test_set_efficiency_weights(get_campaign, get_eff_metrics, caplog):
    campaign = get_campaign
    eff_metrics = get_eff_metrics

    assert (
        "Efficiency metrics are not specified, so skipping " "efficiency calculations."
    ) not in caplog.text
    assert campaign.efficiency_calculator.efficiency_weights is get_eff_metrics
    with pytest.raises(
        RuntimeError,
        match="Attempting to overwrite efficiency_weights",
    ):
        campaign.set_efficiency_weights(eff_metrics)

    # Creating a new campaign object, because overwriting efficiency
    # metrics is not allowed.
    get_campaign.wd.config.efficiency_metrics = None
    new_campaign = _build_campaign(get_campaign.wd)
    assert (
        "Efficiency metrics are not specified, so skipping " "efficiency calculations."
    ) in caplog.text
    assert new_campaign.efficiency_calculator.efficiency_weights is None

    new_campaign.set_efficiency_weights(get_eff_metrics)
    new_campaign.compute_efficiency_scores()
    assert get_campaign.projects[2].efficiency_metric_scores == pytest.approx(
        new_campaign.projects[2].efficiency_metric_scores
    )


@pytest.fixture(name="get_efficiency_calculator", scope="function")
def get_efficiency_calculator_fixture(get_campaign, get_eff_metrics):
    # get_campaign is instantiated with get_eff_metrics, so
    # returning the campaign object as it
    assert get_campaign.wd.config.efficiency_metrics is get_eff_metrics
    return get_campaign


@pytest.fixture(name="get_efficiency_metrics_minimal", scope="function")
def get_efficiency_metrics_minimal_fixture():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 0,
            "age_range": 30,
            "depth_range": 30,
            "num_unique_owners": 40,
        }
    )

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


def test_compute_efficiency_score_edge_cases(
    get_minimal_campaign, get_efficiency_metrics_minimal
):
    get_minimal_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_efficiency_metrics_minimal
    )

    new_campaign = Campaign(
        get_minimal_campaign.wd,
        {1: [0, 1], 2: [2, 3], 3: [4]},
        {1: 10, 2: 15, 3: 20},
        get_minimal_campaign.opt_model_inputs,
    )
    assert all(
        "num_wells_eff_score" not in entry for entry in dir(new_campaign.projects[1])
    )


def test_single_well(get_minimal_campaign, get_eff_metrics_simple):
    get_minimal_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_eff_metrics_simple
    )
    new_campaign = Campaign(
        get_minimal_campaign.wd,
        {1: [0, 1], 2: [2, 3], 3: [4]},
        {1: 10, 2: 15, 3: 20},
        get_minimal_campaign.opt_model_inputs,
    )
    assert new_campaign.projects[3].efficiency_score == pytest.approx(30.8)


def test_zeros(get_minimal_campaign, get_efficiency_metrics_minimal):
    get_minimal_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_efficiency_metrics_minimal
    )
    get_minimal_campaign.wd.data["Age [Years]"] = np.zeros(
        len(get_minimal_campaign.wd.data["Age [Years]"])
    )
    new_campaign = Campaign(
        get_minimal_campaign.wd,
        {1: [0, 1], 2: [2, 3], 3: [4]},
        {1: 10, 2: 15, 3: 20},
        get_minimal_campaign.opt_model_inputs,
    )
    assert new_campaign.projects[1].age_range_eff_score_0_30 == 30.0


def test_compute_efficiency_attributes_for_project(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    project = campaign.projects[2]
    assert project.num_wells_eff_score_0_20 == pytest.approx(1.6)
    assert project.num_unique_owners_eff_score_0_30 == pytest.approx(22.5)
    assert project.elevation_delta_eff_score_0_20 == pytest.approx(13.33333)
    assert project.age_range_eff_score_0_10 == pytest.approx(8)
    assert project.depth_range_eff_score_0_20 == pytest.approx(16)


def test_compute_overall_efficiency_scores_project(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    project = campaign.projects[2]
    assert project.efficiency_score == pytest.approx(61.4333)


def test_compute_efficiency_attributes_for_all_projects(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    for _, project in campaign.projects.items():
        assert project.num_wells_eff_score_0_20 >= 0.0
        assert project.num_unique_owners_eff_score_0_30 >= 0.0
        assert project.elevation_delta_eff_score_0_20 >= 0.0
        assert project.age_range_eff_score_0_10 >= 0.0
        assert project.depth_range_eff_score_0_20 >= 0.0


def test_compute_efficiency_scores(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    for _, project in campaign.projects.items():
        assert project.efficiency_score > 0


# last test for the campaign class
def test_get_efficiency_metric_scores(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    efficiency_metric_output = campaign.get_efficiency_metric_scores().data
    assert {
        "Project ID",
        "Age Range Score [0-10]",
        "Depth Range Score [0-20]",
        "Elevation Delta Score [0-20]",
        "Num Unique Owners Score [0-30]",
        "Num Wells Score [0-20]",
        "Efficiency Score [0-100]",
    } == set(efficiency_metric_output.columns)
    assert "Accessibility Score [0-20]" not in efficiency_metric_output.columns
    assert len(efficiency_metric_output) == 3

    # Tests the existence of columns as well as values
    assert list(efficiency_metric_output["Project ID"]) == pytest.approx([2, 3, 4])
    assert list(efficiency_metric_output["Num Wells Score [0-20]"]) == pytest.approx(
        [1.6, 1.6, 1.6]
    )
    assert list(
        efficiency_metric_output["Num Unique Owners Score [0-30]"]
    ) == pytest.approx([22.5, 22.5, 30.0])
    assert list(
        efficiency_metric_output["Elevation Delta Score [0-20]"]
    ) == pytest.approx([13.3333, 6.6666, 0], rel=0.001)
    assert list(efficiency_metric_output["Age Range Score [0-10]"]) == pytest.approx(
        [8, 8, 8], rel=0.001
    )
    assert list(efficiency_metric_output["Depth Range Score [0-20]"]) == pytest.approx(
        [16, 16, 16], rel=0.001
    )
    assert list(efficiency_metric_output["Efficiency Score [0-100]"]) == pytest.approx(
        [57.4333, 50.77, 51.6], rel=0.1
    )


def test_get_accessibility_scores(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    accessibility_score = campaign.get_accessibility_scores()
    assert list(accessibility_score.columns) == [
        "Project ID",
        "Accessibility Score [0-20]",
    ]
    assert list(accessibility_score["Project ID"]) == [2, 3, 4]
    assert list(accessibility_score["Accessibility Score [0-20]"]) == pytest.approx(
        [13.3333, 6.6666, 0], rel=0.001
    )


def test_display_efficiency_metric_scores(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    efficiency_metric_output = campaign.display_efficiency_metric_scores().data
    assert {
        "Project ID",
        "Age Range Score [0-10]",
        "Depth Range Score [0-20]",
        "Elevation Delta Score [0-20]",
        "Num Unique Owners Score [0-30]",
        "Num Wells Score [0-20]",
        "Accessibility Score [0-20]",
        "Efficiency Score [0-100]",
    } == set(efficiency_metric_output.columns)
    assert len(efficiency_metric_output) == 3

    # Tests the existence of columns as well as values
    assert list(efficiency_metric_output["Project ID"]) == pytest.approx([2, 3, 4])
    assert list(efficiency_metric_output["Num Wells Score [0-20]"]) == pytest.approx(
        [1.6, 1.6, 1.6]
    )
    assert list(
        efficiency_metric_output["Num Unique Owners Score [0-30]"]
    ) == pytest.approx([22.5, 22.5, 30.0])
    assert list(
        efficiency_metric_output["Elevation Delta Score [0-20]"]
    ) == pytest.approx([13.3333, 6.6666, 0], rel=0.001)
    assert list(efficiency_metric_output["Age Range Score [0-10]"]) == pytest.approx(
        [8, 8, 8], rel=0.001
    )
    assert list(efficiency_metric_output["Depth Range Score [0-20]"]) == pytest.approx(
        [16, 16, 16], rel=0.001
    )
    assert list(efficiency_metric_output["Efficiency Score [0-100]"]) == pytest.approx(
        [57.4333, 50.77, 51.6], rel=0.1
    )
    assert list(
        efficiency_metric_output["Accessibility Score [0-20]"]
    ) == pytest.approx([13.3333, 6.6666, 0], rel=0.001)


def test_export_data_to_excel(tmp_path, get_campaign):
    output_file_path = tmp_path / "export_data_test.xlsx"
    campaigns = [get_campaign]
    campaign_labels = ["export data test"]
    export_data_to_excel(output_file_path, campaigns, campaign_labels)
    assert True


def test_plot_campaign(get_campaign):
    get_campaign.plot_campaign("Just some toy data")
    plt.close()


def test_efficiency_scores_dict_eff_model(get_campaign):
    campaign = get_campaign
    with pytest.raises(
        RuntimeError,
        match=(
            "Calculated and model efficiency scores did not match. "
            "Please contact PRIMO developers."
        ),
    ):
        Campaign(
            campaign.wd,
            {2: [0, 1], 3: [2, 3], 4: [4, 5]},
            {2: 10, 3: 15, 4: 20},
            campaign.opt_model_inputs,
            efficiency_model_scores={
                2: {
                    "num_wells": 10,
                    "num_unique_owners": 15,
                    "elevation_delta": 20,
                    "age_range": 20,
                    "depth_range": 20,
                },
                3: {
                    "num_wells": 10,
                    "num_unique_owners": 15,
                    "elevation_delta": 20,
                    "age_range": 20,
                    "depth_range": 20,
                },
                4: {
                    "num_wells": 10,
                    "num_unique_owners": 15,
                    "elevation_delta": 20,
                    "age_range": 20,
                    "depth_range": 20,
                },
            },
        )

    with pytest.raises(
        KeyError,
        match=(
            "The keys from the efficiency model score computation "
            "and the keys of the project efficiency score computation"
            " do not match. Please contact PRIMO developers."
        ),
    ):
        Campaign(
            campaign.wd,
            {2: [0, 1], 3: [2, 3], 4: [4, 5]},
            {2: 10, 3: 15, 4: 20},
            campaign.opt_model_inputs,
            efficiency_model_scores={
                2: {
                    "Metric 1": 10,
                    "num_unique_owners": 15,
                    "elevation_delta": 20,
                    "age_range": 20,
                    "depth_range": 20,
                },
                3: {
                    "Metric 2": 10,
                    "num_unique_owners": 15,
                    "elevation_delta": 20,
                    "age_range": 20,
                    "depth_range": 20,
                },
                4: {
                    "Metric 3": 10,
                    "num_unique_owners": 15,
                    "elevation_delta": 20,
                    "age_range": 20,
                    "depth_range": 20,
                },
            },
        )

    new_campaign = Campaign(
        campaign.wd,
        {2: [0, 1], 3: [2, 3], 4: [4, 5]},
        {2: 10, 3: 15, 4: 20},
        campaign.opt_model_inputs,
        efficiency_model_scores={
            2: campaign.projects[2].efficiency_metric_scores,
            3: campaign.projects[3].efficiency_metric_scores,
            4: campaign.projects[4].efficiency_metric_scores,
        },
    )
    assert new_campaign.projects[2].efficiency_score == pytest.approx(
        campaign.projects[2].efficiency_score
    )


def test_impact_score_custom_basis_less_than_100(
    get_campaign_custom_basis_less_than_100,
):
    campaign = get_campaign_custom_basis_less_than_100
    assert [project.impact_score for project in campaign.projects.values()] == [
        25,
        18.5,
        18,
    ]
    assert [
        project.modified_impact_score for project in campaign.projects.values()
    ] == [
        50,
        37,
        36,
    ]


def test_impact_score_custom_basis_greater_than_100(
    get_campaign_custom_basis_greater_than_100,
):
    campaign = get_campaign_custom_basis_greater_than_100
    assert [project.impact_score for project in campaign.projects.values()] == [
        12.5,
        87.5,
        35,
    ]
    assert [
        project.modified_impact_score for project in campaign.projects.values()
    ] == [
        25,
        175,
        70,
    ]


def test_impact_score_user_priority_scores_less_than_100(
    get_campaign_user_priority_scores_less_than_100,
):
    campaign = get_campaign_user_priority_scores_less_than_100
    assert [project.impact_score for project in campaign.projects.values()] == [
        25,
        30.5,
        29,
    ]
    assert [
        project.modified_impact_score for project in campaign.projects.values()
    ] == [
        50,
        61,
        58,
    ]


def test_impact_score_user_priority_scores_greater_than_100(
    get_campaign_user_priority_scores_greater_than_100,
):
    campaign = get_campaign_user_priority_scores_greater_than_100
    assert [project.impact_score for project in campaign.projects.values()] == [
        12.5,
        87.5,
        47.5,
    ]
    assert [
        project.modified_impact_score for project in campaign.projects.values()
    ] == [
        25,
        175,
        95,
    ]


def test_impact_score_user_priority_scores_equal_to_100(
    get_campaign_ranking_scores_equal_to_100,
):
    campaign = get_campaign_ranking_scores_equal_to_100
    assert [project.impact_score for project in campaign.projects.values()] == [
        25,
        30.5,
        29,
    ]
    assert [
        project.modified_impact_score for project in campaign.projects.values()
    ] == [
        25,
        30.5,
        29,
    ]


def test_impact_score_custom_basis_greater_than_100_scores_less_than_100(
    get_campaign_custom_basis_greater_than_100_scores_less_than_100,
):
    campaign = get_campaign_custom_basis_greater_than_100_scores_less_than_100
    assert [project.impact_score for project in campaign.projects.values()] == [
        12.5,
        25,
        22.5,
    ]
    assert [
        project.modified_impact_score for project in campaign.projects.values()
    ] == [
        25,
        50,
        45,
    ]
