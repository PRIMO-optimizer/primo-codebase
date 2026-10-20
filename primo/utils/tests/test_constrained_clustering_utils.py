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
import os

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo import OptModelInputs
from primo.data_parser import WellDataColumnNames
from primo.data_parser.well_data import WellData
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
    # pylint: disable=duplicate-code

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
    "objective": "target",
    "uniformity": 0,
    "max_iter": 10,
    "target_clusters": 10,
    "max_wells_in_project": 50,
    "min_wells_in_project": 5,
    "budget": 5e6,
}


@pytest.mark.parametrize(
    "config_override, result, expected_logs",
    [
        ({"objective": "minimum"}, 4, []),
        ({"objective": "maximum", "uniformity": 0.25}, 55, []),
        ({"uniformity": 0.5}, 10, []),
        ({"objective": "center", "uniformity": 1}, 28, []),
        (
            {"max_iter": 1, "target_clusters": 1, "max_wells_in_project": 20},
            RuntimeError,
            [],
        ),
        (
            {
                "objective": "center",
                "min_wells_in_project": 35,
                "max_wells_in_project": 25,
                "budget": 13e6,
            },
            ValueError,
            [],
        ),
        # Target exceeds the 275 total wells, so it is clamped to one cluster per well
        (
            {"target_clusters": 1000, "min_wells_in_project": None},
            275,
            [
                "The initial number of clusters is outside the feasible range."
                " Adjusted to 275."
            ],
        ),
        # Both checks fail on the first iteration
        (
            {
                "target_clusters": 100,
                "min_wells_in_project": 5,
                "max_wells_in_project": 5,
                "budget": 13.5e6,
            },
            RuntimeError,
            [
                "Iteration 0: Trying with 100 clusters",
                "Valid budget: False, Valid min constraints: False",
                "Could not find a valid clustering that satisfies all constraints. "
                "Please relax the constraints and try again.",
            ],
        ),
        # Budget sufficient case
        (
            {"objective": "minimum", "budget": 14e6},
            6,
            [
                "The budget is sufficient to plug all wells. "
                "Constrained heuristic clustering is designed for budget insufficient cases, "
                "consider using a different clustering method."
            ],
        ),
    ],
)
def test_perform_heuristic_constrained_clustering(
    config_override, result, expected_logs, make_well_data, caplog
):
    """
    Tests for perform_heuristic_constrained_clustering method
    """
    caplog.set_level(logging.INFO)
    config = {**DEFAULT_CONFIG, **config_override}
    wd = make_well_data(tail=None, cost_of_plugging=50000)

    if isinstance(result, int):
        opt_model_inputs = OptModelInputs(
            well_data=wd,
            total_budget=config["budget"],
            max_cost_project=None,
            min_cost_project=None,
            max_wells_in_project=config["max_wells_in_project"],
            min_wells_in_project=config["min_wells_in_project"],
            cluster_method="Constrained_Heuristic",
            constrained_clustering_objective=config["objective"],
            target_number_of_clusters=config["target_clusters"],
            uniformity_parameter=config["uniformity"],
            objective_weight_impact=100,
        )

        clusters = opt_model_inputs.campaign_candidates
        assert len(clusters) == result

    else:
        with pytest.raises(result):
            opt_model_inputs = OptModelInputs(
                well_data=wd,
                total_budget=config["budget"],
                max_cost_project=None,
                min_cost_project=None,
                max_wells_in_project=config["max_wells_in_project"],
                min_wells_in_project=config["min_wells_in_project"],
                cluster_method="Constrained_Heuristic",
                constrained_clustering_objective=config["objective"],
                target_number_of_clusters=config["target_clusters"],
                uniformity_parameter=config["uniformity"],
                constrained_clustering_max_iterations=config["max_iter"],
                objective_weight_impact=100,
            )

    for message in expected_logs:
        assert message in caplog.text


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
