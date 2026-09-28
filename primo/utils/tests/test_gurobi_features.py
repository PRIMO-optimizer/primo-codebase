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

# Installed libs
import pandas as pd
import pyomo.environ as pyo
import pytest

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.opt_model.model_options import OptModelInputs

# pylint: disable = unused-import
from primo.opt_model.tests.test_efficiency_model import get_column_names_fixture
from primo.utils.gurobi_features import GurobiSolver

# Check if the test is running inside GitHub Actions
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(name="get_opt_model", scope="function")
def get_opt_model_fixture(get_column_names):
    """Builds an instance of the optimization model"""

    eff_metrics, im_metrics, col_names, filename = get_column_names
    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]
    wd_gas.compute_priority_scores()
    wd_gas = wd_gas.get_high_priority_wells(200)

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=1500000,  # 1.5 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        objective_weight_impact=100,
    )

    return opt_mdl_inputs


@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_object_instantiation(get_opt_model, writer_type):
    """Tests object instantiation"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.build_optimization_model()

    assert isinstance(opt_model_inputs.config.well_data.data, pd.DataFrame)
    assert isinstance(opt_model_inputs, OptModelInputs)

    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    assert solver.formulation_type == "Max Scaling"
    assert solver.model is opt_model_inputs.optimization_model


@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_set_branch_priorities(get_opt_model, writer_type):
    """Tests the set_branch_priorities method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)

    solver.set_branch_priorities(num_wells_vars_priority=50)

    m = opt_model_inputs.optimization_model
    for blk in m.cluster.values():
        for v in blk.num_wells_var.values():
            assert solver.pm_to_gb[v].BranchPriority == 50

        assert solver.pm_to_gb[blk.select_cluster].BranchPriority == 10000


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_set_partition_number(get_opt_model, writer_type):
    """Tests the set_partition_number method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)

    solver.set_partition_number()

    m = opt_model_inputs.optimization_model
    for idx, blk in enumerate(m.cluster.values()):
        for v in blk.component_data_objects(pyo.Var):
            try:
                assert solver.pm_to_gb[v].Partition == idx + 1
            except KeyError:
                # This variable is not used in constraints, so the writer
                # did not introduce a gurobi var for this pyomo var
                # This happens only with the new writer
                assert writer_type == "new"

    for v in m.component_data_objects(pyo.Var, descend_into=False):
        try:
            assert solver.pm_to_gb[v].Partition == 0
        except KeyError:
            assert writer_type == "new"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_make_efficiency_constraints_lazy(caplog, get_opt_model, writer_type):
    """Tests the set_efficiency_constraints_lazy method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.config.objective_weight_impact = 100
    opt_model_inputs.build_optimization_model()

    assert (
        "Efficiency calculation constraints are not present in "
        "the model. No constraint is made lazy."
    ) not in caplog.text

    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    assert not hasattr(solver.model.cluster[0], "efficiency_model")
    solver.make_efficiency_constraints_lazy()
    assert (
        "Efficiency calculation constraints are not present in "
        "the model. No constraint is made lazy."
    ) in caplog.text

    opt_model_inputs.config.objective_weight_impact = 50
    opt_model_inputs.build_optimization_model()

    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    assert hasattr(solver.model.cluster[0], "efficiency_model")
    solver.make_efficiency_constraints_lazy()

    m = opt_model_inputs.optimization_model
    assert (
        solver.solver.get_linear_constraint_attr(
            m.cluster[0].efficiency_model.num_wells.calculate_score, "Lazy"
        )
        == 0
    )

    eff_model = m.cluster[0].efficiency_model
    for constr in eff_model.dist_range.calculate_score.values():
        assert solver.solver.get_linear_constraint_attr(constr, "Lazy") == 1

    for constr in eff_model.elevation_delta.calculate_score.values():
        assert solver.solver.get_linear_constraint_attr(constr, "Lazy") == 1


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_change_variable_domain(caplog, get_opt_model, writer_type):
    """Tests the change_variable_domain method from binary to"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)

    # Testing the incorrect domain type
    with pytest.raises(ValueError, match="Unrecognized domain type foo"):
        solver.change_variable_domain(to_domain="foo", var_list=["select_well"])

    # Testing the incorrect variable type error
    with pytest.raises(
        ValueError,
        match=(
            "Unrecognized variable type foo. Supported variables are "
            "\\['select_cluster', 'select_well', 'num_wells_var', 'select_zone'\\]"
        ),
    ):
        solver.change_variable_domain(to_domain="unit_interval", var_list=["foo"])

    # Test binary to unit interval conversion (all variables)
    assert (
        "Variables names are not specified. Changing the "
        "domain of these variables \\['select_cluster', "
        "'select_well', 'num_wells_var', 'select_zone'\\]"
    ) not in caplog.text
    solver.change_variable_domain(to_domain="unit_interval")
    assert (
        "Variables names are not specified. Changing the "
        "domain of these variables \\['select_cluster', "
        "'select_well', 'num_wells_var', 'select_zone'\\]"
    ) not in caplog.text

    for blk in opt_model_inputs.optimization_model.cluster.values():
        for v in blk.select_well.values():
            assert v.domain is pyo.UnitInterval

        for v in blk.num_wells_var.values():
            assert v.domain is pyo.UnitInterval

        assert blk.select_cluster.domain is pyo.UnitInterval

    # Test unit interval to binary conversion
    solver.change_variable_domain(
        to_domain="binary", var_list=["select_cluster", "num_wells_var"]
    )
    for blk in opt_model_inputs.optimization_model.cluster.values():
        for v in blk.select_well.values():
            assert v.domain is pyo.UnitInterval

        for v in blk.num_wells_var.values():
            assert v.domain is pyo.Binary

        assert blk.select_cluster.domain is pyo.Binary


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_set_pool_ignore(get_opt_model, writer_type):
    """Tests the set_pool_ignore method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    solver.set_pool_ignore()

    m = opt_model_inputs.optimization_model
    for v in m.well_selected.values():
        assert solver.solver.get_var_attr(v, "PoolIgnore") == 1

    for blk in m.cluster.values():
        for v in blk.select_well.values():
            assert solver.solver.get_var_attr(v, "PoolIgnore") == 1


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_add_binary_cut(caplog, get_opt_model, writer_type):
    """Tests the add_binary_cut method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.config.objective_weight_impact = 100
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)

    assert (
        "Model is not initialized. Returning without adding a cut." not in caplog.text
    )
    solver.add_binary_cut()
    assert "Model is not initialized. Returning without adding a cut." in caplog.text
    assert len(solver.model.campaign_elimination_cuts) == 0

    # Solve the model and find the optimal solution
    solver.solve()
    model_size = solver.gurobi_model.NumConstrs

    # Add binary cut and solve the model
    solver.add_binary_cut()
    assert len(solver.model.campaign_elimination_cuts) == 1
    assert solver.gurobi_model.NumConstrs == model_size + 1
    solver.solve()

    solver.add_binary_cut()
    assert len(solver.model.campaign_elimination_cuts) == 2
    assert solver.gurobi_model.NumConstrs == model_size + 2


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_solve_method(get_opt_model, writer_type):
    """Tests the solve method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.config.objective_weight_impact = 100
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    solver.solve(MIPGap=0.01, TimeLimit=1000, tee=True, NodeLimit=1)

    # These tests ensure that the MIP is solved to a higher gap than the default gap
    assert solver.gurobi_model.MIPGap <= 0.01
    assert solver.gurobi_model.MIPGap > 0.001

    # Check if the default values have been over-written
    assert solver.gurobi_model.Params.MIPGap == pytest.approx(0.01)
    assert solver.gurobi_model.Params.TimeLimit == pytest.approx(1000)
    assert solver.gurobi_model.Params.NodeLimit == 1

    # Check if the parameter values have been restored to their defaults
    solver.reset_solver_params()
    assert solver.gurobi_model.Params.MIPGap == pytest.approx(1e-4)
    assert solver.gurobi_model.Params.TimeLimit >= 10000
    assert solver.gurobi_model.Params.NodeLimit != 1


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_get_solution_pool_sequentially(get_opt_model, writer_type):
    """Tests the get_solution_pool_sequentially method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.config.objective_weight_impact = 100
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)

    solutions = solver.get_solution_pool_sequentially(sol_count=3)
    assert isinstance(solutions, pd.DataFrame)
    assert solutions.shape[1] == 3


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_get_solution_pool(get_opt_model, writer_type):
    """Tests the get_solution_pool method"""
    opt_model_inputs = get_opt_model
    opt_model_inputs.config.objective_weight_impact = 100
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    solver.solve(PoolSearchMode=2, PoolSolutions=15)

    solutions = solver.get_solution_pool()
    assert isinstance(solutions, pd.DataFrame)
    assert solutions.shape[1] == 15


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is skipped in GitHub Actions.")
@pytest.mark.parametrize("writer_type", ["old", "new"])
def test_solve_with_lazy_constraints(get_opt_model, writer_type):
    """Tests the solve_with_lazy_constraints method"""
    # NOTE: This test exists only for code coverage. Other than that, we are
    # not testing anything specific here, since the model cannot be solved to
    # global optimality.
    opt_model_inputs = get_opt_model
    opt_model_inputs.build_optimization_model()
    solver = GurobiSolver(opt_model_inputs.optimization_model, writer_type=writer_type)
    solver.solve_with_lazy_constraints(TimeLimit=100)
