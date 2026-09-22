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

# Installed libs
import pandas as pd
import pyomo.environ as pyo
import pytest

# User-defined libs
# pylint: disable = no-name-in-module
from primo.opt_model.efficiency_max_formulation import MaxFormulationBlock

LOGGER = logging.getLogger(__name__)


@pytest.fixture(name="get_dummy_model", scope="function")
def get_dummy_model_fixture():
    """
    Pytest fixture to create a dummy model
    """
    m = pyo.ConcreteModel()

    m.cm = pyo.Block()  # Cluster model
    m.cm.set_wells = pyo.Set(initialize=[1, 2, 3, 4])
    m.cm.set_well_pairs = pyo.Set(
        initialize=[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    )
    m.cm.select_cluster = pyo.Var()
    m.cm.select_well = pyo.Var(m.cm.set_wells)
    m.cm.num_wells_var = pyo.Var(range(1, len(m.cm.set_wells) + 1), within=pyo.Binary)

    m.cm.eff_blk = pyo.Block()  # Efficiency Model
    return m


def test_well_based_model(get_dummy_model):
    """Tests the formulation for well-based metrics"""
    m = get_dummy_model
    eff_blk = m.cm.eff_blk

    eff_blk.metric = MaxFormulationBlock(
        weight=20,
        metric_data=pd.Series([2, 4, 8, 10], index=[1, 2, 3, 4]),
        scaling_factor=8.0,
        metric_type="well_based",
    )

    assert eff_blk.metric.cluster_model is m.cm
    assert hasattr(eff_blk.metric, "score")
    assert len(eff_blk.metric.calculate_score) == 4

    assert (
        str(eff_blk.metric.calculate_score[1].expr)
        == "cm.eff_blk.metric.score  <=  20*(cm.select_cluster - 0.25*cm.select_well[1])"
    )
    assert (
        str(eff_blk.metric.calculate_score[2].expr)
        == "cm.eff_blk.metric.score  <=  20*(cm.select_cluster - 0.5*cm.select_well[2])"
    )
    assert (
        str(eff_blk.metric.calculate_score[3].expr)
        == "cm.eff_blk.metric.score  <=  20*(cm.select_cluster - cm.select_well[3])"
    )
    assert (
        str(eff_blk.metric.calculate_score[4].expr)
        == "cm.eff_blk.metric.score  <=  20*(cm.select_cluster - cm.select_well[4])"
    )


def test_well_pair_based_model(get_dummy_model):
    """Tests the formulation for well-pair-type metrics"""
    m = get_dummy_model
    eff_blk = m.cm.eff_blk

    eff_blk.metric = MaxFormulationBlock(
        weight=20,
        metric_data=pd.Series(
            [4, 6, 8, 9, 10, 12],
            index=[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        ),
        scaling_factor=10.0,
        metric_type="well_pair",
    )

    assert eff_blk.metric.cluster_model is m.cm
    assert hasattr(eff_blk.metric, "score")
    assert len(eff_blk.metric.calculate_score) == 6

    expected_expressions = {
        (1, 2): "0.4*(cm.select_well[1] + cm.select_well[2] - cm.select_cluster)"
        "  <=  cm.select_cluster - 0.05*cm.eff_blk.metric.score",
        (1, 3): "0.6*(cm.select_well[1] + cm.select_well[3] - cm.select_cluster)"
        "  <=  cm.select_cluster - 0.05*cm.eff_blk.metric.score",
        (1, 4): "0.8*(cm.select_well[1] + cm.select_well[4] - cm.select_cluster)"
        "  <=  cm.select_cluster - 0.05*cm.eff_blk.metric.score",
        (2, 3): "0.9*(cm.select_well[2] + cm.select_well[3] - cm.select_cluster)"
        "  <=  cm.select_cluster - 0.05*cm.eff_blk.metric.score",
        (2, 4): "cm.select_well[2] + cm.select_well[4] - cm.select_cluster"
        "  <=  cm.select_cluster - 0.05*cm.eff_blk.metric.score",
        (3, 4): "cm.select_well[3] + cm.select_well[4] - cm.select_cluster"
        "  <=  cm.select_cluster - 0.05*cm.eff_blk.metric.score",
    }

    for pair, expected_expr in expected_expressions.items():
        assert str(eff_blk.metric.calculate_score[pair].expr) == expected_expr


def test_num_wells_model(get_dummy_model):
    """Tests the formulation for num_wells metric"""
    m = get_dummy_model
    eff_blk = m.cm.eff_blk

    eff_blk.metric = MaxFormulationBlock(
        weight=10,
        metric_data=pd.Series(),
        scaling_factor=20,
        metric_type="num_wells",
    )

    assert eff_blk.metric.cluster_model is m.cm
    assert hasattr(eff_blk.metric, "score")
    assert str(eff_blk.metric.calculate_score.expr) == (
        "cm.eff_blk.metric.score  <=  10*("
        "cm.select_well[1] + cm.select_well[2] + "
        "cm.select_well[3] + cm.select_well[4]"
        ")/20"
    )


# The test for the unique owner constraints are in the test_efficiency_model.py file
