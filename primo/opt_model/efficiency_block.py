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

"""
This module defines a new class called EfficiencyBlock. All the
efficiency related calculations will be contained in this block.

The hierarchy of the entire optimization model is as follows:
PluggingCampaignModel/Pyomo ConcreteModel
    |__ClusterBlock
        |__EfficiencyBlock
            |__EffMetricBlock
"""

# Standard libs
import logging
from types import SimpleNamespace

# Installed libs
import numpy as np
from pyomo.core.base.block import Block, BlockData, declare_custom_block
from pyomo.environ import Expression, NonNegativeReals, Var

# User-defined libs
from primo.opt_model.efficiency_max_formulation import (
    build_cluster_efficiency_model as max_eff_model,
)
from primo.opt_model.efficiency_zone_formulation import (
    build_cluster_efficiency_model as zone_eff_model,
)

LOGGER = logging.getLogger(__name__)


@declare_custom_block("EfficiencyBlock", rule="build")
class EfficiencyBlockData(BlockData):
    """Container for storing the efficiency calculations"""

    @property
    def cluster_model(self):
        """Returns a pointer to the corresponding cluster block"""
        return self.parent_block()

    @property
    def plugging_campaign_model(self):
        """Returns a pointer to the plugging campaign model"""
        return self.parent_block().parent_block()

    # pylint: disable = unused-argument
    def build(self, *args, formulation_type: str | None):
        """
        Builds efficiency model
        """
        if formulation_type is None:
            # Return an empty efficiency block model
            pass
        elif formulation_type == "Max Scaling":
            max_eff_model(self)
        else:
            zone_eff_model(self)

        self._append_cluster_eff_vars(formulation_type)

    # pylint: disable = attribute-defined-outside-init
    def _append_cluster_eff_vars(self, formulation_type: str | None):
        """
        Adds the common constraints

        Parameters
        ----------
        formulation_type : str | None
            Efficiency model type
        """

        try:
            params = self.plugging_campaign_model.model_inputs.config
        except AttributeError:
            # This exception occurs only in tests
            LOGGER.warning("Unable to find config block in OptModelInputs")
            params = SimpleNamespace(
                **{"embedding_b": 0.0, "objective_weight_impact": 0}
            )

        wt_efficiency = (100 - params.objective_weight_impact) / 100

        if formulation_type == "Zone":
            self.cluster_efficiency_score = Expression(
                expr=wt_efficiency
                * sum(blk.score for blk in self.component_data_objects(Block))
            )
            return

        cm = self.cluster_model  # ClusterBlock Model

        # Declare variables
        self.cluster_efficiency = Var(within=NonNegativeReals)
        self.aggregated_efficiency = Var(
            cm.set_num_wells,
            within=NonNegativeReals,
            doc="num_wells_var * cluster_efficiency",
        )

        @self.Constraint(cm.set_num_wells)
        def calculate_aggregated_efficiency_1(blk, n):
            return blk.aggregated_efficiency[n] <= 100 * cm.num_wells_var[n]

        @self.Constraint()
        def calculate_aggregated_efficiency_2(blk):
            return sum(blk.aggregated_efficiency[:]) == blk.cluster_efficiency

        @self.Param(cm.set_num_wells)
        def num_wells_adjustment(_, i):
            return (
                1
                if np.isclose(params.embedding_b, 0)
                else np.log10(params.embedding_b * i) + 1
            )

        @self.Expression(doc="Total Efficiency Score")
        def cluster_efficiency_score(blk):
            return wt_efficiency * sum(
                n * blk.num_wells_adjustment[n] * blk.aggregated_efficiency[n]
                for n in cm.set_num_wells
            )

        @self.Constraint(doc="Computes the efficiency of the project")
        def calculate_cluster_efficiency(blk):
            return blk.cluster_efficiency == sum(
                eff_metric_blk.score
                for eff_metric_blk in blk.component_data_objects(Block)
            )

    def get_efficiency_scores(self):
        """Returns a dictionary containing efficiency scores"""
        if hasattr(self, "cluster_efficiency"):
            # Score is NOT scaled by number of wells, so use it as is.
            return {
                blk.name.split(".")[-1]: blk.score.value
                for blk in self.component_data_objects(Block)
            }

        # Score is scaled by number of wells
        num_wells_chosen = round(self.cluster_model.num_wells_chosen.value)
        return {
            blk.name.split(".")[-1]: (
                blk.score.value / num_wells_chosen if num_wells_chosen >= 1 else 0
            )
            for blk in self.component_data_objects(Block)
        }
