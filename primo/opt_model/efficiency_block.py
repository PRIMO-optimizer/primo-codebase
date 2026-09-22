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

# Installed libs
from pyomo.core.base.block import Block, BlockData, declare_custom_block
from pyomo.environ import NonNegativeReals, Var

# User-defined libs
from primo.opt_model.efficiency_max_formulation import build_cluster_efficiency_model

LOGGER = logging.getLogger(__name__)


@declare_custom_block("EfficiencyBlock", rule="build")
class EfficiencyBlockData(BlockData):
    """Container for storing the efficiency calculations"""

    # pylint: disable = unused-argument
    def build(self, *args, formulation_type):
        """
        Builds efficiency model
        """
        if formulation_type is None:
            # Return an empty efficiency block model
            pass
        elif formulation_type == "Max Scaling":
            build_cluster_efficiency_model(self)
        else:
            raise NotImplementedError("Zone formulation is not supported currently")
        self.append_cluster_eff_vars()

    # pylint: disable = attribute-defined-outside-init
    def append_cluster_eff_vars(self):
        """
        Adds the common constraints

        Parameters
        ----------
        eff_vars : list, optional, default = None
            List of efficiency variables. This is needed for the
            max formulation, and this is not needed for the zone
            formulation
        """

        cm = self.parent_block()  # ClusterBlock Model
        # Declare variables
        self.cluster_efficiency = Var(within=NonNegativeReals)
        self.aggregated_efficiency = Var(
            cm.num_wells_var.index_set(),
            within=NonNegativeReals,
            doc="num_wells_var * cluster_efficiency",
        )

        @self.Constraint(cm.num_wells_var.index_set())
        def calculate_aggregated_efficiency_1(blk, n):
            return blk.aggregated_efficiency[n] <= 100 * cm.num_wells_var[n]

        @self.Constraint()
        def calculate_aggregated_efficiency_2(blk):
            return (
                sum(blk.aggregated_efficiency[n] for n in cm.num_wells_var.index_set())
                == blk.cluster_efficiency
            )

        try:
            wt_impact = (
                self.parent_block()
                .parent_block()
                .model_inputs.config.objective_weight_impact
            )
        except AttributeError:
            wt_impact = 0
        wt_efficiency = (100 - wt_impact) / 100

        @self.Expression(doc="Total Efficiency Score")
        def cluster_efficiency_score(blk):
            return wt_efficiency * sum(
                n * cm.num_wells_adjustment[n] * blk.aggregated_efficiency[n]
                for n in cm.num_wells_var.index_set()
            )

        @self.Constraint(doc="Computes the efficiency of the project")
        def calculate_cluster_efficiency(blk):
            return blk.cluster_efficiency == sum(
                eff_metric_blk.score
                for eff_metric_blk in blk.component_data_objects(Block)
            )

    def get_efficiency_scores(self):
        """Returns a dictionary containing efficiency scores"""
        scores = {}
        for blk in self.component_data_objects(Block):
            scores[blk.name.split(".")[-1]] = blk.score.value

        return scores
