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
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.environ import (
    Binary,
    Constraint,
    NonNegativeReals,
    RangeSet,
    Set,
    UnitInterval,
    Var,
)

# User-defined libs
from primo.data_parser.metric_data import Metric

LOGGER = logging.getLogger(__name__)


@declare_custom_block("ZoneFormulationBlock", rule="compute_metric_score")
class ZoneFormulationBlockData(BlockData):
    """Template for building zone-based efficiency model"""

    @property
    def cluster_model(self):
        """Returns a pointer to the cluster model"""
        return self.parent_block().parent_block()

    @property
    def plugging_campaign_model(self):
        """
        Returns a pointer to the plugging campaign model
        """
        return self.parent_block().parent_block().parent_block()

    # pylint: disable = unused-argument, attribute-defined-outside-init
    def compute_metric_score(
        self, *args, zone_wells: pd.Series, metric: Metric, model_type: str
    ):
        """Builds the zone-based efficiency model for a given metric

        Parameters
        ----------
        zone_wells : pd.Series
            Vector containing zone data. Index contains well index (for well
            based metrics) or well pairs (for well pair metrics), and values
            contain zone number.

        metric : Metric
            Object containing metric data

        model_type : str
            Describes the type of constraints (Valid values: "weak" or "strong")
        """
        if metric.metric_type == "num_wells":
            self._build_num_wells_model(metric=metric)
            return

        cm = self.cluster_model
        self.score = Var(
            domain=NonNegativeReals,
            doc="Efficiency score for this metric",
        )
        if len(zone_wells) == 0:
            # This happens for pairwise metrics in the case of single well clusters.
            # Set of well pairs is null for single well clusters.
            LOGGER.warning(
                f"Fixing {metric.name} score to zero in cluster {cm.index()}, "
                f"because it is likely a single well cluster."
            )
            self.score.fix(0)
            return

        self.set_zones = RangeSet(zone_wells.min(), zone_wells.max())
        self.select_zone = Var(
            self.set_zones,
            domain=Binary,
            doc="Binary variable associated with each zone",
        )
        LOGGER.info(
            f"Zone range for metric {metric.name} in cluster {cm.index()} "
            f"is {min(self.set_zones)} - {max(self.set_zones)}"
        )
        self.choose_single_zone = Constraint(
            expr=sum(self.select_zone[:]) == cm.select_cluster
        )

        if len(self.set_zones) == 1:
            # No constraint on which wells/well pairs can be selected.
            # So, no need to introduce auxiliary variables in this case
            _score = (
                metric.effective_weight
                * metric.zone_data.efficiency[min(self.set_zones)]
            )
            self.calculate_score = Constraint(
                expr=self.score
                == _score * sum(n * cm.num_wells_var[n] for n in cm.set_num_wells)
            )
            return

        self.num_wells_var_zone = Var(
            cm.set_num_wells,
            self.set_zones,
            domain=UnitInterval,
            doc="Auxiliary variable = num_wells_var * select_zone",
        )

        # Based on the zone, some num_wells_var variables can be set to zero
        max_num_wells_in_project = max(cm.set_num_wells)
        if metric.metric_type == "well_based":
            for zone in self.set_zones:
                num_wells_in_zone = len(zone_wells[zone_wells <= zone])
                # Project cannot have more than this number if this zone is selected
                for n in range(num_wells_in_zone + 1, max_num_wells_in_project + 1):
                    self.num_wells_var_zone[n, zone].fix(0)
        else:
            # The following map helps identify the maximum number of wells we can
            # select based on the number of well pairs in a zone.
            num_pair_num_well_map = self.parent_block().num_well_pairs_num_wells_map
            for zone in self.set_zones:
                num_well_pairs_in_zone = len(zone_wells[zone_wells <= zone])
                num_wells_in_zone = num_pair_num_well_map[num_well_pairs_in_zone]
                for n in range(num_wells_in_zone + 1, max_num_wells_in_project + 1):
                    self.num_wells_var_zone[n, zone].fix(0)

        @self.Constraint(self.set_zones)
        def rlt_choose_one_num_wells_var(blk, z):
            return sum(blk.num_wells_var_zone[:, z]) == self.select_zone[z]

        @self.Constraint(cm.set_num_wells)
        def rlt_num_wells_var_definition(blk, n):
            return cm.num_wells_var[n] == sum(blk.num_wells_var_zone[n, :])

        # The following variable helps to set efficiency to zero for well pair
        # metrics if a single-well project is chosen
        mult_factor = {n: 1 for n in cm.set_num_wells}
        if 1 in mult_factor and metric.metric_type == "well_pair":
            mult_factor[1] = 0

        coeff = metric.zone_data.efficiency
        self.calculate_score = Constraint(
            expr=self.score
            == metric.effective_weight
            * sum(
                mult_factor[n] * n * coeff[z] * self.num_wells_var_zone[n, z]
                for n in cm.set_num_wells
                for z in self.set_zones
            )
        )

        if metric.metric_type == "well_based" and model_type == "weak":
            # The remaining variables are not necessary, so skipping them
            self._build_weaker_well_based_model(zone_wells)
            return

        self.select_well_zone = Var(
            cm.set_wells,
            self.set_zones,
            domain=UnitInterval,
            doc="Auxiliary variable = select_well * select_zone",
        )

        # Convex hull of the bilinear terms
        @self.Constraint(self.set_zones)
        def rlt_compute_num_wells_constraint(blk, z):
            return sum(blk.select_well_zone[:, z]) == sum(
                n * blk.num_wells_var_zone[n, z] for n in cm.set_num_wells
            )

        @self.Constraint(cm.set_wells)
        def rlt_select_well_definition(blk, w):
            return cm.select_well[w] == sum(blk.select_well_zone[w, :])

        @self.Constraint(cm.set_wells, self.set_zones)
        def upper_bound_select_well_zone_vars(blk, w, z):
            return blk.select_well_zone[w, z] <= self.select_zone[z]

        # Build remaining constraints based on metric type
        if model_type == "strong":
            getattr(self, "_build_" + metric.metric_type + "_model")(zone_wells)
        else:
            # Code reaches here only for weak well-pair case
            self._build_weaker_well_pair_model(zone_wells)

    def _build_num_wells_model(self, metric: Metric):
        """Builds constraints to limit number of wells based on zones"""
        self.score = Var(
            domain=NonNegativeReals,
            doc="Efficiency score for this metric",
        )

        cm = self.cluster_model
        scaling_factor = metric.normalization_factor
        efficiencies = {n: min(1, n / scaling_factor) for n in cm.set_num_wells}

        self.calculate_score = Constraint(
            expr=self.score
            == metric.effective_weight
            * sum(n * efficiencies[n] * cm.num_wells_var[n] for n in cm.set_num_wells)
        )

    def _build_num_unique_owners_model(self):
        """Builds constraints to limit number of unique owners based on zones"""
        raise NotImplementedError("This metric is not supported currently.")

    def _build_well_based_model(self, zone_wells: pd.Series):
        """
        Builds constraints to eliminate wells based on zones for well-based metrics
        """
        # suppose that well w is located in zone z, then it can not be selected if
        # any of the prior zones is selected. So, we need
        # w . zone_1 = w . zone_2 = ... = w . zone_{z-1} = 0
        starting_zone = min(self.set_zones)
        for well, zone in zone_wells.items():
            for z in range(starting_zone, zone):
                self.select_well_zone[well, z].fix(0)

    def _build_well_pair_model(self, zone_wells: pd.Series):
        reject_well_pairs = []
        for zone in self.set_zones:
            well_pairs_out_of_zone = zone_wells[zone_wells > zone].index
            reject_well_pairs += [(zone, *wp) for wp in well_pairs_out_of_zone]

        @self.Constraint(reject_well_pairs)
        def eliminate_well_pairs(blk, z, w1, w2):
            return (
                blk.select_well_zone[w1, z] + blk.select_well_zone[w2, z]
                <= blk.select_zone[z]
            )

    def _build_weaker_well_based_model(self, zone_wells: pd.Series):
        cm = self.cluster_model

        @self.Constraint(self.set_zones)
        def acceptable_wells(blk, z):
            allowed_wells = zone_wells[zone_wells <= z].index
            return sum(
                n * blk.num_wells_var_zone[n, z] for n in cm.set_num_wells
            ) <= sum(cm.select_well[w] for w in allowed_wells)

        # @self.Constraint(self.set_zones)
        # def eliminate_wells(blk, z):
        #     reject_wells = zone_wells[zone_wells > z].index

        #     if len(reject_wells) == 0:
        #         return Constraint.Skip

        #     return sum(cm.select_well[w] for w in reject_wells) <= len(reject_wells) * (
        #         cm.select_cluster - blk.select_zone[z]
        #     )

    @staticmethod
    def construct_zone_matrix(zone_wells: pd.Series):
        """Unstack the "zone vector" for well pair metrics, and returns it
        as a DataFrame.

        Parameters
        ----------
        zone_wells : pd.Series
            Series containing zone information of all well pairs in the
            cluster. Index -> Well pair, and values -> zone number.

        Returns
        -------
        pd.DataFrame
            Matrix/DataFrame containing zone information
        """
        zone_matrix = zone_wells.unstack()
        zone_matrix.insert(loc=0, column=zone_matrix.index[0], value=0)
        zone_matrix.loc[zone_matrix.columns[-1]] = [0] * zone_matrix.shape[1]
        zone_matrix = zone_matrix.fillna(0)
        return (zone_matrix + zone_matrix.transpose()).astype(int)

    def _build_weaker_well_pair_model(self, zone_wells: pd.Series):
        """
        Builds constraints to eliminate well pairs based on zones for
        well pair-based metrics
        """
        cm = self.cluster_model
        zone_matrix = self.construct_zone_matrix(zone_wells)

        @self.Constraint(cm.set_wells, self.set_zones)
        def suppress_wells_rlt_1(blk, w, z):
            wells_outside_zone = zone_matrix[zone_matrix[w] > z].index
            if len(wells_outside_zone) == 0:
                # All wells pairs are within zone <= z
                return Constraint.Skip

            return sum(blk.select_well_zone[v, z] for v in wells_outside_zone) <= len(
                wells_outside_zone
            ) * (blk.select_zone[z] - blk.select_well_zone[w, z])

        # @self.Constraint(cm.set_wells, self.set_zones)
        # def suppress_wells_rlt_2(blk, w, z):
        #     wells_outside_zone = zone_matrix[zone_matrix[w] > z].index
        #     if len(wells_outside_zone) == 0:
        #         # All wells pairs are within zone <= z
        #         return Constraint.Skip

        #     return sum(
        #         cm.select_well[v] - blk.select_well_zone[v, z]
        #         for v in wells_outside_zone
        #     ) <= len(wells_outside_zone) * (cm.select_cluster - blk.select_zone[z])


def build_cluster_efficiency_model(eff_blk):
    """
    Builds efficiency model for each cluster

    Parameters
    ----------
    cm : ClusterBlock
        Cluster model object
    """
    # # For reference, this is the model Hierarchy
    # PluggingCampaignModel/Pyomo ConcreteModel
    #     |__ClusterBlock
    #         |__EfficiencyBlock
    #             |__ZoneFormulationBlock

    # OptModelInputs's config object that contains zone information
    cm = eff_blk.parent_block()  # cluster model block
    pm = cm.parent_block()  # Plugging campaign model/ConcreteModel
    wd = pm.model_inputs.config.well_data  # WellData object
    eff_metrics = wd.config.efficiency_metrics
    list_wells = list(cm.set_wells)  # List of wells in this cluster
    eff_blk.zone_wells = pd.DataFrame(index=list_wells)

    pairwise_metrics = pm.model_inputs.pairwise_metrics[cm.index()]
    if pairwise_metrics is not None:
        cm.set_well_pairs = Set(initialize=pairwise_metrics.index.to_list())
        # Creating a map to determine the maximum number of wells we can have
        # based on the number of well pairs selected
        eff_blk.num_well_pairs_num_wells_map = {
            round(i * (i - 1) / 2 + j): i
            for i in range(1, len(cm.set_wells))
            for j in range(i)
        }
        eff_blk.num_well_pairs_num_wells_map[
            len(eff_blk.num_well_pairs_num_wells_map)
        ] = len(cm.set_wells)

    for metric in eff_metrics.selected_metrics:
        # Collect information on wells belonging to different zones
        if metric.metric_type == "well_based":
            data = wd.data.loc[list_wells, metric.data_col_name]
            zone_data = pd.cut(
                data,
                bins=metric.zone_data.points,
                labels=metric.zone_data.zones,
                include_lowest=True,
            )
            eff_blk.zone_wells[metric.name] = zone_data  #  Storing zone data
            model_type = pm.model_inputs.config.zone_well_based_constraint_type

        elif metric.metric_type == "well_pair":
            data = pm.model_inputs.pairwise_metrics[cm.index()][metric.name]
            zone_data = pd.cut(
                data,
                bins=metric.zone_data.points,
                labels=metric.zone_data.zones,
                include_lowest=True,
            )
            pm.model_inputs.pairwise_metrics[cm.index()][
                metric.name + "_zones"
            ] = zone_data  # Storing zone data for reference
            model_type = pm.model_inputs.config.zone_well_pair_constraint_type

        else:
            zone_data = pd.Series([0, 0, 0])  # Dummy vector
            model_type = "none"

        # Raise an error if there is a well that is not assigned to
        # any of the zones.
        if zone_data.isnull().any():
            raise ValueError(
                f"Failed to assign some wells to specified zones for metric {metric.name}"
                f" in cluster {cm.index()}"
            )

        # pylint: disable = undefined-variable
        setattr(
            eff_blk,
            metric.name,
            ZoneFormulationBlock(
                zone_wells=zone_data, metric=metric, model_type=model_type
            ),
        )
