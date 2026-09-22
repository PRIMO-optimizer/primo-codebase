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
from typing import Dict, List, Optional, Tuple

# Installed libs
import numpy as np
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.environ import (
    Binary,
    Constraint,
    Expression,
    NonNegativeReals,
    Param,
    RangeSet,
    Set,
    Var,
)

# User-defined libs
from primo.data_parser.default_data import EARTH_RADIUS
from primo.data_parser.well_data import WellData
from primo.utils.clustering_utils import build_ball_tree


# pylint: disable = too-many-instance-attributes
@declare_custom_block("ClusterBlock", rule="build")
class ClusterBlockData(BlockData):
    """
    A custom block class for storing variables and constraints
    belonging to a cluster.
    Essential variables and constraints will be added via "rule"
    argument. Here, define methods only for optional cluster-level
    constraints and expressions.
    """

    # pylint: disable = too-many-locals
    @staticmethod
    def _get_distant_well_pairs(
        wd: WellData, well_index: Dict, threshold_distance: Optional[float] = 10.0
    ) -> List[Tuple]:
        """
        Return well pairs within a cluster whose separation exceeds the
        threshold distance.

        Computes coefficients for efficiency score in the objective function

            Parameters
            ----------
            wd : WellData
                WellData object

            well_index : Dict
                All well indexes in the cluster

            threshold_distance: float
                Maximum distance [in miles] allowed between wells

        Returns
        -------
        List[Tuple]
            Returns a list of all well pairs whose separation exceeds the threshold distance

        Notes
        -----
        Every violating pair still has to be materialized for Pyomo, so the
        run time is ultimately bounded by the number of returned pairs. The
        BallTree avoids explicitly evaluating all pairwise distances.
        """
        well_ids = list(well_index)
        if threshold_distance is None or len(well_ids) < 2:
            return []

        cn = wd.col_names
        cluster_data = wd.data.loc[well_ids, [cn.latitude, cn.longitude]]
        coordinates = list(cluster_data.itertuples(index=False, name=None))
        ball_tree, coords_rad = build_ball_tree(coordinates)

        radius_radians = threshold_distance / EARTH_RADIUS
        neighbors_indices = ball_tree.query_radius(coords_rad, r=radius_radians)

        distant_pairs = []
        num_wells = len(well_ids)
        for i, w1 in enumerate(well_ids[:-1]):
            neighbors = np.asarray(neighbors_indices[i], dtype=int)
            all_after = np.arange(i + 1, num_wells)
            distant_pairs.extend(
                (w1, well_ids[j])
                for j in all_after[~np.isin(all_after, neighbors)].tolist()
            )

        return distant_pairs

    # pylint: disable = attribute-defined-outside-init, no-member
    # pylint: disable = too-many-locals, too-many-statements
    def build(self, cluster, override_data=None):
        """
        Builds the model block (adds essential variables and constraints)
        for a given cluster `cluster`
        """
        # Parameters are located in the parent block
        params = self.parent_block().model_inputs
        wd = params.config.well_data
        well_index = params.campaign_candidates[cluster]
        well_pairs_remove = self._get_distant_well_pairs(
            wd=wd,
            well_index=well_index,
            threshold_distance=params.config.threshold_distance,
        )

        # Essential model sets
        self.set_wells = Set(
            initialize=well_index,
            doc="Set of wells in cluster c",
        )
        self.set_num_wells = RangeSet(len(self.set_wells))
        self.set_well_pairs_remove = Set(
            dimen=2,
            initialize=well_pairs_remove,
            doc="Well-pairs which cannot be a part of the project",
        )

        # Essential variables
        self.select_cluster = Var(
            within=Binary,
            doc="1, if wells from the cluster are chosen for plugging, 0 Otherwise",
        )
        self.select_well = Var(
            self.set_wells,
            within=Binary,
            doc="1, if the well is selected for plugging, 0 otherwise",
        )
        self.num_wells_var = Var(
            range(1, len(self.set_wells) + 1),
            within=Binary,
            doc="Variables to track the total number of wells chosen",
        )

        def num_wells_efficiency_embedding(i, b):
            """
            Computes coefficients for efficiency score in the objective function

            Parameters
            ----------
            i : int
                The number of wells in a project
            b : float
                A parameter that adjusts the coefficient based on the number of wells

            Returns
            -------
            float
                The coefficient for efficiency score in the objective. Returns 1 if b is zero.
            """
            if abs(b) <= 1e-6:
                return 1
            return np.log10(b * i) + 1

        self.num_wells_adjustment = Param(
            range(1, len(self.set_wells) + 1),
            initialize=lambda i: num_wells_efficiency_embedding(
                i, params.config.embedding_b
            ),
        )

        self.plugging_cost = Var(
            within=NonNegativeReals,
            doc="Total cost for plugging wells in this cluster",
        )

        # Although the following two variables are of type Integer, they
        # can be declared as continuous. The optimal solution is guaranteed to have
        # integer values.
        self.num_wells_chosen = Var(
            within=NonNegativeReals,
            doc="Total number of wells chosen in the project",
        )

        self.min_well_violation = Var(
            within=NonNegativeReals,
            initialize=0,
            doc=(
                "Amount by which the number of wells falls short "
                "of the minimum required per project"
            ),
        )

        self.max_well_violation = Var(
            within=NonNegativeReals,
            initialize=0,
            doc="Amount by which the number of wells exceeds the maximum allowed per project",
        )

        self.is_distant_pair = Var(
            self.set_well_pairs_remove,
            within=Binary,
            doc="1 if both wells are selected and exceed distance threshold",
        )

        self.min_cost_project_violation = Var(
            within=NonNegativeReals,
            initialize=0,
            doc=(
                "Amount by which the plugging cost of a project falls "
                "short of the minimum budget per well"
            ),
        )

        self.max_cost_project_violation = Var(
            within=NonNegativeReals,
            initialize=0,
            doc=(
                "Amount by which the plugging cost of a project exceeds "
                "the maximum allowed per project"
            ),
        )

        # Set the maximum & minimum cost for a project: default is None.
        if params.get_max_cost_project is not None:
            self.set_upper_bound_project_cost = Constraint(
                expr=self.plugging_cost
                <= params.get_max_cost_project + self.max_cost_project_violation
            )
            if override_data is None or override_data.override_status is False:
                self.max_cost_project_violation.fix(0)

        if params.get_min_cost_project is not None:
            self.set_lower_bound_project_cost = Constraint(
                expr=self.select_cluster * params.get_min_cost_project
                <= self.plugging_cost + self.min_cost_project_violation
            )
            if override_data is None or override_data.override_status is False:
                self.min_cost_project_violation.fix(0)

        # Set the minimum and maximum number of wells allowed for a project
        # Defaults are none
        if params.config.min_wells_in_project is not None:
            self.min_project_size = Constraint(
                expr=self.select_cluster * params.config.min_wells_in_project
                <= self.num_wells_chosen + self.min_well_violation
            )
            if override_data is None or override_data.override_status is False:
                self.min_well_violation.fix(0)

        if params.config.max_wells_in_project is not None:
            self.max_project_size = Constraint(
                expr=self.num_wells_chosen
                <= params.config.max_wells_in_project + self.max_well_violation
            )
            if override_data is None or override_data.override_status is False:
                self.max_well_violation.fix(0)

        # Useful expressions
        # Scale the priority score so that the maximum priority score is 100
        # Keep the lowest priority score at whatever (scaled) nominal value it is
        priority_score = (
            wd["Priority Score [0-100]"] * 100 / wd["Priority Score [0-100]"].max()
        )
        wt_impact = params.config.objective_weight_impact / 100
        self.cluster_impact_score = Expression(
            expr=(
                wt_impact
                * sum(priority_score[w] * self.select_well[w] for w in self.set_wells)
            ),
            doc="Computes the total priority score for the cluster",
        )

        # Essential constraints
        self.calculate_num_wells_chosen = Constraint(
            expr=(
                sum(self.select_well[w] for w in self.set_wells)
                == self.num_wells_chosen
            ),
            doc="Calculate the total number of wells chosen",
        )

        # add plugging cost constraints
        self.add_plugging_cost_constraints(params, wd, well_index)

        self.campaign_length = Constraint(
            expr=(
                sum(i * self.num_wells_var[i] for i in self.num_wells_var)
                == self.num_wells_chosen
            ),
            doc="Determines the number of wells chosen",
        )
        self.num_well_uniqueness = Constraint(
            expr=(
                sum(self.num_wells_var[i] for i in self.num_wells_var)
                == self.select_cluster
            ),
            doc="Ensures at most one num_wells_var is selected",
        )

    def deactivate(self):
        """
        Deactivates the constraints present in this block.
        The variables will not be passed to the solver, unless
        they are used in other active constraints.
        """
        super().deactivate()
        self.select_cluster.fix(0)
        self.plugging_cost.fix(0)
        self.num_wells_chosen.fix(0)

    def activate(self):
        super().activate()
        self.select_cluster.unfix()
        self.plugging_cost.unfix()
        self.num_wells_chosen.unfix()

    def fix(
        self,
        cluster: Optional[int] = None,
        wells: Optional[Dict[int, int]] = None,
    ):
        """
        Fixes the binary variables associated with the cluster
        and/or the wells with in the cluster. To fix all variables
        within the cluster, use the fix_all_vars() method.

        Parameters
        ----------
        cluster : 0 or 1, default = None
            `select_cluster` variable will be fixed to this value.
            If None, select_cluster will be fixed to its incumbent value.

        wells : dict, default = None
            key => index of the well, value => value of `select_well`
            binary variable.
        """

        if cluster in [0, 1]:
            self.select_cluster.fix(cluster)

        if wells is not None:
            for w in self.set_wells:
                if w in wells:
                    self.select_well[w].fix(wells[w])

    def unfix(self):
        """
        Unfixes all the variables within the cluster.
        """
        self.unfix_all_vars()

    def add_distant_well_cuts(self):
        """
        Delete well pairs which are farther than the threshold distance
        """

        @self.Constraint(
            self.set_well_pairs_remove,
            doc="Removes well pairs which are far apart",
        )
        def skip_distant_well_cuts(b, w1, w2):
            return (
                b.select_well[w1] + b.select_well[w2]
                <= b.select_cluster + b.is_distant_pair[w1, w2]
            )

        if (
            self.parent_block().override_data is None
            or self.parent_block().override_data.override_status is False
        ):

            @self.Constraint(
                self.set_well_pairs_remove,
                doc="Fix is_distant_pairs to zero for well pairs violating threshold distance",
            )
            def fix_is_distant_pairs(b, w1, w2):
                return b.is_distant_pair[w1, w2] == 0

    def add_plugging_cost_constraints(self, param, wd, well_index):
        """
        Adds plugging cost constraints to the optimization model

        Parameters
        ----------
        param : OptModelInputs
            OptModelInputs object mainly used to access the config

        wd : WellData
            The WellData object used in the optimization model

        well_index : List
            List of well index in the cluster
        """
        if wd.col_names.cost_of_plugging is None:
            mob_cost = param.get_mobilization_cost
            self.calculate_plugging_cost = Constraint(
                expr=(
                    sum(mob_cost[i] * self.num_wells_var[i] for i in self.num_wells_var)
                    == self.plugging_cost
                ),
                doc="Calculates the total plugging cost for the cluster",
            )
            return
        beta = param.config.discount_cost_factor
        upc = param.get_plugging_cost
        if beta == 1:
            self.calculate_plugging_cost = Constraint(
                expr=(
                    sum(upc[w] * self.select_well[w] for w in self.set_wells)
                    == self.plugging_cost
                ),
                doc="Calculates the total plugging cost for the cluster",
            )
            return
        self.adjusted_plugging_cost = Var(
            within=NonNegativeReals,
            doc="Adjusted Total cost of plugging wells using economies of scale",
        )
        self.calculate_plugging_cost = Constraint(
            expr=(
                sum(upc[w] * self.select_well[w] for w in self.set_wells)
                == self.adjusted_plugging_cost
            ),
            doc="Calculates the total plugging cost before economies of scale",
        )
        # pylint: disable = not-an-iterable
        if param.config.min_wells_in_project is None:
            min_wells = 1
        else:
            min_wells = param.config.min_wells_in_project
        bigm_max = upc.loc[well_index].sum()
        beta_min = (min_wells**beta) / min_wells
        beta_max = (len(self.set_num_wells) ** beta) / len(self.set_num_wells)
        bigm_common = {i: bigm_max for i in self.set_num_wells}
        bigm_dict = {
            i: bigm_max * max(((i**beta) / i) - beta_max, beta_min - ((i**beta) / i))
            for i in list(self.set_num_wells)
        }
        if param.config.cost_formulation_type in ["standard_bigm", "alternate_bigm"]:
            bigm = (
                bigm_common
                if param.config.cost_formulation_type == "standard_bigm"
                else bigm_dict
            )
            self.calculate_adj_plugging_cost_1 = Constraint(
                self.set_num_wells,
                expr=lambda model, i: (
                    model.plugging_cost - (i**beta / i) * model.adjusted_plugging_cost
                    <= bigm[i] * (1 - model.num_wells_var[i])
                ),
                doc="Calculates the total plugging cost for the cluster (constraint 1)",
            )
            self.calculate_adj_plugging_cost_2 = Constraint(
                self.set_num_wells,
                expr=lambda model, i: (
                    model.plugging_cost - (i**beta / i) * model.adjusted_plugging_cost
                    >= -1 * bigm[i] * (1 - model.num_wells_var[i])
                ),
                doc="Calculates the total plugging cost for the cluster (constraint 2)",
            )
        else:
            bigm_dict = {i: upc.nlargest(i).sum() for i in self.num_wells_var}
            self.aux_adjusted_plugging_cost = Var(
                self.set_num_wells,
                within=NonNegativeReals,
                doc="Adjusted Total cost of plugging wells using economoies of scale",
            )
            self.calculate_aux_adjusted_plugging_cost = Constraint(
                expr=(
                    self.adjusted_plugging_cost
                    == sum(
                        self.aux_adjusted_plugging_cost[i] for i in self.num_wells_var
                    )
                ),
                doc="Calculates the total plugging cost before economies of scale",
            )

            @self.Constraint(self.set_num_wells)
            def adjusted_plugging_cost_bounds(model, i):
                return (
                    model.aux_adjusted_plugging_cost[i]
                    <= model.num_wells_var[i] * bigm_dict[i]
                )

            @self.Constraint()
            def plugging_cost_with_economies_of_scale(model):
                return model.plugging_cost == sum(
                    model.aux_adjusted_plugging_cost[i] * (i**beta / i)
                    for i in self.set_num_wells
                )
