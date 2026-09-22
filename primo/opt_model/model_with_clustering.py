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
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    Var,
    maximize,
)

# User-defined libs
# pylint: disable=no-name-in-module, import-error
from primo.opt_model.cluster_block import ClusterBlock
from primo.opt_model.efficiency_block import EfficiencyBlock
from primo.opt_model.result_parser import Campaign
from primo.utils.override_utils import distance_reassigned_well

LOGGER = logging.getLogger(__name__)


# pylint: disable-next = too-many-ancestors, too-many-instance-attributes
class PluggingCampaignModel(ConcreteModel):
    """
    Builds the optimization model
    """

    def __init__(self, model_inputs, *args, override_data=None, **kwargs):
        # pylint: disable=too-many-branches
        """
        Builds the optimization model for identifying the set of projects that
        maximize the overall impact and/or efficiency of plugging.

        Parameters
        ----------
        model_inputs : OptModelInputs
            Object containing the necessary inputs for the optimization model

        override_data : ReOptimizationData
            A ReOptimizationData object containing the necessary information
            related to the re-optimization of override
        """
        super().__init__(*args, **kwargs)
        self.override_data = override_data
        self.model_inputs = model_inputs
        self.set_clusters = Set(
            initialize=list(model_inputs.campaign_candidates.keys())
        )

        # Define only those parameters which are useful for sensitivity analysis
        self.total_budget = Param(
            initialize=model_inputs.get_total_budget,
            mutable=True,
            doc="Total budget available [Million USD]",
        )

        self.unused_budget_scaling = Param(
            initialize=0,
            mutable=True,
            within=NonNegativeReals,
            doc="Unused budget variable scaling factor in the objective function",
        )

        # Define variables and parameters related to override re-optimization
        self.unused_budget = Var(
            within=NonNegativeReals,
            doc="The unutilized amount of total budget",
        )

        self.excess_budget = Var(
            within=NonNegativeReals,
            doc="Amount by which plugging costs exceed the total available budget",
        )

        self.excess_owc = Var(
            list(self.model_inputs.owner_well_count.keys()),
            within=NonNegativeReals,
            doc="Number of wells exceeding the maximum allowed per well owner",
        )

        self.excess_budget_scaling = Param(
            initialize=0,
            mutable=True,
            within=NonNegativeReals,
            doc="Scaling factor for the excess budget",
        )

        self.override_slack_scaling = Param(
            initialize=0,
            mutable=True,
            within=NonNegativeReals,
            doc="Scaling factor for override slack variables",
        )

        # pylint: disable=undefined-variable
        self.cluster = ClusterBlock(self.set_clusters, override_data=override_data)

        # Add total budget constraint
        self.total_budget_constraint = Constraint(
            expr=(
                self.total_budget
                - sum(self.cluster[c].plugging_cost for c in self.set_clusters)
                == self.unused_budget - self.excess_budget
            ),
            doc="Total cost of plugging must be within the total budget",
        )
        if model_inputs.config.objective_weight_impact < 100:
            model_inputs.compute_efficiency_scaling_factors()
            for c in self.set_clusters:
                self.cluster[c].efficiency_model = EfficiencyBlock(
                    formulation_type=model_inputs.config.efficiency_formulation
                )

        wd = model_inputs.config.well_data
        self.set_wells = Set(initialize=wd.data.index.tolist())

        self.well_selected = Var(
            self.set_wells,
            within=Binary,
            doc="If well is selected in any project",
        )

        @self.Constraint(self.set_wells)
        def well_selection_constraint(self, w):
            return self.well_selected[w] == sum(
                self.cluster[c].select_well[w]
                for c in self.set_clusters
                if w in self.cluster[c].set_wells
            )

        if model_inputs.config.cluster_method == "Exhaustive":

            @self.Constraint(self.set_clusters)
            def anchor_well_constraint(self, c):
                return self.cluster[c].select_cluster == self.cluster[c].select_well[c]
                # the index c works for select_well because in exhaustive clustering
                # the well index is the same as the cluster number

        # Add optional constraints:
        if model_inputs.config.threshold_distance is not None:
            if model_inputs.config.cluster_method == "Louvain":
                for c in self.set_clusters:
                    self.cluster[c].add_distant_well_cuts()

        if model_inputs.config.max_wells_per_owner is not None:
            self.add_owner_well_count()

        scaling_factor_budget, scaling_factor_override, budget_sufficient = (
            self._slack_variable_scaling()
        )
        if model_inputs.config.min_budget_usage is not None:
            if budget_sufficient:
                LOGGER.warning(
                    "Ignoring min_budget_usage as the total_budget is sufficient to plug all wells."
                )
            else:
                self.add_min_budget_usage()

        if model_inputs.config.penalize_unused_budget:
            self.unused_budget_scaling = scaling_factor_budget

        # Add variables and constraints related to override re-optimization only
        if self.override_data and self.override_data.override_status:
            if model_inputs.config.cluster_method == "Exhaustive":
                self.add_well_pair_constraint()
                self.add_well_preserve_constraint()
                self.fix_var()
            else:
                self.fix_var()

            self.excess_budget_scaling = 100 * scaling_factor_budget
            self.override_slack_scaling = scaling_factor_override

            if model_inputs.config.threshold_distance is not None:
                self.excess_distant_pairs = Var(
                    within=NonNegativeReals,
                    doc="Number of well pairs exceeding the distance threshold",
                )
                self.add_distant_well_count()

        # Append the objective function
        self.append_objective()

    def add_owner_well_count(self):
        """
        Constrains the maximum number of wells belonging to a specific owner
        chosen for plugging.
        """
        max_owc = self.model_inputs.config.max_wells_per_owner
        owner_dict = self.model_inputs.owner_well_count

        @self.Constraint(
            owner_dict.keys(),
            doc="Limit number of wells belonging to each owner",
        )
        def max_well_owner_constraint(b, owner):
            return (
                sum(b.well_selected[w] for w in owner_dict[owner])
                <= max_owc + self.excess_owc[owner]
            )

    def add_distant_well_count(self):
        """
        Count the number of well pairs that exceed the distance threshold
        for clusters that had wells reassigned during override
        re-optimization.
        """

        self.set_clusters_reassign = Set(
            initialize=list(self.override_data.reassign.keys()),
            doc="Clusters that have wells reassigned",
        )

        self.num_distant_well_pairs = Var(
            self.set_clusters_reassign,
            within=NonNegativeReals,
            doc="Number of distant well pairs for each reassigned cluster",
        )

        reassign_dict = self.override_data.reassign

        @self.Constraint(self.set_clusters_reassign)
        def count_distant_wells(b, c):
            reassign_well_list = reassign_dict[c]
            params = b.model_inputs
            wd = params.config.well_data
            wells_in_cluster = params.campaign_candidates[c]

            pairwise_distance = distance_reassigned_well(
                wells_in_cluster, reassign_well_list, wd
            )

            # Filter well pairs that exceed the threshold
            well_pairs_excess = [
                (w1, w2)
                for (w1, w2), dist in pairwise_distance.items()
                if dist > b.model_inputs.config.threshold_distance
            ]

            # Attach filtered well pairs as the set_well_pairs_remove set of the block
            b.cluster[c].set_well_pairs_remove.set_value(well_pairs_excess)

            # Add constraint for skipping distant pairs
            b.cluster[c].add_distant_well_cuts()

            # Constraint: Count the number of distant pairs with both wells selected
            return b.num_distant_well_pairs[c] == sum(
                b.cluster[c].is_distant_pair[w1, w2]
                for (w1, w2) in b.cluster[c].set_well_pairs_remove
            )

        @self.Constraint()
        def total_excess_distant_wells(b):
            return self.excess_distant_pairs == sum(
                b.num_distant_well_pairs[c] for c in b.set_clusters_reassign
            )

    def _obtain_well_preserved(self):
        """
        Obtain wells must be selected in the re-optimization results - wells being
        added, locked, and reassigned, as well as the corresponding destination cluster.
        """
        well_fix_dict = self.override_data.re_optimize_well_dict
        well_fix_dict_updated = {}

        # Obtaining well lock information in each modified cluster
        for cluster, well_dict in well_fix_dict.items():
            well_fix_list = []
            for well in well_dict.keys():
                if well_dict[well] == 1:
                    well_fix_list.append(well)
            well_fix_dict_updated[cluster] = well_fix_list
        return well_fix_dict_updated

    def add_well_pair_constraint(self):
        """
        Constrain the wells being locked to be selected in the same cluster.
        """
        # Generate a list of tuples for each pair of wells in each cluster in
        # well_fix_dict_updated
        well_fix_dict_updated = self._obtain_well_preserved()
        lock_pairs = []
        for _, well_fix_list in well_fix_dict_updated.items():
            for i, w1 in enumerate(well_fix_list):
                for w2 in well_fix_list[i + 1 :]:
                    lock_pairs.append((w1, w2))

        # Apply constraints to ensure that fixed well pairs in a cluster
        # are selected together
        self.set_well_pairs_lock = Set(
            dimen=2,
            initialize=lock_pairs,
            doc="Well pairs should appear in the same project",
        )

        @self.Constraint(self.set_clusters, self.set_well_pairs_lock)
        def lock_reassign_well_constraint(self, c, w1, w2):
            if w1 in self.cluster[c].set_wells and w2 in self.cluster[c].set_wells:
                return (
                    self.cluster[c].select_well[w1] == self.cluster[c].select_well[w2]
                )
            return Constraint.Skip

    def add_well_preserve_constraint(self):
        """
        Constrain the wells being added, locked, and reassigned must be selected.
        """
        # Generate a list of wells that need to be preserved in the re-optimization
        # results
        well_preserve_dict_updated = self._obtain_well_preserved()
        well_preserve_list = [
            w for well_list in well_preserve_dict_updated.values() for w in well_list
        ]

        # Apply constraints to ensure that locked and reassigned wells in a cluster
        # are selected together
        self.set_well_preserve = Set(
            dimen=1,
            initialize=well_preserve_list,
            doc="Wells must appear in the recommendation",
        )

        @self.Constraint(self.set_well_preserve)
        def preserve_well_constraint(self, w):
            return (
                sum(
                    self.cluster[c].select_well[w]
                    for c in self.set_clusters
                    if w in self.cluster[c].set_wells
                )
                == 1
            )

    def fix_var(self):
        """
        identify clusters and/or the wells with in the cluster that will be fixed
        based on the override selection

        Parameters
        ----------
        override_data : ReOptimizationData
            A ReOptimizationData object containing the necessary information
            related to the re-optimization of override
        """

        # obtain the dictionary for clusters being fixed and wells being fixed
        cluster_fix_dict = self.override_data.re_optimize_cluster_dict
        well_fix_dict = self.override_data.re_optimize_well_dict

        # Under Exhaustive clustering, update well_fix_dict to only fix wells that are removed
        # or removed from their original cluster and not in unfix_cluster_list
        if self.model_inputs.config.cluster_method == "Exhaustive":
            # A list of qualified clusters from Step 3-2-2 of update_cluster_exhaustive
            unfix_cluster_list = set(self.model_inputs.unfix_cluster_list or [])
            well_fix_dict = {
                cluster: {
                    well: status
                    for well, status in well_status_dict.items()
                    if status == 0 and well not in unfix_cluster_list
                }
                for cluster, well_status_dict in well_fix_dict.items()
            }

            cluster_fix_dict = {
                cluster: status
                for cluster, status in cluster_fix_dict.items()
                if status == 0
            }

            # Remove clusters with empty dicts
            well_fix_dict = {
                cluster: well_status_dict
                for cluster, well_status_dict in well_fix_dict.items()
                if well_status_dict
            }

        # Combine keys (cluster) of cluster_fix_dict and well_fix_dict and remove
        # duplicates using a set
        unique_clusters = set(cluster_fix_dict.keys()).union(set(well_fix_dict.keys()))
        unique_clusters_list = list(unique_clusters)

        # Obtain information on whether a cluster is fixed and if any wells in the cluster
        # are fixed.
        # The fix() method applies the fixes to the cluster and wells independently.
        # Iterate over all clusters in the unique_clusters_list
        for c in unique_clusters_list:
            # Get the binary variable for the cluster selected to be fixed. If only
            # specific wells in the cluster are fixed, assign None to the cluster.
            if c in cluster_fix_dict:
                cluster_v = cluster_fix_dict[c]
            else:
                cluster_v = None

            # Get the binary variable for the wells selected to be fixed. For a fixed cluster,
            # if no specific wells are selected, assign None to the wells.
            if c in well_fix_dict:
                wells_v = well_fix_dict[c]
            else:
                wells_v = None

            self.cluster[c].fix(cluster_v, wells_v)

    def add_min_budget_usage(self):
        """
        Implements an upper bound on the unused budget to ensure that at
        least the specified percentage of the budget is utilized.
        """

        max_unused_budget = (
            1 - self.model_inputs.config.min_budget_usage / 100
        ) * self.total_budget

        # Define upper bound for the budget amount that is not utilized
        # pylint: disable=no-member
        self.unused_budget.setub(max_unused_budget)

    def append_objective(self):
        """
        Appends objective function to the model
        """
        total_impact_score = sum(
            self.cluster[c].cluster_impact_score for c in self.set_clusters
        )

        # Compute the total cluster efficiency score (if applicable)
        total_efficiency_score = sum(
            (
                self.cluster[c].efficiency_model.cluster_efficiency_score
                if hasattr(self.cluster[c], "efficiency_model")
                else 0
            )
            for c in self.set_clusters
        )

        # Compute the total penalty for constraints being violated when
        # performing the re-optimization of override
        self.excess_campaign_length = Expression(
            expr=sum(
                self.cluster[c].max_well_violation + self.cluster[c].min_well_violation
                for c in self.set_clusters
            )
        )

        self.excess_campaign_cost = Expression(
            expr=sum(
                self.cluster[c].min_cost_project_violation
                + self.cluster[c].max_cost_project_violation
                for c in self.set_clusters
            )
        )

        # Set all re-optimization-related variables to zero when solving the
        # base optimization problem or when no override selections are provided.
        if not self.override_data or not self.override_data.override_status:
            self.excess_budget.fix(0)  # pylint: disable=no-member
            for owc_value in self.excess_owc.values():
                owc_value.fix(0)  # pylint: disable=no-member
            self.excess_campaign_length_constraint = Constraint(
                expr=(self.excess_campaign_length == 0),
                doc="Ensure the minimum and maximum number of wells in a project constraint "
                "is satisfied under the non-override scenario.",
            )
            self.total_constraint_penalty = 0
        else:
            self.budget_status = Var(
                within=Binary,
                doc="1 if there is excess budget; 0 if there is unused budget",
            )

            self.excess_budget_constraint = Constraint(
                expr=(self.excess_budget <= self.budget_status * self.total_budget),
                doc="Ensure excess_budget and unused_budget are not non-zero at the same time.",
            )

            self.unused_budget_constraint = Constraint(
                expr=(
                    self.unused_budget <= (1 - self.budget_status) * self.total_budget
                ),
                doc="Ensure excess_budget and unused_budget are not non-zero at the same time.",
            )
            self.total_constraint_penalty = Expression(
                expr=self.excess_budget * self.excess_budget_scaling
                + self.excess_campaign_cost * self.excess_budget_scaling
                + self.override_slack_scaling
                * (
                    sum(
                        self.excess_owc[owner]
                        for owner in self.model_inputs.owner_well_count.keys()
                    )
                    + (
                        self.excess_distant_pairs
                        if hasattr(self, "excess_distant_pairs")
                        else 0
                    )
                    + self.excess_campaign_length
                )
            )

        # Define the total priority score as the objective
        self.total_priority_score = Objective(
            expr=(
                total_impact_score
                + total_efficiency_score
                - self.unused_budget_scaling * self.unused_budget
                - self.total_constraint_penalty
            ),
            sense=maximize,
            doc=(
                "Total impact and efficiency score minus scaled slack "
                "variables for unutilized budget and penalties for constraint violations."
            ),
        )

    def _slack_variable_scaling(self):
        """
        Check whether the budget is sufficient to plug all wells and (1)
        calculate the scaling factor for the budget slack variable based on the
        corresponding scenario. (2) Calculate the scaling factor for the slack
        variables of constraints violated during re-optimization of override.
        """
        # estimate the maximum number of wells can be plugged with the budget.
        wd = self.model_inputs.config.well_data
        if wd.col_names.cost_of_plugging is None:
            unit_cost = max(self.model_inputs.get_mobilization_cost.values()) / max(
                self.model_inputs.get_mobilization_cost
            )
            max_well_num = self.model_inputs.get_total_budget / unit_cost
        else:
            upc = self.model_inputs.get_plugging_cost
            sorted_upc = upc.sort_values().reset_index(drop=True)
            cumulative_cost = sorted_upc.cumsum()
            max_well_num = (cumulative_cost <= self.model_inputs.get_total_budget).sum()

        # calculate the scaling factor for the budget slack variable if the
        # budget is not sufficient to plug all wells
        if max_well_num < len(self.model_inputs.config.well_data):
            scaling_budget_slack = (
                max_well_num
                * max(self.model_inputs.config.well_data["Priority Score [0-100]"])
            ) / self.model_inputs.get_total_budget
            budget_sufficient = False

            # scale the penalty for violated constraints to be 100 times greater
            # than the possible maximum total priority score across all wells.
            scaling_override_slack = max_well_num * max(
                self.model_inputs.config.well_data["Priority Score [0-100]"]
            )

        # calculate the scaling factor for the budget slack variable if the
        # budget is sufficient to plug all wells
        else:
            scaling_budget_slack = (
                len(self.model_inputs.config.well_data)
                * max(self.model_inputs.config.well_data["Priority Score [0-100]"])
            ) / self.model_inputs.get_total_budget
            budget_sufficient = True

            # scale the penalty for violated constraints to be 100 times greater
            # than the possible maximum total priority score across all wells.
            scaling_override_slack = len(self.model_inputs.config.well_data) * max(
                self.model_inputs.config.well_data["Priority Score [0-100]"]
            )
            budget_sufficient = True

        return scaling_budget_slack, scaling_override_slack, budget_sufficient

    def get_optimal_campaign(self):
        """
        Extracts the optimal choice of wells from the solved model
        """
        optimal_campaign = {}
        plugging_cost = {}
        efficiency_scores_projects = {}

        for c in self.set_clusters:
            blk = self.cluster[c]
            if blk.select_cluster.value < 0.05:
                # Cluster c is not chosen, so continue
                continue

            # Wells in cluster c are chosen
            optimal_campaign[c] = []
            plugging_cost[c] = blk.plugging_cost.value
            for w in blk.set_wells:
                if blk.select_well[w].value > 0.95:
                    # Well w is chosen, so store it in the dict
                    optimal_campaign[c].append(w)

            if hasattr(blk, "efficiency_model"):
                efficiency_scores_projects[c] = (
                    blk.efficiency_model.get_efficiency_scores()
                )

        wd = self.model_inputs.config.well_data
        return Campaign(
            wd,
            optimal_campaign,
            plugging_cost,
            self.model_inputs,
            efficiency_scores_projects,
        )

    def get_solution_pool(self, solver):
        """
        Extracts solutions from the solution pool

        Parameters
        ----------
        solver : Pyomo solver object
        """
        pm = self  # This is the Pyomo model
        # pylint: disable=protected-access
        gm = solver._solver_model  # This is the Gurobipy model
        # Get Pyomo var to Gurobipy var map.
        # Gurobi vars can be accessed as pm_to_gm[<pyomo var>]
        pm_to_gm = solver._pyomo_var_to_solver_var_map

        # Number of solutions found
        num_solutions = gm.SolCount
        solution_pool = {}
        # Well data
        # wd = self.model_inputs.config.well_data

        for i in range(num_solutions):
            gm.Params.SolutionNumber = i

            optimal_campaign = {}
            plugging_cost = {}

            for c in pm.set_clusters:
                blk = pm.cluster[c]
                if pm_to_gm[blk.select_cluster].Xn < 0.05:
                    # Cluster c is not chosen, so continue
                    continue

                # Wells in cluster c are chosen
                optimal_campaign[c] = []
                plugging_cost[c] = pm_to_gm[blk.plugging_cost].Xn
                for w in blk.set_wells:
                    if pm_to_gm[blk.select_well[w]].Xn > 0.95:
                        # Well w is chosen, so store it in the dict
                        optimal_campaign[c].append(w)

            solution_pool[i + 1] = Campaign(
                wd=self.model_inputs.config.well_data,
                clusters_dict=optimal_campaign,
                plugging_cost=plugging_cost,
                opt_model_inputs=self.model_inputs,
            )

        return solution_pool
