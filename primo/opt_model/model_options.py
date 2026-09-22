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
# pylint: disable=too-many-lines

# Standard libs
import copy
import logging

# Installed libs
import numpy as np
from pyomo.common.config import (
    Bool,
    ConfigDict,
    ConfigValue,
    In,
    IsInstance,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    document_kwargs_from_configdict,
)

# User-defined libs
from primo.data_parser.default_data import (
    DEFAULT_MAX_NUM_UNIQUE_OWNERS,
    DEFAULT_MAX_NUM_WELLS,
    WELL_BASED_METRICS,
    WELL_PAIR_METRICS,
)
from primo.data_parser.input_config import ScenarioType
from primo.data_parser.well_data import WellData
from primo.opt_model.model_with_clustering import PluggingCampaignModel
from primo.utils.clustering_utils import (
    check_existing_cluster,
    get_pairwise_metrics,
    perform_agglomerative_clustering,
    perform_boundary_clustering,
    perform_exhaustive_clustering,
    perform_louvain_clustering,
)
from primo.utils.config_utils import OverrideRemoveLockInfo, OverrideSelections
from primo.utils.constrained_clustering_utils import (
    perform_heuristic_constrained_clustering,
)
from primo.utils.domain_validators import InRange, validate_mobilization_cost
from primo.utils.raise_exception import raise_exception
from primo.utils.solvers import get_solver

LOGGER = logging.getLogger(__name__)


def model_config() -> ConfigDict:
    """
    Returns a Pyomo ConfigDict object that includes all user options
    associated with optimization modeling
    """
    # Container for storing and performing domain validation
    # of the inputs of the optimization model.
    # ConfigValue automatically performs domain validation.
    config = ConfigDict()

    # Essential inputs for the optimization model

    config.declare(
        "well_data",
        ConfigValue(
            domain=IsInstance(WellData),
            doc="WellData object containing the entire dataset",
        ),
    )
    config.declare(
        "total_budget",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Total budget for plugging [in USD]",
        ),
    )
    config.declare(
        "mobilization_cost",
        ConfigValue(
            domain=validate_mobilization_cost,
            doc="Cost of plugging wells [in USD]",
        ),
    )

    config.declare(
        "shallow_gas_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single shallow gas well in $",
        ),
    )

    config.declare(
        "deep_gas_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single deep gas well in $",
        ),
    )

    config.declare(
        "shallow_oil_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single shallow oil well in $",
        ),
    )

    config.declare(
        "deep_oil_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single deep oil well in $",
        ),
    )

    config.declare(
        "beta",
        ConfigValue(
            domain=InRange(0, 1),
            doc="Parameter that defines economies of scale "
            "when plugging multiple wells",
        ),
    )

    # Model type and model nature options
    config.declare(
        "efficiency_formulation",
        ConfigValue(
            default="Max Scaling",
            domain=In(["Max Scaling", "Zone"]),
            doc="Efficiency Formulation",
        ),
    )
    config.declare(
        "objective_weight_impact",
        ConfigValue(
            default=50,
            domain=InRange(0, 100),
            doc="Weight associated with Impact in the objective function",
        ),
    )
    config.declare(
        "model_nature",
        ConfigValue(
            default="linear",
            domain=In(["linear", "quadratic", "aggregated_linear"]),
            doc="Nature of the optimization model: MILP or MIQCQP",
        ),
    )

    # Parameters for optional constraints
    config.declare(
        "threshold_distance",
        ConfigValue(
            default=10.0,
            domain=NonNegativeFloat,
            doc="Maximum distance [in miles] allowed between wells",
        ),
    )
    config.declare(
        "max_wells_per_owner",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells per owner",
        ),
    )
    config.declare(
        "max_cost_project",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum cost per project [in USD]",
        ),
    )
    config.declare(
        "min_cost_project",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Minimum cost per project [in USD]",
        ),
    )
    config.declare(
        "max_num_projects",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of projects admissible in a campaign",
        ),
    )

    config.declare(
        "min_wells_in_project",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Minimum number of wells required in a project",
        ),
    )

    config.declare(
        "max_wells_in_project",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells allowed in a project",
        ),
    )
    config.declare(
        "cluster_method",
        ConfigValue(
            default="Agglomerative",
            domain=In(
                [
                    "Agglomerative",
                    "Louvain",
                    "Exhaustive",
                    "Boundary",
                    "Constrained_Heuristic",
                ]
            ),
            doc="Method used for clustering the wells",
        ),
    )
    config.declare(
        "subcluster_method",
        ConfigValue(
            default=None,
            domain=In([None, "Agglomerative", "Louvain"]),
            doc="Method used for sub-clustering the wells within a geographic boundary",
        ),
    )
    config.declare(
        "max_boundary_cluster_size",
        ConfigValue(
            default=200,
            domain=NonNegativeInt,
            doc="Maximum size of clusters for boundary clustering",
        ),
    )
    config.declare(
        "threshold_cluster_size",
        ConfigValue(
            default=300,
            domain=NonNegativeInt,
            doc="Maximum size of clusters for Louvain clustering",
        ),
    )
    config.declare(
        "num_nearest_neighbors",
        ConfigValue(
            default=10,
            domain=NonNegativeInt,
            doc=(
                "Number of nearest neighbors to consider adding edges to "
                "while constructing the graph for Louvain clustering"
            ),
        ),
    )
    config.declare(
        "max_resolution",
        ConfigValue(
            default=10,
            domain=NonNegativeFloat,
            doc="Maximum resolution parameter value for Louvain clustering",
        ),
    )
    config.declare(
        "min_budget_usage",
        ConfigValue(
            default=None,
            domain=InRange(0, 100),
            doc="Minimum percentage of the total budget to be used for plugging",
        ),
    )
    config.declare(
        "penalize_unused_budget",
        ConfigValue(
            default=False,
            domain=Bool,
            doc=(
                "If True, unused budget will be penalized in the objective function\n"
                "with suitably chosen weight factor"
            ),
        ),
    )

    # Parameters for heuristic constrained clustering
    config.declare(
        "constrained_clustering_objective",
        ConfigValue(
            default="center",
            domain=In(["maximum", "minimum", "center", "target"]),
            doc="Objective for heuristic constrained clustering",
        ),
    )
    config.declare(
        "target_number_of_clusters",
        ConfigValue(
            default=None,
            domain=PositiveInt,
            doc="Target number of clusters for heuristic constrained clustering",
        ),
    )
    config.declare(
        "uniformity_parameter",
        ConfigValue(
            default=0,
            domain=InRange(0, 1),
            doc="Parameter controlling the uniformity of cluster sizes",
        ),
    )
    config.declare(
        "constrained_clustering_max_iterations",
        ConfigValue(
            default=10,
            domain=NonNegativeInt,
            doc="Maximum number of iterations for heuristic constrained clustering",
        ),
    )

    # Parameters for computing efficiency metrics

    config.declare(
        "max_num_wells",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells selected in a project",
        ),
    )
    config.declare(
        "max_dist_to_road",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum distance to road allowed for selected wells",
        ),
    )
    config.declare(
        "max_elevation_delta",
        ConfigValue(
            domain=NonNegativeFloat,
            doc=(
                "Maximum elevation delta from the closest road "
                "point allowed for selected wells"
            ),
        ),
    )
    config.declare(
        "max_population_density",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum population density allowed to have near a well",
        ),
    )
    config.declare(
        "max_record_completeness",
        ConfigValue(
            default=1.0,
            domain=NonNegativeFloat,
            doc="Maximum record completeness of a well",
        ),
    )
    config.declare(
        "max_num_unique_owners",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of unique owners allowed in a project",
        ),
    )
    config.declare(
        "max_dist_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum distance [in miles] allowed between wells",
        ),
    )
    config.declare(
        "max_age_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum age range allowed in a project",
        ),
    )
    config.declare(
        "max_depth_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum depth range allowed in a project",
        ),
    )

    config.declare(
        "embedding_b",
        ConfigValue(
            domain=NonNegativeFloat,
            default=0.0,
            doc="Parameter 'b' for the efficiency embedding function",
        ),
    )

    config.declare(
        "discount_cost_factor",
        ConfigValue(
            domain=InRange(0, 1),
            default=1,
            doc="Beta value to be used for economies of scale",
        ),
    )

    config.declare(
        "cost_formulation_type",
        ConfigValue(
            domain=In(["standard_bigm", "alternate_bigm", "hull"]),
            default="hull",
            doc="Alternate formulation to incorporate economies of scale",
        ),
    )

    return config


# pylint: disable = too-many-statements
class OptModelInputs:  # pylint: disable=too-many-instance-attributes
    """
    Assembles all the necessary inputs for the optimization model.
    """

    # Using ConfigDict from Pyomo for domain validation.
    CONFIG = model_config()

    # pylint: disable = too-many-arguments, too-many-positional-arguments
    @document_kwargs_from_configdict(CONFIG)
    def __init__(
        self,
        well_data: WellData,
        total_budget: float = None,
        mobilization_cost: dict = None,
        cluster_mapping=None,
        scenario_type=None,
        **kwargs,
    ):
        # pylint: disable=too-many-branches
        # Update the values of all the inputs
        # ConfigDict handles KeyError, other input errors, and domain errors
        LOGGER.info("Processing optimization model inputs.")
        kwargs.update(
            {
                "well_data": well_data,
                "total_budget": total_budget,
                "mobilization_cost": mobilization_cost,
            }
        )
        self.config = self.CONFIG(kwargs)
        self.scenario_type = scenario_type or ScenarioType()

        # Raise an error if the essential inputs are not provided
        wd = self.config.well_data
        if self.scenario_type.project_recommendation:
            if mobilization_cost is None and wd.col_names.cost_of_plugging is None:
                msg = "Mobilization cost is an essential input for the optimization model."
                raise_exception(msg, ValueError)
            if total_budget is None:
                msg = "Total budget is an essential input for the optimization model."
                raise_exception(msg, ValueError)

        # Raise an error if priority scores are not calculated.
        if not hasattr(wd.column_names, "priority_score"):
            msg = (
                "Unable to find priority scores in the WellData object. Compute the scores "
                "using the compute_priority_scores method."
            )
            raise_exception(msg, ValueError)

        if (
            self.scenario_type.project_recommendation
            or self.scenario_type.project_comparison
        ):
            if self.config.objective_weight_impact < 100:
                if wd.config.efficiency_metrics is None:
                    raise_exception(
                        "Weight of efficiency is non-zero."
                        "Efficiency metrics object is not specified.",
                        ValueError,
                    )
                if (
                    wd.config.efficiency_metrics.record_completeness.effective_weight
                    > 0
                ):
                    self._compute_record_incompleteness(wd)

        if cluster_mapping is None:
            LOGGER.info("Clustering Data in OptModelInputs")
            # Construct campaign candidates
            # Step 1: Perform clustering, Should distance_threshold be a user argument?
            # Structure: {cluster_1: [index_1, index_2,..], cluster_2: [], ...}
            if self.config.cluster_method == "Agglomerative":
                LOGGER.info(
                    "Clustering wells using the Agglomerative clustering method."
                )
                self.campaign_candidates = perform_agglomerative_clustering(
                    wd, threshold_distance=self.config.threshold_distance
                )
            elif self.config.cluster_method == "Exhaustive":
                LOGGER.info("Clustering wells using the exhaustive clustering method.")
                self.campaign_candidates = perform_exhaustive_clustering(
                    wd, threshold_distance=self.config.threshold_distance
                )
                self.unfix_cluster_list = []
            elif self.config.cluster_method == "Louvain":
                LOGGER.info("Clustering wells using the Louvain clustering method.")
                self.campaign_candidates = perform_louvain_clustering(
                    wd,
                    threshold_distance=self.config.threshold_distance,
                    threshold_cluster_size=self.config.threshold_cluster_size,
                    nearest_neighbors=self.config.num_nearest_neighbors,
                    max_resolution=self.config.max_resolution,
                )
            elif self.config.cluster_method == "Boundary":
                LOGGER.info("Clustering wells by geographic boundary")
                self.campaign_candidates = perform_boundary_clustering(
                    wd,
                    subcluster_method=self.config.subcluster_method,
                    max_boundary_cluster_size=self.config.max_boundary_cluster_size,
                    threshold_distance=self.config.threshold_distance,
                    threshold_cluster_size=self.config.threshold_cluster_size,
                    nearest_neighbors=self.config.num_nearest_neighbors,
                    max_resolution=self.config.max_resolution,
                )
            elif self.config.cluster_method == "Constrained_Heuristic":
                LOGGER.info(
                    "Clustering wells using the heuristic constrained clustering method."
                )
                self.campaign_candidates = perform_heuristic_constrained_clustering(
                    model_options=self,
                    objective=self.config.constrained_clustering_objective,
                    uniformity_parameter=self.config.uniformity_parameter,
                    max_iterations=self.config.constrained_clustering_max_iterations,
                    target_clusters=self.config.target_number_of_clusters,
                )

        else:
            LOGGER.info("Skipping clustering step in OptModelInputs")
            self.campaign_candidates = cluster_mapping
            well_cluster_map = {index: "" for index in wd}
            for cluster, wells in self.campaign_candidates.items():
                for well in wells:
                    well_cluster_map[well] = cluster

            assert "" not in well_cluster_map.values()
            wd.add_new_column_ordered(
                "cluster", "Clusters", list(well_cluster_map.values())
            )

        # Construct owner well count data
        col_names = wd.column_names
        if wd.config.verify_operator_name:
            operator_list = set(wd[col_names.operator_name])
            self.owner_well_count = {owner: [] for owner in operator_list}
            for well in wd:
                # {Owner 1: [i2, i3, i7, ...], ...}
                # Key => Owner name, value => [index]
                self.owner_well_count[
                    wd.data.loc[well, col_names.operator_name]
                ].append(well)

        # NOTE: Attributes _opt_model and _solver are defined in
        # build_optimization_model and solve_model methods, respectively.
        self.pairwise_metrics = {c: None for c in self.campaign_candidates}
        self._opt_model = None
        self._solver = None
        LOGGER.info("Finished processing optimization model inputs.")

    @property
    def get_total_budget(self):
        """Returns scaled total budget [in million USD]"""
        # Optimization model uses scaled total budget value to avoid numerical issues
        return self.config.total_budget / 1e6

    @property
    def get_mobilization_cost(self):
        """Returns scaled mobilization cost [in million USD]"""
        # Optimization model uses Scaled mobilization costs to avoid numerical issues
        return {
            num_wells: cost / 1e6
            for num_wells, cost in self.config.mobilization_cost.items()
        }

    @property
    def get_plugging_cost(self):
        """Returns scaled plugging cost [in million USD]"""
        # Optimization model uses Scaled plugging costs to avoid numerical issues
        wcn = self.config.well_data.column_names
        return self.config.well_data[wcn.cost_of_plugging] / 1e6

    @staticmethod
    def check_sufficient_budget_static(
        well_data,
        max_wells_in_project,
        mobilization_cost,
        total_budget,
    ):
        """
        Returns True if the budget is enough to plug all the wells.

        Parameters:
            well_data: WellData object
                The well data object (with .data as a DataFrame).
            max_wells_in_project: Int
                Maximum number of wells per project.
            mobilization_cost: Dict
                A dict mapping number of wells to mobilization cost.
            total_budget: Int
                Available budget.
        """
        num_wells = len(well_data)

        # check if budget is enough to plug all the wells
        # compare budget with the case if all wells are in a single project
        # NOTE: we may have false positives in this case and have added an additional warning
        # after the solve statement to verify if the check was flagged correctly
        if max_wells_in_project is None:
            return total_budget >= mobilization_cost[num_wells]

        # check the case where max wells in project is greater than the size of the dataset
        max_wells_in_project = min(max_wells_in_project, num_wells)

        # calculate cost of project in case max wells is specified
        # NOTE: this case will have false positives too just like the previous case
        # the check after solve statement should flag these false positives
        exp_num_projects = num_wells // max_wells_in_project
        excess_well = num_wells % max_wells_in_project
        mobilization_cost_excess = (
            0 if excess_well == 0 else mobilization_cost[excess_well]
        )
        cost = (
            exp_num_projects * mobilization_cost[max_wells_in_project]
            + mobilization_cost_excess
        )
        # return if budget exceeds the cost
        return total_budget >= cost

    @property
    def check_sufficient_budget(self):
        """Returns True if the budget is enough to plug all the wells."""
        wd = self.config.well_data
        wcn = wd.column_names
        if wcn.cost_of_plugging is not None:
            return sum(wd.data[wcn.cost_of_plugging]) <= self.config.total_budget
        return OptModelInputs.check_sufficient_budget_static(
            self.config.well_data,
            self.config.max_wells_in_project,
            self.config.mobilization_cost,
            self.config.total_budget,
        )

    @property
    def get_max_cost_project(self):
        """Returns scaled maximum cost of the project [in million USD]"""
        if self.config.max_cost_project is None:
            return None

        return self.config.max_cost_project / 1e6

    @property
    def get_min_cost_project(self):
        """Returns scaled minimum cost of the project [in million USD]"""
        if self.config.min_cost_project is None:
            return None

        return self.config.min_cost_project / 1e6

    @property
    def optimization_model(self):
        """Returns the Pyomo optimization model"""
        return self._opt_model

    @property
    def solver(self):
        """Returns the solver object"""
        return self._solver

    def build_optimization_model(self, override_data=None):
        """Builds the optimization model"""
        LOGGER.info("Checking budget")
        if (
            self.check_sufficient_budget
            and not np.isclose(self.config.objective_weight_impact, 0)
            and not self.config.objective_weight_impact == 100
        ):
            self.config.objective_weight_impact = 0
            LOGGER.info(
                "Setting weight of impact score in objective function to 0 "
                "since we have the budget to plug all wells in the dataset."
            )

        LOGGER.info("Beginning to construct the optimization model.")
        self._opt_model = PluggingCampaignModel(self, override_data=override_data)
        LOGGER.info("Completed the construction of the optimization model.")
        return self._opt_model

    def _get_optimal_campaign_with_resolve_on_efficiency_mismatch(
        self,
        solver_name,
        solve_kwargs,
        resolve_mip_gap,
    ):
        """
        Returns the optimal campaign and resolves the model with a tighter gap
        when campaign score validation fails after the original solve.
        """
        try:
            return self._opt_model.get_optimal_campaign()
        except RuntimeError as err:
            mismatch_msg = "Calculated and model efficiency scores did not match."
            if mismatch_msg not in str(err):
                raise_exception(str(err), RuntimeError)

        LOGGER.warning(
            "Efficiency score validation failed after solve. Fixing selected "
            "cluster and well decisions and re-solving with a tighter MIP gap."
        )
        for cluster_id in self._opt_model.set_clusters:
            blk = self._opt_model.cluster[cluster_id]
            if blk.select_cluster.value >= 0.95:
                blk.select_cluster.fix(1)
                for well in blk.set_wells:
                    blk.select_well[well].fix(int(blk.select_well[well].value >= 0.95))
            else:
                blk.select_cluster.fix(0)

        retry_kwargs = solve_kwargs.copy()
        retry_kwargs["solver"] = solver_name
        retry_kwargs["mip_gap"] = resolve_mip_gap
        self._solver = get_solver(**retry_kwargs)

        if solver_name == "gurobi_persistent":
            self._solver.set_instance(self._opt_model)

        self._solver.solve(self._opt_model, tee=retry_kwargs.get("stream_output", True))
        return self._opt_model.get_optimal_campaign()

    # pylint: disable=too-many-branches
    def solve_model(self, **kwargs):
        """Solves the optimization"""

        # Adding support for pool search if gurobi_persistent is available
        # To get n-best solutions, pass pool_search_mode = 2 and pool_size = n
        pool_search_mode = kwargs.pop("pool_search_mode", 0)
        pool_size = kwargs.pop("pool_size", 10)
        solve_kwargs = kwargs.copy()

        solver = get_solver(**kwargs)
        self._solver = solver

        # Name attribute is not defined for HiGHS. But it works for all
        # other supported solvers. So, set name as highs, if it does not exist
        solver_name = getattr(solver, "name", "highs")

        if solver_name == "gurobi_persistent":
            # For persistent solvers, model instance need to be set manually
            solver.set_instance(self._opt_model)
            solver.set_gurobi_param("PoolSearchMode", pool_search_mode)
            solver.set_gurobi_param("PoolSolutions", pool_size)

        # Solve the optimization problem
        solver.solve(self._opt_model, tee=kwargs.get("stream_output", True))

        # TODO: remove the addition of the cluster column after updating override
        campaign = self._get_optimal_campaign_with_resolve_on_efficiency_mismatch(
            solver_name,
            solve_kwargs,
            resolve_mip_gap=1e-4,
        )
        project = campaign.projects
        number_of_wells_in_projects = sum(
            len(proj.well_data.data) for proj in project.values()
        )

        if self.config.cluster_method == "Exhaustive":
            if self._opt_model.override_data:
                cluster_col = getattr(self.config.well_data.col_names, "cluster", None)
                if cluster_col:
                    del self.config.well_data.data[
                        self.config.well_data.col_names.cluster
                    ]
                    delattr(self.config.well_data.col_names, "cluster")

            # redefine cluster column
            if not check_existing_cluster(self.config.well_data):
                well_cluster_map = {index: "" for index in self.config.well_data}
                for _, project_object in project.items():
                    for well in project_object.well_data.data.index:
                        well_cluster_map[well] = project_object.project_id
                for well, cluster in well_cluster_map.items():
                    if cluster == "":
                        well_cluster_map[well] = well
                assert "" not in well_cluster_map.values()
                self.config.well_data.add_new_column_ordered(
                    "cluster", "Clusters", list(well_cluster_map.values())
                )

        # Check if budget is sufficient or not
        if self.check_sufficient_budget:
            if len(self.config.well_data.data) != number_of_wells_in_projects:
                LOGGER.warning(
                    "The sufficient budget check returned True but "
                    "some wells in the dataset are not selected in projects. "
                )
        else:
            if len(self.config.well_data.data) == number_of_wells_in_projects:
                LOGGER.warning(
                    "The sufficient budget check returned False but "
                    "all wells in the dataset are selected in projects. "
                )

        # Return the solution pool, if it is requested
        if solver_name == "gurobi_persistent" and pool_search_mode == 2:
            # Return the solution pool if pool_search_mode is active
            return self._opt_model.get_solution_pool(self._solver)

        # In all other cases, return the optimal campaign
        return campaign

    def update_cluster(
        self,
        override_selection: OverrideSelections,
    ):
        """
        Updates the campaign candidates by changing the cluster numbers for specific wells.

        Parameters
        ---------
        override_selection : OverrideSelections
            An OverrideSelections object which includes information on clusters and wells
            being modified in override
        """
        existing_clusters = override_selection.add_widget_return.existing_clusters
        new_clusters = override_selection.add_widget_return.new_clusters
        lock_widget_return = override_selection.lock_widget_return

        wd = self.config.well_data
        col_names = wd.column_names

        # Update the clusters when Exhaustive clustering is used
        if self.config.cluster_method == "Exhaustive":
            self.update_cluster_exhaustive(
                existing_clusters, new_clusters, lock_widget_return
            )

        # Update the clusters under the rest of the clustering method
        else:
            if existing_clusters != new_clusters:
                # Remove wells from existing clusters and update owner well counts
                for existing_cluster, existing_wells in existing_clusters.items():
                    for well in existing_wells:
                        self.campaign_candidates[existing_cluster].remove(well)

                # Add wells to new clusters and update the well data and owner well counts
                for new_cluster, wells in new_clusters.items():
                    for well in wells:
                        self.campaign_candidates[new_cluster].append(well)
                        self.config.well_data.data.loc[well, col_names.cluster] = (
                            new_cluster
                        )

            # Remove empty clusters after updating campaign_candidates
            self.campaign_candidates = {
                cluster: list(set(wells))
                for cluster, wells in self.campaign_candidates.items()
                if len(wells) > 0
            }

    @staticmethod
    def _compute_record_incompleteness(wd: WellData):
        """
        Computes the record incompleteness of the given data.
        Higher the score, more of the required data is not available
        for the well.
        Parameters
        ----------
        wd : WellData
            Object containing wells
        """
        data = wd.data[wd.get_flag_columns].sum(axis=1)
        num_columns = max(1, len(wd.get_flag_columns))
        wd.add_new_column_ordered(
            "record_completeness", "Fraction Data Incomplete", data / num_columns
        )

    def compute_efficiency_scaling_factors(self):
        # pylint: disable=too-many-branches
        """
        Checks whether scaling factors for efficiency metrics are provided by
        the user or not. If not, computes the scaling factors using the entire
        dataset.

        Parameters
        ----------
        self : OptModelInputs
            OptModelInputs object
        """
        LOGGER.info("Computing scaling factors for efficiency metrics")
        config = self.config
        wd = config.well_data
        eff_metrics = wd.config.efficiency_metrics
        eff_weights = eff_metrics.get_weights

        def set_scaling_factor(metric_name, scale_value):
            """Function for logging warning message"""
            LOGGER.warning(
                f"Scaling factor for {metric_name} metric is not "
                f"provided, so it is set to {scale_value}. \n\t To specify the "
                f"scaling factor, pass argument max_{metric_name} while instantiating "
                f"the OptModelInputs object."
            )
            setattr(config, "max_" + metric_name, scale_value)

        # Setting a scaling factor for num_wells metric
        if config.max_num_wells is None and eff_weights.num_wells > 0:
            if config.max_wells_in_project is not None:
                config.max_num_wells = config.max_wells_in_project
            else:
                set_scaling_factor("num_wells", DEFAULT_MAX_NUM_WELLS)

        # Setting a scaling factor for num_unique_owners metric
        if config.max_num_unique_owners is None and eff_weights.num_unique_owners > 0:
            if config.max_wells_in_project is not None:
                set_scaling_factor("num_unique_owners", config.max_wells_in_project)
            else:
                set_scaling_factor("num_unique_owners", DEFAULT_MAX_NUM_UNIQUE_OWNERS)

        for metric in WELL_BASED_METRICS:
            if (
                getattr(eff_weights, metric, 0) > 0
                and getattr(config, "max_" + metric) is None
            ):
                # Metric is chosen, but the scaling factor is not specified
                scale_value = wd[getattr(eff_metrics, metric).data_col_name].max()
                if np.isclose(scale_value, 0):
                    LOGGER.warning(
                        f"Scaling factor for {metric} is close to 0. Setting it to 1."
                    )
                    scale_value = 1
                set_scaling_factor(metric, scale_value)

        if sum(getattr(eff_weights, metric, 0) for metric in WELL_PAIR_METRICS) == 0:
            # None of the pairwise metrics are selected, so return
            return

        # Append the pairwise metrics to the model
        for c in self.campaign_candidates:
            self.pairwise_metrics[c] = get_pairwise_metrics(
                wd, self.campaign_candidates[c]
            )

        for metric in WELL_PAIR_METRICS:
            if (
                getattr(eff_weights, metric, 0) > 0
                and getattr(config, "max_" + metric) is None
            ):
                # Metric is chosen, but the scaling factor is not specified
                if metric in ["age_range", "depth_range"]:
                    scale_value = (
                        wd[getattr(eff_metrics, metric).data_col_name].max()
                        - wd[getattr(eff_metrics, metric).data_col_name].min()
                    )
                else:
                    scale_value = self.config.threshold_distance
                if np.isclose(scale_value, 0):
                    LOGGER.warning(
                        f"Scaling factor for {metric} is close to 0. Setting it to 1."
                    )
                    scale_value = 1
                set_scaling_factor(metric, scale_value)

    def has_efficiency_scaling_factors(self):
        """
        checks if all efficiency metrics have scaling factors

        Parameters
        ----------
        self : OptModelInputs
            OptModelInputs object

        Returns
        ----------
        Bool : False if scaling factor for any metric does not exist, True otherwise
        """
        config = self.config
        wd = config.well_data
        eff_metrics = wd.config.efficiency_metrics
        for metric in eff_metrics:
            if (
                metric.effective_weight > 0
                and getattr(config, "max_" + metric.name) is None
            ):
                return False
        return True

    # pylint: disable=too-many-branches, too-many-locals
    def update_cluster_exhaustive(
        self,
        existing_clusters: dict,
        new_clusters: dict,
        lock_widget_return: OverrideRemoveLockInfo,
    ):
        """
        Updates the campaign candidates by changing the cluster numbers of reassigned wells
        when the Exhaustive clustering method is used.

        Parameters
        ----------
        existing_clusters : Dict[int, List[int]]
            A dictionary mapping of reassigned well indices with their original clusters, where
            the key is the original cluster number and the value is a list of well indices in the
            cluster.

        new_clusters : Dict[int, List[int]]
            A dictionary mapping of reassigned well indices with their new clusters, where
            the key is the new cluster number that the wells will be reassigned to and the value is
            a list of well indices in the cluster.

        lock_widget_return : OverrideRemoveLockInfo
            An OverrideRemoveLockInfo object containing information about wells that should be
            locked. Those are wells whose clusters do not change.
        """

        cluster_wo_lock_well = {}
        reassigned_non_anchor_well = {}
        well_new_cluster_map = {}

        if existing_clusters != new_clusters:
            # Record clusters with reassigned wells where no wells in the cluster are locked
            for cluster, well_list in new_clusters.items():
                # Clusters with a single reassigned well.
                if (
                    cluster not in lock_widget_return.well.keys()
                    or not lock_widget_return.well[cluster]
                ):
                    cluster_wo_lock_well[cluster] = well_list

                # Map reassigned wells to their new clusters
                for well in well_list:
                    well_new_cluster_map[well] = cluster

            # Updates the campaign_candidates
            # Step 1: Remove reassigned wells from their current clusters
            # if they are not the cluster's anchor well
            wells_to_be_added = set(
                new_well for wells in new_clusters.values() for new_well in wells
            )
            for existing_cluster, existing_wells in existing_clusters.items():
                for well in existing_wells:
                    for cluster, well_list in self.campaign_candidates.items():
                        if well in well_list and well != cluster:
                            well_list.remove(well)

                    # Check if any well in the cluster, where the reassigned well performs
                    # as the anchor well, only appears in the current cluster
                    self._check_missing_well(well)

                    # Identify the reassigned non-anchor wells and their corresponding new clusters
                    if well != existing_cluster and well in wells_to_be_added:
                        reassigned_non_anchor_well[well] = well_new_cluster_map[well]

            # Remove all wells in the cluster where the reassigned non-anchor well performs as
            # the anchor well
            for well in reassigned_non_anchor_well:
                self.campaign_candidates[well] = [well]

        # Step 2: Add the reassigned well to clusters where all associated locked wells are present.
        # Locked wells are wells whose clusters do not change (they are not reassigned).
        if lock_widget_return.well:
            well_lock = copy.deepcopy(lock_widget_return.well)
            for new_cluster, wells in new_clusters.items():
                if new_cluster in lock_widget_return.well.keys():
                    well_lock_list = well_lock[new_cluster]
                    for well_list in self.campaign_candidates.values():
                        if set(well_lock_list).issubset(well_list):
                            well_list.extend(
                                [well for well in wells if well not in well_list]
                            )

        # Step 3: Update clusters when well(s) are reassigned to the cluster without
        # locking
        if cluster_wo_lock_well:
            self._handle_no_lock_cluster(cluster_wo_lock_well, existing_clusters)

        # Step 4: Update the cluster where the reassigned non-anchor well perform as
        # the anchor well, ensuring it matches the destination cluster to which the
        # non-anchor well has been reassigned
        if reassigned_non_anchor_well:
            for well, cluster in reassigned_non_anchor_well.items():
                updated_well_list = copy.deepcopy(self.campaign_candidates[cluster])
                self.campaign_candidates[well] = updated_well_list

    def _check_missing_well(self, well: int):
        """
        Check if all wells from the cluster (excluding the anchor well) appear in at least one
        other cluster and add a new cluster for wells that do not appear in any other cluster

        Parameters
        ----------
        well : int
            Index of anchor well.
        """
        check_well_list = [w for w in self.campaign_candidates[well] if w != well]

        # Check if all wells from the cluster (excluding the anchor well)
        # appear in at least one other cluster
        all_wells_in_clusters = set(
            w
            for cluster, well_list in self.campaign_candidates.items()
            if cluster != well
            for w in well_list
        )
        missing_wells = [
            check_well
            for check_well in check_well_list
            if check_well not in all_wells_in_clusters
        ]
        # Add a new cluster for wells that do not appear in any other cluster
        # to ensure all wells can be considered for selection
        if missing_wells:
            self.campaign_candidates[missing_wells[0]] = check_well_list

    def _handle_no_lock_cluster(
        self, cluster_wo_lock_well: dict, existing_clusters: dict
    ):
        """
        Update clusters when multiple wells are reassigned to the same cluster without
        locking

        Parameters
        ----------
        cluster_wo_lock_well : Dict[int, List[int]]
            Dictionary where the key is the cluster where well(s) reassigned to
            it but there is no well locked in the cluster. The corresponding value is a list of
            indices for the wells reassigned to the cluster.

        existing_clusters : Dict[int, List[int]]
            A dictionary mapping of reassigned well indices with their original clusters, where
            the key is the original cluster number and the value is a list of well indices in the
            cluster.
        """
        # Dictionary for clusters being updated and their updated well lists
        updated_well_cluster = {}

        for (
            cluster,
            reassign_well_list,
        ) in cluster_wo_lock_well.items():
            # Step 3-1: Add reassigned well(s) to its destination cluster
            self.campaign_candidates[cluster].extend(reassign_well_list)

            # Step 3-2: Assert whether the destination cluster contains any extra well(s)
            # Step 3-2-0: Obtain extra well(s) in the destination cluster
            skip_check_well_list = [*reassign_well_list]
            # Check if the anchor well in the destination cluster is reassigned, if yes,
            # any cluster that has the anchor well not be updated
            reassigned_wells = existing_clusters.get(cluster, [])
            skip_check_well_list = [*reassign_well_list, *reassigned_wells]

            current_well_list = self.campaign_candidates[cluster]

            extra_well = [
                well for well in current_well_list if well not in skip_check_well_list
            ]
            # Step 3-2-1: Add reassigned well(s) to clusters that contain any of the extra
            # well(s)
            if extra_well:
                self._propagate_extra_well(
                    extra_well,
                    updated_well_cluster,
                    reassign_well_list,
                )

            # Step 3-2-2: When no extra wells exist, update clusters where reassigned wells
            # are anchor wells to the list of reassigned wells to ensure the reassigned
            # wells remain able to be selected
            else:
                for well in reassign_well_list:
                    updated_well_cluster[well] = reassign_well_list

                # Record updated clusters for later use in fix_var to exclude wells from
                # the fix list
                self.unfix_cluster_list.extend(reassign_well_list)

        # Update the campaign_candidates
        if updated_well_cluster:
            for cluster, well_list in updated_well_cluster.items():
                self.campaign_candidates[cluster] = well_list

    def _propagate_extra_well(
        self, extra_well: list, updated_well_cluster: dict, reassign_well_list: list
    ):
        """
        Add reassigned well(s) to clusters that contain any of the extra well(s)

        Parameters
        ----------
        extra_well : List
            A list of wells in the destination cluster that are not reassigned wells.

        updated_well_cluster : Dict[int, List]
            Dictionary where the key is the destination cluster where multiple wells
            reassigned to it and the value is a updated list of well in the cluster.

        reassign_well_list : List
            A list of reassigned well in the destination cluster.
        """
        for cluster, well_list in self.campaign_candidates.items():
            if any(well in well_list for well in extra_well):
                if cluster not in updated_well_cluster:
                    updated_well_cluster[cluster] = copy.deepcopy(well_list)

                for well in reassign_well_list:
                    if well not in updated_well_cluster[cluster]:
                        updated_well_cluster[cluster].append(well)

        # Ensure that when multiple wells are reassigned to the same project, any
        # cluster whose corresponding well is an anchor well includes all reassigned wells
        if len(reassign_well_list) > 1:
            for well in reassign_well_list:
                check_list = self.campaign_candidates[well]
                if not set(reassign_well_list).issubset(check_list):
                    updated_well_cluster[well] = copy.deepcopy(
                        self.campaign_candidates[well]
                    )
                    for w in reassign_well_list:
                        if w not in updated_well_cluster[well]:
                            updated_well_cluster[well].append(w)
