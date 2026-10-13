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
from dataclasses import dataclass

# Installed libs
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils.clustering_utils import get_exhaustive_clusters
from primo.utils.constrained_clustering_utils import (
    min_project_cost_to_num_wells_mobilization_cost,
)
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
@dataclass
class MIPNonOverlappingClustering:
    """
    Clusters the given well population by solving a mixed integer programming model.

    Parameters
    ----------
    wd : WellData
        Object containing well population

    min_num_wells : int | None
        Minimum number of wells required in each cluster (equal to minimum
        number of wells allowed in a project), by default = None

    max_num_wells : int | None
        Maximum number of wells allowed in each cluster, by default = None

    max_num_wells_in_project : int | None
        Maximum number of wells allowed in a project, by default = None

    min_project_cost : float | None
        Minimum cost of a plugging project, by default = None

    max_project_cost : float | None
        Maximum cost of a plugging project, by default = None

    total_budget : float | None
        Total budget available for plugging operations, by default = None

    mobilization_cost : dict | None
        Dictionary containing mobilization cost, by default = None

    objective_type : str
        Choice of objective for clustering, default = "cluster_radius".
    """

    wd: WellData
    min_num_wells: int | None = None
    max_num_wells: int | None = None
    max_num_wells_in_project: int | None = None
    min_project_cost: float | None = None
    max_project_cost: float | None = None
    total_budget: float | None = None
    mobilization_cost: dict | None = None

    def __post_init__(self):
        self._supported_objective_types = [
            "cluster_radius",
            "num_clusters",
            "cluster_size",
        ]
        self._objective_type = "cluster_radius"
        self._model = None

    @property
    def objective_type(self):
        """Returns the choice of objective type"""
        return self._objective_type

    @objective_type.setter
    def objective_type(self, value: str):
        if value not in self._supported_objective_types:
            raise ValueError(f"Unrecognized value {value} for objective_type")

        self._objective_type = value

    def get_exhaustive_clustering(self, threshold_distance: float):
        """Performs exhaustive clustering of a given well population

        Parameters
        ----------
        threshold_distance : float
            Radius value from an anchor well

        Returns
        -------
        clusters : dict[int, list[int]]
            Dictionary with anchor wells as keys, and their neighboring wells as values

        distances : dict[int, dict[int, float]]
            Dictionary with anchor wells as keys, and square of distances to their
            neighboring wells as values
        """

        clusters, dists = get_exhaustive_clusters(
            self.wd, radius=threshold_distance, max_neighbors=self.max_num_wells
        )

        distances = {
            anchor_well: {well: dist**2 for well, dist in neighbors.items()}
            for anchor_well, neighbors in dists.items()
        }

        return clusters, distances

    def solve_model(self, model: pyo.ConcreteModel, **kwargs):
        """
        Solves the optimization model with Gurobi and returns the solution.
        Pass additional Gurobi options as keyword arguments.

        Parameters
        ----------
        model : pyo.ConcreteModel
            Optimization model
        """
        solver_name = kwargs.pop("solver_name", "gurobi_persistent")

        if solver_name == "gurobi_persistent":
            solver = pyo.SolverFactory("gurobi_persistent")
            solver.set_instance(model)

            for k, v in kwargs.items():
                if k != "tee":
                    solver.set_gurobi_param(k, v)

        elif solver_name == "scip":
            solver = pyo.SolverFactory("scip")
            for k, v in kwargs.items():
                if k != "tee":
                    solver.options[k] = v

        else:
            raise ValueError(f"Solver {solver_name} is not supported")

        result = solver.solve(model, tee=kwargs.get("tee", False))

        if result["Solver"][0]["Termination condition"] in [
            "infeasible",
            "infeasibleOrUnbounded",
        ]:
            LOGGER.warning("Clustering failed. The optimization model is infeasible.")
            return {}

        return self._retrieve_mip_solution(model)

    @staticmethod
    def _retrieve_mip_solution(model: pyo.ConcreteModel):
        """Retrieves solution of the MIP model

        Parameters
        ----------
        model : pyo.ConcreteModel
            Instance of the optimization model
        """
        clusters = model.exhaustive_clustering_data
        solution = {
            well: list(model.cluster[well].essential_wells_set)
            + [
                w
                for w in model.cluster[well].optional_wells_set
                if model.cluster[well].select_well[w].value > 0.95
            ]
            for well in clusters
            if model.cluster[well].select_cluster.value > 0.95
        }

        # NOTE: This may not be the best way to check solution correctness.
        # For now, this seems okay. More rigorous checks can be implemented later
        # based on the need.
        if sum(len(w) for w in solution.values()) < len(model.wells_set):
            raise RuntimeError(
                "One or more wells have not been assigned to any cluster."
            )

        return solution

    def check_costs_available(self):
        """Checks if either a full cost of plugging column or mobilization cost is provided."""
        if (
            self.mobilization_cost is not None
            or self.wd.column_names.cost_of_plugging in self.wd
        ):
            return

        raise_exception(
            "Well costs are required but have not been provided. Please pass a mobilization_cost "
            "dictionary or ensure plugging costs are included in WellData",
            ValueError,
        )

    def _get_min_num_wells_in_clusters(self, plug_cost: pd.Series | None):
        """
        Helper method for computing minimum number of wells required
        in a cluster of wells
        """
        min_wells_needed = 1 if self.min_num_wells is None else self.min_num_wells

        if self.min_project_cost is None:
            return min_wells_needed

        self.check_costs_available()
        if self.mobilization_cost is not None:
            min_wells_needed = max(
                min_wells_needed,
                min_project_cost_to_num_wells_mobilization_cost(
                    self.min_project_cost, self.mobilization_cost
                ),
            )

        else:
            min_wells_needed = max(
                min_wells_needed,
                plug_cost.cumsum().searchsorted(self.min_project_cost) + 1,
            )

        return min_wells_needed

    def _get_min_num_projects(self):
        """Helper method to compute the minimum number of projects needed"""
        if self.total_budget is None or (
            self.max_num_wells_in_project is None and self.max_project_cost is None
        ):
            # No restriction on number of clusters or projects in this case
            return None

        min_num_projects = 0

        if self.max_project_cost is not None:
            min_num_projects = max(
                min_num_projects, np.ceil(self.total_budget / self.max_project_cost)
            )

        if self.max_num_wells_in_project is not None:
            self.check_costs_available()
            if self.mobilization_cost is not None:
                min_num_projects = max(
                    min_num_projects,
                    np.ceil(
                        self.total_budget
                        / self.mobilization_cost[self.max_num_wells_in_project]
                    ),
                )
            else:
                plug_cost = self.wd[self.wd.column_names.cost_of_plugging]
                min_num_projects = max(
                    min_num_projects,
                    np.ceil(
                        self.total_budget
                        / (plug_cost.mean() * self.max_num_wells_in_project)
                    ),
                )

        return min_num_projects

    # pylint: disable=too-many-statements
    def build_model(
        self, threshold_distance: float, fix_num_clusters: int | None = None
    ):
        """Builds the mixed integer program for non-overlapping clustering.

        Parameters
        ----------
        threshold_distance : float
            Maximum distance allowed between a neighboring well and an anchor well.

        fix_num_clusters : int | None, optional
            Number of clusters desired in the solution. If it is not specified, the optimizer
            determines the value, by default None
        """

        clusters, distances = self.get_exhaustive_clustering(threshold_distance)

        m = pyo.ConcreteModel()
        m.exhaustive_clustering_data = clusters
        m.wells_set = pyo.Set(initialize=self.wd.data.index.to_list())
        m.clusters_set = pyo.Set(initialize=list(clusters.keys()))

        plug_cost = (
            None
            if self.wd.column_names.cost_of_plugging is None
            else self.wd[self.wd.column_names.cost_of_plugging]
        )

        @m.Block(m.clusters_set)
        def cluster(blk, c):
            # Compute minimum number of wells required in cluster c
            _min_num_wells = self._get_min_num_wells_in_clusters(
                None if plug_cost is None else plug_cost[clusters[c]]
            )

            blk.select_cluster = pyo.Var(within=pyo.Binary)
            if len(clusters[c]) < _min_num_wells:
                # This anchor well must not be selected as one of the clusters
                blk.select_cluster.fix(0)
                blk.essential_wells_set = pyo.Set(initialize=[c])
                blk.optional_wells_set = pyo.Set(initialize=[])
                blk.cluster_radius = pyo.Expression(expr=0 * blk.select_cluster)
                blk.cluster_size = pyo.Expression(expr=blk.select_cluster)
                return

            blk.essential_wells_set = pyo.Set(initialize=clusters[c][:_min_num_wells])
            blk.optional_wells_set = pyo.Set(initialize=clusters[c][_min_num_wells:])
            blk.select_well = pyo.Var(blk.optional_wells_set, within=pyo.Binary)

            @blk.Constraint(pyo.RangeSet(len(blk.optional_wells_set)))
            def ordering_of_selected_wells(b, idx):
                if idx == 1:
                    return (
                        b.select_well[blk.optional_wells_set.at(idx)]
                        <= b.select_cluster
                    )

                return (
                    b.select_well[blk.optional_wells_set.at(idx)]
                    <= b.select_well[blk.optional_wells_set.at(idx - 1)]
                )

            increments = {
                clusters[c][idx]: (
                    distances[c][clusters[c][idx]] - distances[c][clusters[c][idx - 1]]
                )
                for idx in range(1, len(clusters[c]))
            }
            _min_radius = distances[c][blk.essential_wells_set.at(-1)]
            blk.cluster_radius = pyo.Expression(
                expr=_min_radius * blk.select_cluster
                + sum(
                    increments[w] * blk.select_well[w] for w in blk.optional_wells_set
                )
            )

            increments = {
                w: (_min_num_wells + idx + 1) ** 2 - (_min_num_wells + idx) ** 2
                for idx, w in enumerate(blk.optional_wells_set)
            }
            blk.cluster_size = pyo.Expression(
                expr=_min_num_wells**2 * blk.select_cluster
                + sum(
                    increments[w] * blk.select_well[w] for w in blk.optional_wells_set
                )
            )

        @m.Constraint(m.wells_set)
        def assign_well_to_one_cluster(blk, w):
            return 1 == sum(
                (
                    blk.cluster[c].select_well[w]
                    if w in blk.cluster[c].optional_wells_set
                    else 0
                )
                + (
                    blk.cluster[c].select_cluster
                    if w in blk.cluster[c].essential_wells_set
                    else 0
                )
                for c in m.clusters_set
            )

        min_num_projects = self._get_min_num_projects()
        if min_num_projects is not None:
            m.ensure_enough_clusters_exist = pyo.Constraint(
                expr=sum(m.cluster[:].select_cluster) >= min_num_projects
            )

        if fix_num_clusters is not None:
            m.constrain_num_clusters = pyo.Constraint(
                expr=sum(m.cluster[:].select_cluster) == fix_num_clusters
            )

            if self.objective_type == "num_clusters":
                LOGGER.warning(
                    "Number of clusters is specified, and the objective_type is chosen"
                    "to be num_clusters. \n\t Changing the objective_type to cluster_radius"
                )
                self.objective_type = "cluster_radius"

        m.clustering_metrics = pyo.Expression(
            self._supported_objective_types,
            initialize={
                "num_clusters": sum(m.cluster[:].select_cluster),
                "cluster_radius": sum(m.cluster[:].cluster_radius),
                "cluster_size": sum(m.cluster[:].cluster_size),
            },
        )

        m.obj = pyo.Objective(expr=m.clustering_metrics[self.objective_type])
        LOGGER.info(f"Choosing {self.objective_type} as the objective function")
        return m

    def find_threshold_distance(
        self,
        start_value: float = 10.0,
        increment: float = 2.5,
        max_iterations: int = 10,
        num_clusters: int | None = None,
        **kwargs,
    ):
        """Finds the threshold distance for which a feasible clustering exists.
        Specify additional solver options as keyword arguments.

        Parameters
        ----------
        start_value : float, optional
            Starting value of threshold distance, by default 10.0

        increment : float, optional
            Increment in threshold distance if a feasible clustering is
            not found, by default 2.5

        max_iterations : int, optional
            Maximum number of iterations, by default 10

        num_clusters : int | None, optional
            Number of clusters in the solution, by default None

        Returns
        -------
        dict[int, list[int]]:
            The dictionary of clusters found

        float
            The final threshold distance used
        """
        feasible_clustering_found = False
        td = start_value
        iterations = 0

        while not feasible_clustering_found:
            LOGGER.info(
                f"Attempting to find minimum number of clusters with "
                f"a threshold_distance = {td} miles."
            )
            clusters = self.perform_clustering(td, num_clusters, **kwargs)

            if len(clusters) == 0:
                LOGGER.warning(
                    f"Failed to cluster with a threshold_distance = {td} miles."
                )
                td += increment
                iterations += 1
                if iterations >= max_iterations:
                    raise_exception(
                        f"Unable to find a feasible cluster after {max_iterations} iterations",
                        RuntimeError,
                    )

            else:
                LOGGER.info(
                    f"Found a feasible solution. Minimum threshold distance = {td} miles."
                    f"\n\t Solution has {len(clusters)} clusters."
                )
                feasible_clustering_found = True

        return clusters, td

    def perform_clustering(
        self, threshold_distance: float, num_clusters: int | None = None, **kwargs
    ):
        """
        Performs clustering for a given threshold distance. If desired, the number of
        clusters can also be specified. Specify additional solver options as
        keyword arguments.

        Parameters
        ----------
        threshold_distance : float
            The threshold distance to use for clustering

        num_clusters : int | None, optional
            Desired number of clusters in the solution, by default = None
        """
        self._model = self.build_model(threshold_distance, num_clusters)
        return self.solve_model(self._model, **kwargs)


@dataclass
class MIPOverlappingClustering(MIPNonOverlappingClustering):
    """Class for MIP-based clustering that allows overlaps"""

    @staticmethod
    def _retrieve_mip_solution(model: pyo.ConcreteModel):
        """Retrieves solution of the MIP model"""
        clusters = model.exhaustive_clustering_data
        return {
            well: nearby_wells
            for well, nearby_wells in clusters.items()
            if model.select_cluster[well].value > 0.95
        }

    def build_model(
        self, threshold_distance: float, fix_num_clusters: int | None = None
    ):
        """Builds the mixed integer program for overlapping clustering.

        Parameters
        ----------
        threshold_distance : float
            Maximum distance allowed between a neighboring well and an anchor well.

        fix_num_clusters : int | None, optional
            Number of clusters desired in the solution. If it is not specified, the optimizer
            determines the value, by default None
        """
        clusters, distances = self.get_exhaustive_clustering(threshold_distance)

        m = pyo.ConcreteModel()
        m.exhaustive_clustering_data = clusters
        m.wells_set = pyo.Set(initialize=list(clusters.keys()))
        m.select_cluster = pyo.Var(m.wells_set, within=pyo.Binary)

        @m.Expression(m.wells_set)
        def num_overlaps(blk, c):
            return (
                sum(blk.select_cluster[w] for w in m.wells_set if c in clusters[w]) - 1
            )

        m.assign_well_to_cluster = pyo.Constraint(
            m.wells_set, rule=lambda blk, c: blk.num_overlaps[c] >= 0
        )

        plug_cost = (
            None
            if self.wd.column_names.cost_of_plugging is None
            else self.wd[self.wd.column_names.cost_of_plugging]
        )

        for well, nearby_wells in clusters.items():
            _min_num_wells = self._get_min_num_wells_in_clusters(
                None if plug_cost is None else plug_cost[nearby_wells]
            )
            if len(nearby_wells) < _min_num_wells:
                m.select_cluster[well].fix(0)

        min_num_projects = self._get_min_num_projects()
        if min_num_projects is not None:
            m.ensure_enough_clusters_exist = pyo.Constraint(
                expr=sum(m.select_cluster.values()) >= min_num_projects
            )

        if fix_num_clusters is not None:
            m.fixing_num_clusters = pyo.Constraint(
                expr=sum(m.select_cluster.values()) == fix_num_clusters
            )

            if self.objective_type == "num_clusters":
                LOGGER.warning(
                    "Number of clusters is specified, and the objective_type is chosen"
                    "to be num_clusters. \n\t Changing the objective_type to cluster_radius"
                )
                self.objective_type = "cluster_radius"

        m.clustering_metrics = pyo.Expression(
            self._supported_objective_types,
            initialize={
                "num_clusters": sum(m.select_cluster.values()),
                "cluster_radius": sum(
                    distances[well][nearby_wells[-1]] * m.select_cluster[well]
                    for well, nearby_wells in clusters.items()
                ),
                "cluster_size": sum(
                    len(nearby_wells) ** 2 * m.select_cluster[well]
                    for well, nearby_wells in clusters.items()
                ),
            },
        )

        m.obj = pyo.Objective(expr=m.clustering_metrics[self.objective_type])
        LOGGER.info(f"Choosing {self.objective_type} as the objective function")
        return m
