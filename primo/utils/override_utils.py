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
import copy
import logging
import operator
from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Dict, List, Tuple

# Installed libs
import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector
from sklearn.neighbors import BallTree

# User-defined libs
from primo.data_parser.default_data import EARTH_RADIUS
from primo.opt_model.result_parser import Campaign
from primo.utils.clustering_utils import distance_matrix

LOGGER = logging.getLogger(__name__)


class AssessFeasibility:
    """
    Class for assessing whether the P&A projects adhere to the constraints
    defined in the optimization problem.

    Parameters
    ----------
    opt_inputs : OptModelInputs
        The optimization model inputs.

    opt_campaign : Dict
        A dictionary where keys are cluster numbers and values
        are list of wells for each cluster in the P&A projects.


    Attributes
    ----------
    campaign_cost_dict : Dict
        A dictionary that will hold the cost calculations for the projects within the campaign,
        mapped to their respective project identifiers.
    """

    def __init__(self, opt_inputs, opt_campaign: Dict):

        self.opt_inputs = opt_inputs
        self.new_campaign = opt_campaign
        self.campaign_cost_dict = {}

        # prevent duplication in plug_list
        self.plug_list = list(
            {well for well_list in self.new_campaign.values() for well in well_list}
        )
        self.wd = self.opt_inputs.config.well_data._construct_sub_data(self.plug_list)

        for cluster, groups in self.new_campaign.items():
            if self.wd.col_names.cost_of_plugging is not None:
                cluster_df = self.wd.data.loc[groups]
                num_wells = len(groups)
                beta = self.opt_inputs.config.discount_cost_factor
                cost_scale = (num_wells**beta) / num_wells
                campaign_cost = (
                    (cluster_df[self.wd.col_names.cost_of_plugging].sum())
                    * 1e-6
                    * cost_scale
                )
            else:
                # Fallback to mobilization cost if plugging cost is unavailable
                campaign_cost = self.opt_inputs.get_mobilization_cost[len(groups)]
            self.campaign_cost_dict[cluster] = campaign_cost

    def assess_budget(self) -> float:
        """
        Assesses whether the budget constraint is violated and returns the
        amount by which the budget is violated. A 0 or negative value indicates
        that we are still under budget
        """
        total_cost = 0

        total_cost = sum(
            project_cost for _, project_cost in self.campaign_cost_dict.items()
        )

        return round((total_cost - self.opt_inputs.get_total_budget) * 1e6)

    def assess_owner_well_count(self) -> Dict:
        # pylint: disable=protected-access
        """
        Assess whether the owner well count constraint is violated or not.
        Returns list of owners and wells selected for each for whom the owner
        well count constraint is violated
        """
        opt_inputs = self.opt_inputs.config
        max_wells_per_owner = opt_inputs.max_wells_per_owner
        if max_wells_per_owner is None:
            # When the user does not have owner information
            # or does not wish to prioritize
            # this constraint becomes meaningless
            return {}
        violated_operators = {}
        for well_operator, groups in self.wd.data.groupby(
            self.wd._col_names.operator_name
        ):
            n_wells = len(groups)
            if n_wells > self.opt_inputs.config.max_wells_per_owner:
                violated_operators.setdefault("Owner", []).append(well_operator)
                violated_operators.setdefault("Number of wells", []).append(n_wells)
                violated_operators.setdefault("Wells", []).append(
                    groups[self.wd._col_names.well_id].to_list()
                )

        return violated_operators

    def assess_distances(self) -> Dict:
        # pylint: disable=protected-access
        """
        Assess whether the maximum distance between two wells constraint is violated or not
        """
        distance_threshold = self.opt_inputs.config.threshold_distance
        distance_violation = {}
        # Assign weight for distance as 1 to ensure the distance matrix returns physical
        # distance between two well pairs
        metric_array = distance_matrix(self.wd, {"distance": 1})

        for cluster, well_list in self.new_campaign.items():
            for w1, w2 in combinations(well_list, 2):
                well_distance = metric_array.loc[w1, w2]
                if well_distance > distance_threshold:
                    distance_violation.setdefault("Project", []).append(cluster)
                    distance_violation.setdefault("Well 1", []).append(
                        self.wd.data.loc[w1][self.wd._col_names.well_id]
                    )
                    distance_violation.setdefault("Well 2", []).append(
                        self.wd.data.loc[w2][self.wd._col_names.well_id]
                    )
                    distance_violation.setdefault(
                        "Distance between Well 1 and 2 [Miles]", []
                    ).append(well_distance)

        return distance_violation

    def _assess_project_size_violation(self, threshold, comparator) -> Dict:
        """
        Generic function to assess whether a project violates the project size constraint.
        """
        num_wells_in_project_violation = {}

        for cluster, well_list in self.new_campaign.items():
            if comparator(len(well_list), threshold):
                num_wells_in_project_violation.setdefault("Project", []).append(cluster)
                num_wells_in_project_violation.setdefault("# of Wells", []).append(
                    len(well_list)
                )

        return num_wells_in_project_violation

    def assess_max_num_well_in_project(self) -> Dict:
        """
        Assess whether the max number of wells in a project constraint is violated or not
        """
        max_wells_in_project_threshold = self.opt_inputs.config.max_wells_in_project
        if max_wells_in_project_threshold is None:
            return {}

        return self._assess_project_size_violation(
            max_wells_in_project_threshold, operator.gt
        )

    def assess_min_num_well_in_project(self) -> Dict:
        """
        Assess whether the min number of wells in a project constraint is violated or not
        """
        min_wells_in_project_threshold = self.opt_inputs.config.min_wells_in_project
        if min_wells_in_project_threshold is None:
            return {}

        return self._assess_project_size_violation(
            min_wells_in_project_threshold, operator.lt
        )

    def _assess_project_budget_violation(
        self, threshold: float, comparator: Callable[[float, float], bool]
    ) -> Dict:
        """
        Generic function to assess whether a project violates the budget per project constraint.
        comparator is a comparison function, either operator.gt or operator, to evaluate whether
        the cost of a project violates the threshold
        """
        budget_per_project_violation = {}

        for cluster, campaign_cost in self.campaign_cost_dict.items():
            if comparator(campaign_cost, threshold):
                budget_per_project_violation.setdefault("Project", []).append(cluster)
                budget_per_project_violation.setdefault("Cost", []).append(
                    campaign_cost * 1e6
                )
        return budget_per_project_violation

    def assess_max_cost_project(self) -> Dict:
        """
        Assess whether the max budget per project constraint is violated or not
        """
        max_cost_project_threshold = self.opt_inputs.get_max_cost_project
        if max_cost_project_threshold is None:
            return {}

        return self._assess_project_budget_violation(
            max_cost_project_threshold, operator.gt
        )

    def assess_min_cost_project(self) -> Dict:
        """
        Assess whether the min budget per project constraint is violated or not
        """
        min_cost_project_threshold = self.opt_inputs.get_min_cost_project
        if min_cost_project_threshold is None:
            return {}

        return self._assess_project_budget_violation(
            min_cost_project_threshold, operator.lt
        )

    def assess_feasibility(self) -> bool:
        """
        Assesses whether current set of selections is feasible
        """
        violations = [
            self.assess_budget() > 0,
            self.assess_owner_well_count(),
            self.assess_distances(),
            self.assess_max_num_well_in_project(),
            self.assess_min_num_well_in_project(),
            self.assess_max_cost_project(),
            self.assess_min_cost_project(),
        ]

        return not any(violations)

    # pylint: disable=too-many-locals
    def violation_info(self):
        """
        Return information on constraints that the new campaign
        have violated.
        """
        violation_info_dict = {}
        if self.assess_feasibility() is False:
            violation_info_dict = {"Project Status:": "CONSTRAINT(S) VIOLATED"}
            violate_cost = self.assess_budget()
            violate_operator = self.assess_owner_well_count()
            violate_distance = self.assess_distances()
            violate_max_num_well_in_project = self.assess_max_num_well_in_project()
            violate_min_num_well_in_project = self.assess_min_num_well_in_project()
            violate_max_cost_project = self.assess_max_cost_project()
            violate_min_cost_project = self.assess_min_cost_project()

            if violate_cost > 0:
                msg = (
                    "After the modification, the total budget is over "
                    f"the limit by ${int(violate_cost):,d}."
                )

                violation_info_dict[msg] = """"""

            if violate_operator:
                msg = (
                    f"You have set a maximum of {self.opt_inputs.config.max_wells_per_owner} "
                    "well(s) per owner. Based on your selections, the following owner(s) "
                    "have wells that exceed this limit."
                )

                violate_operator_df = pd.DataFrame.from_dict(violate_operator)
                violation_info_dict[msg] = violate_operator_df

            if violate_distance:
                msg = (
                    "After the modification, the following projects have "
                    "wells that are far away from each other."
                )

                violate_distance_df = pd.DataFrame.from_dict(violate_distance)
                violation_info_dict[msg] = violate_distance_df

            if violate_max_num_well_in_project:
                msg = (
                    f"You have set a maximum of {self.opt_inputs.config.max_wells_in_project} "
                    "well(s) per project. Based on your selections, the number of "
                    "well(s) in the following project(s) exceeds this limit."
                )

                violate_max_wells_df = pd.DataFrame.from_dict(
                    violate_max_num_well_in_project
                )
                violation_info_dict[msg] = violate_max_wells_df

            if violate_min_num_well_in_project:
                msg = (
                    f"You have set a minimum of {self.opt_inputs.config.min_wells_in_project} "
                    "well(s) per project. Based on your selections, the number of "
                    "well(s) in the following project(s) is below this limit."
                )

                violate_min_wells_df = pd.DataFrame.from_dict(
                    violate_min_num_well_in_project
                )
                violation_info_dict[msg] = violate_min_wells_df

            if violate_max_cost_project:
                msg = (
                    "You have set a maximum budget of "
                    f"${int(self.opt_inputs.config.max_cost_project):,d} per project. "
                    "Based on your selections, the following project(s) "
                    "exceeds this limit."
                )

                violate_max_cost_project_df = pd.DataFrame.from_dict(
                    violate_max_cost_project
                )
                violation_info_dict[msg] = violate_max_cost_project_df

            if violate_min_cost_project:
                msg = (
                    "You have set a minimum budget of "
                    f"${int(self.opt_inputs.config.min_cost_project):,d} per project. "
                    "Based on your selections, the following project(s) "
                    "is below this limit."
                )

                violate_min_cost_project_df = pd.DataFrame.from_dict(
                    violate_min_cost_project
                )
                violation_info_dict[msg] = violate_min_cost_project_df

        else:
            violation_info_dict = {"Project Status:": "FEASIBLE"}

        return violation_info_dict


# pylint: disable=too-many-instance-attributes
class OverrideCampaign:
    """
    Class for constructing new campaigns based on the override results
    and returning infeasibility information.

    Parameters
    ----------
    override_selections : OverrideSelections
        Object containing the override selections

    opt_inputs : OptModelInputs
        Object containing the necessary inputs for the optimization model

    opt_campaign : dict
        A dictionary for the original suggested P&A project
        where keys are cluster numbers and values
        are list of wells for each cluster.

    eff_metrics : EfficiencyMetrics
        The efficiency metrics
    """

    def __init__(
        self,
        override_selections,
        opt_inputs,
        opt_campaign: Dict,
        eff_metrics,
    ):
        opt_campaign_copy = copy.deepcopy(opt_campaign)
        self.new_campaign = opt_campaign_copy
        self.remove = override_selections.remove_widget_return
        self.add = override_selections.add_widget_return
        self.lock = override_selections.lock_widget_return
        self.opt_inputs = opt_inputs
        self.eff_metrics = eff_metrics

        # change well cluster
        self._modify_campaign()

        self.feasibility = AssessFeasibility(self.opt_inputs, self.new_campaign)
        self.violation_info = self.feasibility.violation_info()

    def _modify_campaign(self):
        """
        Modify the original suggested P&A project
        """
        # remove clusters
        for cluster in self.remove.cluster:
            del self.new_campaign[cluster]

        # remove wells
        for cluster, well_list in self.remove.well.items():
            if cluster not in self.remove.cluster:
                for well in well_list:
                    self.new_campaign[cluster].remove(well)

        # add well with new cluster
        for cluster, well_list in self.add.new_clusters.items():
            self.new_campaign.setdefault(cluster, []).extend(well_list)

        # remove empty cluster to avoid the error in getting plugging cost of empty clusters
        cluster_to_delete = [
            cluster for cluster, well_list in self.new_campaign.items() if not well_list
        ]
        for cluster in cluster_to_delete:
            del self.new_campaign[cluster]

    def override_campaign(self):
        """
        Construct the new Campaign object based on the override selection
        """
        plugging_cost = self.feasibility.campaign_cost_dict
        wd = self.opt_inputs.config.well_data
        return Campaign(wd, self.new_campaign, plugging_cost, self.opt_inputs)

    def recalculate(self):
        """
        Recalculate the efficiency scores and impact scores of the new campaign
        based on the override selection
        """
        override_campaign = self.override_campaign()
        return override_campaign

    def recalculate_scores(self):
        """
        A function to return the impact score and efficiency score of
        the new campaign based on the override selection
        """
        override_campaign = self.recalculate()
        return {
            project_id: [project.impact_score, project.efficiency_score]
            for project_id, project in override_campaign.projects.items()
        }

    def re_optimize_data(self):
        """
        Generate dictionaries for clusters and wells to be fixed, along with
        a list of clusters that contain wells reassigned from other clusters,
        based on the override selection
        """
        re_optimize_cluster_dict = {}
        re_optimize_well_dict = {}

        # Assign 0 to wells being removed
        for cluster, well_list in self.remove.well.items():
            re_optimize_well_dict[cluster] = {well: 0 for well in well_list}

        # Assign 1 to wells being added
        for cluster, well_list in self.add.new_clusters.items():
            if cluster not in re_optimize_well_dict:
                re_optimize_well_dict[cluster] = {}
            for well in well_list:
                re_optimize_well_dict[cluster][well] = 1

        # Assign 1 to clusters being locked
        for cluster in self.lock.cluster:
            re_optimize_cluster_dict[cluster] = 1

        # Assign 1 to wells being locked
        for cluster, well_list in self.lock.well.items():
            if cluster not in re_optimize_well_dict:
                re_optimize_well_dict[cluster] = {}
            for well in well_list:
                re_optimize_well_dict[cluster][well] = 1

        # Generate a list of clusters which contain wells that are reassigned
        # from another cluster to this new cluster
        reassign = {
            key: list(
                set(self.add.new_clusters[key])
                - set(self.add.existing_clusters.get(key, []))
            )
            for key in self.add.new_clusters
            if set(self.add.new_clusters[key])
            - set(self.add.existing_clusters.get(key, []))
        }

        # Determine if an override was applied to ensure that re-optimization
        # returns the original results when no manual selections are made
        if (
            not self.remove.cluster
            and not self.remove.well
            and not self.add.existing_clusters
            and not self.lock.cluster
            and not self.lock.well
        ):
            override_status = False
        else:
            override_status = True

        return ReOptimizationData(
            re_optimize_cluster_dict, re_optimize_well_dict, reassign, override_status
        )


@dataclass
class ReOptimizationData:
    """
    Class for storing dictionaries for clusters and wells to be fixed,
    a list of clusters that contain wells reassigned from other clusters,
    and whether any override selection is made

    Parameters
    ----------
    re_optimize_cluster_dict : Dict[int, int]
        A dictionary mapping clusters (key) to binary values
        (0 or 1) indicating the cluster is fixed. Contains information on cluster
        selected to be locked

    re_optimize_well_dict : Dict[int, Dict[int, int]]
        A dictionary mapping clusters (key) to a list of dictionaries,
        where each dictionary contains wells (key) and their associated binary values
        (0 or 1) indicating the well is fixed within the respective cluster.

    reassign: Dict[int, List[int]]
        A dictionary contains ID of clusters (key) that contain wells reassigned from
        other clusters and the wells being reassigned (value).

    override_status: bool
        A boolean flag indicating whether an override selection has been made
    """

    re_optimize_cluster_dict: Dict[int, int]
    re_optimize_well_dict: Dict[int, Dict[int, int]]
    reassign: Dict[int, List[int]]
    override_status: bool


# pylint: disable=protected-access
def distance_reassigned_well(wells_in_cluster, wells_reassigned, wd):
    """
    Calculate the pairwise distances between existing wells in a cluster and
    wells being reassigned to that cluster from other clusters

    Parameters
    ----------
    wells_in_cluster : List
        A list of well currently belonging to the cluster

    wells_reassigned : List
        A list of well being reassigned to the cluster

    wd: WellData
        The WellData object.

    Raises
    ------
    pairwise_distance_dict: Dict
        A dictionary mapping well pairs (existing well, reassigned well) to their
        corresponding distance
    """
    wells_in_cluster_wd = wd._construct_sub_data(wells_in_cluster)
    wells_reassigned_wd = wd._construct_sub_data(wells_reassigned)

    cn = wd.column_names
    wells_in_cluster_coordinates = list(
        zip(
            wells_in_cluster_wd.data[cn.latitude],
            wells_in_cluster_wd.data[cn.longitude],
        )
    )
    wells_reassigned_coordinates = list(
        zip(
            wells_reassigned_wd.data[cn.latitude],
            wells_reassigned_wd.data[cn.longitude],
        )
    )

    pairwise_distance = haversine_vector(
        wells_in_cluster_coordinates,
        wells_reassigned_coordinates,
        unit=Unit.MILES,
        comb=True,
    )

    pairwise_distance_df = pd.DataFrame(
        pairwise_distance.transpose(),
        columns=wells_reassigned_wd.data.index,
        index=wells_in_cluster_wd.data.index,
    )

    well_pairs = list(product(list(wells_in_cluster), list(wells_reassigned)))

    pairwise_distance_matrix = pairwise_distance_df.stack()[well_pairs]

    pairwise_distance_dict = {
        (w1, w2): pairwise_distance_matrix[w1, w2]
        for w1, w2 in product(list(wells_in_cluster), list(wells_reassigned))
    }

    return pairwise_distance_dict


def identify_valid_nearby_projects(
    well_coordinates: Dict[int, Tuple[float, float]],
    project_centroids: Dict[int, Tuple[float, float]],
    distance: float,
) -> Dict[int, list[int]]:
    """
    Return valid projects for each well based on haversine distance.

    Parameters
    ----------
    well_coordinates : Dict[int, Tuple[float, float]]
        Mapping of well id to latitude/longitude coordinates.

    project_centroids : Dict[int, Tuple[float, float]]
        Mapping of project id to centroid latitude/longitude coordinates.

    distance : float
        Distance used to identify valid projects. A distance of 0 only returns
        project centroids at the exact same coordinates as a well. If no
        project is found in the radius, the nearest project is used as a
        fallback.

    Returns
    -------
    Dict[int, list[int]]
        Mapping of well id to valid project ids, ordered nearest to farthest.
    """
    if not well_coordinates:
        return {}

    if not project_centroids:
        raise ValueError("No projects were found for the requested scenario.")

    well_ids = list(well_coordinates.keys())
    project_ids = list(project_centroids.keys())

    well_points = np.deg2rad(
        np.array([well_coordinates[well_id] for well_id in well_ids])
    )
    project_points = np.deg2rad(
        np.array([project_centroids[project_id] for project_id in project_ids])
    )

    tree = BallTree(project_points, metric="haversine")
    _, nearest_project_indices = tree.query(well_points, k=1)
    nearby_project_indices, _ = tree.query_radius(
        well_points,
        r=max(distance or 0, 0) / EARTH_RADIUS,
        return_distance=True,
        sort_results=True,
    )

    valid_projects = {}
    for well_index, well_id in enumerate(well_ids):
        valid_indices = nearby_project_indices[well_index]
        if len(valid_indices) == 0:
            valid_indices = nearest_project_indices[well_index]
        valid_projects[well_id] = [
            project_ids[int(project_index)] for project_index in valid_indices
        ]

    return valid_projects


def find_nearby_efficient_project_id(
    well_id: int,
    project_ids: list[int],
    campaign: Campaign,
) -> int:
    """
    Return the nearby project with the highest efficiency delta.

    Parameters
    ----------
    well_id : int
        Well id to add.

    project_ids : list[int]
        Nearby project ids in nearest-first order. If efficiency deltas tie,
        the first project in this list remains selected.

    campaign : Campaign
        Original campaign.

    Returns
    -------
    int
        Project id with the highest efficiency delta after adding the well.
    """
    if len(project_ids) == 1:
        return project_ids[0]

    best_project_id = None
    best_efficiency_delta = None

    for project_id in project_ids:
        updated_campaign = campaign.update_campaign(
            add_wells_dict={project_id: [well_id]}
        )
        efficiency_delta = updated_campaign.get_efficiency_score_project(
            project_id
        ) - campaign.get_efficiency_score_project(project_id)

        if best_efficiency_delta is None or efficiency_delta > best_efficiency_delta:
            best_project_id = project_id
            best_efficiency_delta = efficiency_delta

    return best_project_id
