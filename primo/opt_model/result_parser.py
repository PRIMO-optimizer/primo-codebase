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
import logging
from typing import List, Optional, Union

# Installed libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# User-defined libs
from primo.data_parser import EfficiencyMetrics
from primo.data_parser.default_data import SORTING_METRICS
from primo.data_parser.well_data import WellData
from primo.utils.clustering_utils import distance_matrix
from primo.utils.raise_exception import MissingDataError, raise_exception

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods, too-many-instance-attributes
class Project:
    """
    Class for storing optimal projects
    """

    def __init__(
        self,
        wd: WellData,
        index: list,
        plugging_cost: Optional[float],
        project_id: int,
    ):
        """
        Constructs an object for storing optimal project results
        Parameters
        ----------
        wd : WellData
            A WellData object

        index : list
            List of indices/rows/wells belonging to the cluster/group

        plugging_cost : Optional[float]
            Cost of plugging

        project_id : int
            Project id
        """
        # Motivation for storing the entire DataFrame: After the problem is solved
        # it is desired to display/highlight flagged wells for which the information
        # was not available, and the missing data was filled. Having the entire
        # DataFrame allows ease of access to flagged wells (columns containing _flag)
        self.well_data = wd._construct_sub_data(index)
        # If priority scores are available, then sort the wells
        if hasattr(self.well_data.column_names, "priority_score"):
            self.well_data = self.well_data.get_high_priority_wells(self.num_wells)
            self.maximum_priority_score = wd.get_maximum_priority_score

        self._col_names = self.well_data.column_names
        self.project_id = project_id
        # Optimization problem uses million USD. Convert it to USD
        if plugging_cost is not None:
            self.plugging_cost = plugging_cost * 1e6
            self.cost_str = f"${round(self.plugging_cost):,}"
        else:
            self.plugging_cost = "N/A"
            self.cost_str = "Not available"

        self.sorting_metric = "total_impact_score"

    def __iter__(self):
        return iter(self.well_data.data.index)

    def __str__(self) -> str:

        msg = (
            f"Number of wells in project {self.project_id}\t\t: {self.num_wells}\n"
            f"Estimated Project Cost\t\t\t: {self.cost_str}\n"
            f"Impact Score [0-100]\t\t\t: {self.impact_score:.2f}\n"
            f"Efficiency Score [0-100]\t\t: {self.efficiency_score:.2f}\n"
        )

        msg = "=" * 70
        msg += (
            f"\nProject ID: {self.project_id} \n\n"
            f"Number of wells in the project\t\t: {self.num_wells}\n"
            f"Estimated project cost\t\t\t: {self.cost_str}\n"
            f"Sum of impact scores of all wells\t: {self.total_impact_score:,.2f}\n"
            f"Average impact Score [0-100]\t\t: {self.impact_score:.2f}\n"
            f"Efficiency Score [0-100]\t\t: {self.efficiency_score:.2f}\n"
        )

        if self.column_names.hospitals is not None:
            _num_wells = self.num_wells_near_hospitals
            msg += f"\nNumber of wells that are near a hospital: {_num_wells}"

        if self.column_names.schools is not None:
            _num_wells = self.num_wells_near_schools
            msg += f"\nNumber of wells that are near a school\t: {_num_wells}"

        if msg[-1] != "\n":
            msg += "\n"

        # Print a warning if input data is missing
        missing_data = self._get_missing_data_information()
        if missing_data is not None:
            msg += (
                f"\n********** WARNING **********\n"
                f"Required data is missing for "
                f"{missing_data[0]} wells in this project. \n"
                f"Use the `missing_data` property to query the wells "
                f"for which input data is missing.\n"
            )
        msg += "=" * 70
        return msg

    def __contains__(self, well: Union[int, str]):
        """Checks if a well is contained in the project or not"""
        # Supporting both index and well id
        return (
            well in self.well_data.data.index
            or well in self.well_data[self.column_names.well_id]
        )

    def __len__(self):
        """Returns number of wells in the project"""
        return len(self.well_data.data)

    @property
    def sorting_metric(self):
        """Returns the current sorting metric."""
        return self._sorting_metric

    @sorting_metric.setter
    def sorting_metric(self, new_metric: str):
        """Updates the sorting metric."""
        if new_metric not in SORTING_METRICS:
            raise ValueError(
                f"Unrecognized sorting metric {new_metric}."
                f"\n\t Supported metrics are {SORTING_METRICS}."
            )
        self._sorting_metric = new_metric

    def _check_comparable(self, other):
        """Ensures the two Project objects use the same config for comparisons."""
        if not isinstance(other, Project):
            raise TypeError("Can only compare Project objects.")

        if (
            self.well_data.config.impact_metrics
            != other.well_data.config.impact_metrics
        ):
            raise ValueError("Projects use different impact metric weights.")

        if (
            self.well_data.config.efficiency_metrics
            != other.well_data.config.efficiency_metrics
        ):
            raise ValueError("Projects use different efficiency metric weights.")

    def __lt__(self, other):
        """
        Defines the less-than comparison for sorting
        Project objects based on the sorting metric.
        """
        self._check_comparable(other)
        return getattr(self, self._sorting_metric) < getattr(
            other, self._sorting_metric
        )

    def __eq__(self, other):
        """
        Defines the equal-to comparison for sorting
        Project objects based on the sorting metric.
        """
        self._check_comparable(other)
        return np.isclose(
            getattr(self, self._sorting_metric), getattr(other, self._sorting_metric)
        )

    def _repr_html_(self):
        """Nicely formats the data in Jupyter notebook"""

        msg = (
            f"<h4>===================================================================</h4>"
            f"<h4><b> Project ID: {self.project_id} </b></h4>"
            f"<pre>Number of wells in the project              : {self.num_wells}\n"
            f"Estimated project cost                      : {self.cost_str}\n"
            f"Sum of impact scores of all wells           : {self.total_impact_score:,.2f}\n"
            f"Average impact score [0-100]                : {self.impact_score:.2f}\n"
            f"Efficiency score [0-100]                    : {self.efficiency_score:.2f} \n\n"
        )

        if self.column_names.hospitals is not None:
            _num_wells = self.num_wells_near_hospitals
            msg += f"Number of wells that are near a hospital    : {_num_wells}\n"

        if self.column_names.schools is not None:
            _num_wells = self.num_wells_near_schools
            msg += f"Number of wells that are near a school      : {_num_wells}\n"

        # Print a warning if input data is missing
        missing_data = self._get_missing_data_information()
        if missing_data is not None:
            msg += (
                f"<h4 style='color:red'><b> WARNING </b></h4>"
                f"<p style='color:red'>Required data is missing for "
                f"{missing_data[0]} wells in this project. "
                f"Default values are used for the missing data. <br>"
                f"Use the `missing_data` property to query the wells "
                f"for which input data is missing. </p>"
            )

        msg += (
            "</pre>"
            "<h4>===================================================================</h4>"
        )

        # pylint: disable = protected-access
        msg += "<h4><b> Summary: </b></h4>"
        msg += self.well_data._repr_html_()

        return msg

    def _get_data_column(self, col_name: str):
        """
        Checks if a column exists. If it exists, it returns the data column.
        """
        # NOTE: `try` is faster than `if`, if we know that the condition
        # is expected to pass in most of the scenarios. This is known as
        # "Easier to Ask Forgiveness Than Permission" (EAFP) style.
        try:
            return self.well_data[getattr(self._col_names, col_name)]
        except KeyError as exp:
            raise MissingDataError(
                f"{col_name} data is not in the input well data"
            ) from exp

    @property
    def num_wells_near_hospitals(self):
        """Returns number of wells that are near hospitals"""
        data_col = self._get_data_column("hospitals")
        return len(self.well_data[data_col > 0].index)

    @property
    def num_wells_near_schools(self):
        """Returns number of wells that are near schools"""
        data_col = self._get_data_column("schools")
        return len(self.well_data[data_col > 0].index)

    @property
    def num_wells(self):
        """Returns the number of wells in the project"""
        return len(self)

    @property
    def average_age(self):
        """
        Returns average age of the wells in the project
        """
        return self._get_data_column("age").mean()

    @property
    def age_range(self):
        """
        Returns the range of the age of the project
        """
        data_col = self._get_data_column("age")
        return data_col.max() - data_col.min()

    @property
    def dist_range(self):
        """
        Returns the range of the distance of the project
        """
        if self.num_wells == 1:
            return 0
        dist_matrix = distance_matrix(self.well_data, {"distance": 1})
        return dist_matrix.max().max()

    @property
    def average_depth(self):
        """
        Returns the average depth of the project
        """
        return self._get_data_column("depth").mean()

    @property
    def depth_range(self):
        """
        Returns the range of the depth of the project
        """
        data_col = self._get_data_column("depth")
        return data_col.max() - data_col.min()

    @property
    def elevation_delta(self):
        """
        Returns the maximum elevation delta of the project
        """
        return self._get_data_column("elevation_delta").max()

    @property
    def centroid(self):
        """
        Returns the centroid of the project
        """
        cols = [self._col_names.latitude, self._col_names.longitude]
        return tuple(self.well_data[cols].mean().round(6))

    @property
    def dist_to_road(self):
        """
        Returns the maximum distance to road for a project
        """
        return self._get_data_column("dist_to_road").max()

    @property
    def population_density(self):
        """
        Returns the maximum population density value for a project
        """
        return self._get_data_column("population_density").max()

    @property
    def column_names(self):
        """
        Returns column names associated with project data
        """
        return self._col_names

    @property
    def num_unique_owners(self):
        """
        Returns the number of well owners for a project
        """
        return len(set(self._get_data_column("operator_name")))

    @property
    def efficiency_metric_scores(self):
        """
        Returns efficiency scores of selected metrics as a dictionary
        """
        eff_metric_scores = {}
        for key, value in self.__dict__.items():
            if "eff_score" in key:
                metric_name = key.split("_eff_score", maxsplit=1)[0]
                eff_metric_scores[metric_name] = value

        return eff_metric_scores

    @property
    def max_efficiency_metric_scores(self):
        """Returns the maximum score possible for all selected metrics"""
        max_eff_scores = {}
        for key in self.__dict__:
            if "eff_score" in key:
                metric_name = key.split("_eff_score", maxsplit=1)[0]
                max_eff_scores[metric_name] = int(key.split("_")[-1])

        return max_eff_scores

    @property
    def efficiency_score(self):
        """Returns the efficiency score of the project"""
        return sum(self.efficiency_metric_scores.values())

    @property
    def impact_score(self):
        """
        Returns the average priority score for the project
        """
        if not hasattr(self._col_names, "priority_score"):
            raise AttributeError(
                "The priority score has not been computed for the Well Data"
            )
        if (
            hasattr(self._col_names, "priority_score_normalized")
            and self.maximum_priority_score > 100
        ) or getattr(self.well_data.config.impact_metrics, "basis", 0) > 100:
            return self.well_data[self._col_names.priority_score_normalized].mean()

        return self.well_data[self._col_names.priority_score].mean()

    @property
    def modified_impact_score(self):
        """
        In the case of a custom basis or user supplied score:

        - If the custom basis/user supplied max score is greater than 100,
          it returns the average of the raw scores (not scaled).
        - If the custom basis/user supplied max score is less than 100,
          it returns the average of the scaled scores (out of 100).
        """
        if not hasattr(self._col_names, "priority_score"):
            raise AttributeError(
                "The priority score has not been computed for the Well Data"
            )

        if hasattr(self._col_names, "priority_score_normalized"):
            if (
                self.maximum_priority_score > 100
                or getattr(self.well_data.config.impact_metrics, "basis", 0) > 100
            ):
                return self.well_data[self._col_names.priority_score].mean()

            return self.well_data[self._col_names.priority_score_normalized].mean()

        # does not have priority score normalized column
        return self.well_data[self._col_names.priority_score].mean()

    average_impact_score = impact_score

    @property
    def total_impact_score(self):
        """
        Returns the average priority score for the project
        """
        return self.impact_score * self.num_wells

    def get_efficiency_metric_scores(self):
        """
        Return efficiency scores of selected metrics as a dictionary
        """
        eff_metric_scores = {}
        for key, value in self.__dict__.items():
            if "eff_score" in key:
                metric_name = key.split("_eff_score", maxsplit=1)[0]
                eff_metric_scores[metric_name] = value
        return eff_metric_scores

    @property
    def accessibility_score(self):
        """
        Returns the accessibility score and the total weight of the accessibility
        score for a project
        """
        eff_scores = self.efficiency_metric_scores
        max_eff_scores = self.max_efficiency_metric_scores
        access_score = 0
        max_access_score = 0

        for metric in ["elevation_delta", "dist_to_road"]:
            access_score += eff_scores.get(metric, 0)
            max_access_score += max_eff_scores.get(metric, 0)

        if max_access_score == 0:
            # Accessibility metrics are not selected
            return None
        return max_access_score, access_score

    def get_max_val_col(self, col_name) -> float:
        """
        Returns the maximum value of a column

        Parameters
        ----------
        col_name : str
            column name

        Returns
        -------
        float
            the maximum value of the column

        """
        return self.well_data[col_name].max()

    def get_well_info_dataframe(self):
        """
        Returns the data frame to display in the notebook
        """
        cols = self.well_data.get_essential_columns
        normalized_score_col = getattr(
            self.well_data.col_names, "priority_score_normalized", None
        )
        if normalized_score_col:
            cols.append(normalized_score_col)
        return self.well_data[cols]

    def _get_missing_data_information(self):
        """
        Returns the summary of missing input data
        """
        flag_cols = self.well_data.get_flag_columns
        md = self.well_data[flag_cols]
        num_missing_data = md.sum(axis=1).astype(int)
        num_wells_missing_data = sum(num_missing_data > 0)

        if num_wells_missing_data == 0:
            # There is no missing data in the input well data
            return None

        col_headers = {
            col: col.split("_flag")[0] for col in md.columns if "_flag" in col
        }

        return num_wells_missing_data, md.rename(columns=col_headers)

    @property
    def missing_data(self):
        """
        Returns a DataFrame that contains information on missing data
        for wells that are in this project.
        """
        _missing_data = self._get_missing_data_information()
        if _missing_data is None:
            return None

        return (
            _missing_data[1]
            .replace({0: "", 1: "NA"})
            .reset_index()
            .rename(columns={"index": "Row Number in Input Data"})
            .style.hide(axis="index")
        )


class Campaign:
    """
    Represents an optimal campaign that consists of multiple projects.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        wd: WellData,
        clusters_dict: dict,
        plugging_cost: Optional[dict],
        opt_model_inputs,
        efficiency_model_scores: Optional[dict[int, dict[str, float]]] = None,
    ):
        """
        Represents an optimal campaign that consists of multiple projects.

        Parameters
        ----------
        wd : WellData
            WellData object

        clusters_dict : dict
            A dictionary where keys are cluster numbers and values
            are list of wells for each cluster.

        plugging_cost : Optional[dict]
            A dictionary where keys are cluster numbers and values
            are plugging cost for that cluster

        efficiency_model_scores : Optional[dict[int, dict[str, float]]]
            A dictionary where keys are cluster numbers and the values
            are a dictionary of efficiency score names mapped to values
        """
        # for now include a pointer to well data, so that I have column names
        self.wd = wd
        self.projects = {}
        self.clusters_dict = clusters_dict
        self.opt_model_inputs = opt_model_inputs

        for cluster, wells in self.clusters_dict.items():
            cost = plugging_cost[cluster] if plugging_cost else None

            self.projects[cluster] = Project(
                wd=wd,
                index=wells,
                plugging_cost=cost,
                project_id=cluster,
            )

        self.efficiency_calculator = EfficiencyCalculator(
            self,
            opt_model_inputs.config.well_data.config.efficiency_metrics,
        )
        self.efficiency_calculator.compute_efficiency_scores()

        # If efficiency scores are provided, then validate them
        # Capturing both None and empty dictionary scenarios
        if efficiency_model_scores is not None and len(efficiency_model_scores) > 0:
            self.validate_efficiency_model_scores(efficiency_model_scores)
        mob_cost = self.opt_model_inputs.config.mobilization_cost
        user_plugging_cost = (
            self.opt_model_inputs.config.well_data.col_names.cost_of_plugging
        )
        if mob_cost is not None or user_plugging_cost is not None:
            self.cost_str = f"${round(self.total_plugging_cost):,}"
        else:
            self.cost_str = "not available"

    def __iter__(self):
        """Iterates over all project objects"""
        return iter(self.projects.values())

    def __len__(self):
        """Returns the number of projects in the campaign"""
        return len(self.projects)

    def __str__(self) -> str:

        msg = (
            f"Optimal campaign has {self.num_projects} projects and "
            f"the cost is {self.cost_str}.\n\n"
        )
        data = self.get_campaign_summary()
        data["Plugging Cost [$]"] = data["Plugging Cost [$]"].apply(
            lambda cost: f"{round(cost):,}" if cost != "N/A" else cost
        )

        if "Num Wells Missing Data" in data.columns:
            msg += (
                "********** WARNING **********\n"
                "Projects in this campaign contain wells for which priority "
                "factor data is missing. \n"
                "The missing data has been filled in with default values.\n\n"
            )

        msg += "Campaign Summary:\n\n" + data.to_string(index=False)
        return msg

    def _repr_html_(self):
        """Nicely formats the output in Jupyter notebook"""

        msg = (
            f"<p><b>Optimal campaign has {self.num_projects} projects and "
            f"the cost is {self.cost_str}.</p></b>\n\n"
        )
        data = self.get_campaign_summary()
        data["Plugging Cost [$]"] = data["Plugging Cost [$]"].apply(
            lambda cost: f"{round(cost):,}" if cost != "N/A" else cost
        )

        if "Num Wells Missing Data" in data.columns:
            msg += (
                "<p style='color: red'><b>WARNING: </b></p>"
                "<p style='color:red'>Projects in this campaign contain "
                "wells for which priority factor data is missing. <br>"
                "The missing data has been filled in with default values. </p>\n\n"
            )

        # pylint: disable = protected-access
        msg += "<p><b>Campaign Summary: </b></p>\n\n\n"

        project_details_html = "<h3><b>Detailed Project Summary</b></h3>"
        for project_id, project in self.projects.items():
            project_details_html += f"""
            <h4><i>Project {project_id}</i></h4>
            <p>
                Well information for Project {project_id} with
                Efficiency Score: {project.efficiency_score:.2f} and
                Average Impact Score: {project.average_impact_score:.2f}
            </p>
            {project.get_well_info_dataframe().to_html(index=False)}
            """

        return (
            msg
            + "<b>"
            + data.to_html(index=False)
            + "</b>"
            + project_details_html
            + "</b>"
        )

    @property
    def num_projects(self):
        """Returns the number of projects in the campaign"""
        return len(self.projects)

    @property
    def total_plugging_cost(self):
        """
        Returns the total plugging cost of the campaign
        """
        return sum(project.plugging_cost for project in self.projects.values())

    def get_project_id_by_well_id(self, well_id: str) -> Optional[int]:
        """
        Returns the project_id associated with the given well_id.

        Parameters
        ----------
        well_id : str
            The ID of the well.

        Returns
        -------
        Optional[int]
            The project ID if the well exists in any project; otherwise, None.
        """
        # pylint: disable=protected-access
        for project_id, project in self.projects.items():
            if (
                well_id
                in project.well_data[project.well_data._col_names.well_id].values
            ):
                return project_id
        return None

    def get_max_value_across_all_projects(self, attribute: str) -> Union[float, int]:
        """
        Returns the max value for an attribute across projects

        Parameters
        ----------
        attribute : str
            name of the attribute of interest
        """
        # Since we expect the attribute to be present in most cases, switching to
        # the EAFP style described above.
        try:
            return max(getattr(project, attribute) for project in self)

        except AttributeError as exp:
            raise AttributeError(
                "The project does not have the requested attribute: " + attribute
            ) from exp

    def get_min_value_across_all_projects(self, attribute: str) -> Union[float, int]:
        """
        Returns the min value for an attribute across projects

        Parameters
        ----------
        attribute : str
            name of the attribute of interest

        """
        try:
            return min(getattr(project, attribute) for project in self)

        except AttributeError as exp:
            raise AttributeError(
                "The project does not have the requested attribute: " + attribute
            ) from exp

    def get_max_value_across_all_wells(self, col_name: str) -> Union[float, int]:
        """
        Returns the max value for all wells in the data set

        Parameters
        ----------
        col_name : str
            name of the column containing the values of interest

        """
        return self.wd[col_name].max()

    def get_min_value_across_all_wells(self, col_name: str) -> Union[float, int]:
        """
        Returns the max value for all wells in the data set

        Parameters
        ----------
        col_name : str
            name of the column containing the values of interest
        """
        return self.wd[col_name].min()

    def plot_campaign(self, title: str):
        """
        Plots the projects of the campaign

        Parameters
        ---------
        title : str
            Title for the plot
        """
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=[
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "yellow",
                "cyan",
                "magenta",
                "pink",
                "brown",
                "black",
            ]
        )
        plt.figure()
        ax = plt.gca()
        for project in self.projects.values():
            ax.scatter(
                project.well_data[project.column_names.longitude],
                project.well_data[project.column_names.latitude],
            )
        plt.title(title)
        plt.xlabel("x-coordinate of wells")
        plt.ylabel("y-coordinate of wells")
        plt.show()

    def get_project_well_information(self):
        """
        Returns a dict of DataFrames corresponding to each project containing essential data
        """
        return {
            project.project_id: project.get_well_info_dataframe()
            for project in self.projects.values()
        }

    def get_efficiency_score_project(self, project_id: int) -> float:
        """
        Returns the efficiency score of a project in the campaign given its id
        Parameters
        ----------
        project_id : int
            Project id

        Returns
        -------
        float
            The efficiency score of the project
        """
        return self.projects[project_id].efficiency_score

    def get_impact_score_project(self, project_id: int) -> float:
        """
        Returns the impact score of a project in the campaign given its id
        Parameters
        ----------
        project_id : int
            Project id

        Returns
        -------
        float
            The impact score of the project
        """
        return self.projects[project_id].impact_score

    def _efficiency_metrics_column_headers(self):
        """
        Returns a dictionary with the appropriate column headers
        """
        first_project = list(self)[0]
        return {
            key: key.replace("_", " ").title() + f" Score [0-{value}]"
            for key, value in first_project.max_efficiency_metric_scores.items()
        }

    def get_efficiency_metric_scores(self):
        """
        Returns a data frame with the different efficiency scores for the projects
        """
        # TODO What to do with single well projects

        df = pd.DataFrame([project.efficiency_metric_scores for project in self])
        df["Efficiency Score [0-100]"] = df.sum(axis=1)
        df.insert(0, "Project ID", [project.project_id for project in self])
        df = df.rename(columns=self._efficiency_metrics_column_headers())

        # Returns a non-DataFrame object. To retrieve the DataFrame use the
        # data attribute: e.g., output.data
        return df.style.hide(axis="index").format(precision=2)

    def get_accessibility_scores(self):
        """Returns accessibility scores for all projects in the campaign"""
        first_project = list(self)[0]
        access_score = first_project.accessibility_score

        if access_score is None:
            LOGGER.warning(
                "Accessibility metrics are not selected, so accessibility "
                "scores are not available."
            )
            return None

        col_name = f"Accessibility Score [0-{access_score[0]}]"
        data = {
            "Project ID": [project.project_id for project in self],
            col_name: [project.accessibility_score[1] for project in self],
        }
        return pd.DataFrame(data)

    def display_efficiency_metric_scores(self):
        """
        Returns a DataFrame containing efficiency metric scores and
        accessibility scores (if they are available) for all projects
        """
        access_scores = self.get_accessibility_scores()
        eff_scores = self.get_efficiency_metric_scores()
        if access_scores is None:
            # Accessibility scores are not available
            return eff_scores

        # Append accessibility scores to efficiency scores and return
        access_score_col = access_scores.columns[-1]
        eff_scores.data[access_score_col] = access_scores[access_score_col]

        return eff_scores.data.style.hide(axis="index").format(precision=2)

    def get_campaign_summary(self):
        """
        Returns a pandas data frame of the project summary for demo printing
        """
        rows = [
            [
                project.project_id,
                project.num_wells,
                project.plugging_cost,
                project.impact_score,
                project.efficiency_score,
            ]
            for project in self.projects.values()
        ]
        header = [
            "Project ID",
            "Number of Wells",
            "Plugging Cost [$]",
            "Impact Score [0-100]",
            "Efficiency Score [0-100]",
        ]

        df = pd.DataFrame(rows, columns=header)
        missing_data = []
        for project in self.projects.values():
            # pylint: disable = protected-access
            md = project._get_missing_data_information()
            if md is None:
                missing_data.append(0)
            else:
                missing_data.append(md[0])

        if sum(missing_data) > 0:
            df["Num Wells Missing Data"] = missing_data

        return df

    def export_data(
        self,
        excel_writer: pd.ExcelWriter,
        campaign_category: str,
        columns_to_export: list = None,
    ):
        """
        Exports campaign data to an excel file

        Parameters
        ----------
        excel_writer : pd.ExcelWriter
            The excel writer
        campaign_category : str
            The label for the category of the campaign (e.g., "oil", "gas")
        """
        first_key = list(self.projects.keys())[0]
        col_names = self.projects[first_key].column_names
        # the priority score must have been previously computed
        assert hasattr(col_names, "priority_score")
        if columns_to_export is None:
            columns_to_export = (
                self.wd.get_essential_columns + self.wd.get_priority_score_columns
            )

        # add the project data
        start_row = 0
        for project in self.projects.values():
            wells_df = project.well_data.data[columns_to_export].copy()
            wells_df["Project ID"] = pd.Series(
                [project.project_id] * len(wells_df), index=wells_df.index
            )
            cols = list(wells_df.columns)
            cols.insert(0, cols.pop(cols.index("Project ID")))
            wells_df = wells_df.loc[:, cols]
            wells_df.rename(
                columns={col_names.priority_score: "Well Priority Score [0-100]"}
            )
            wells_df.to_excel(
                excel_writer,
                sheet_name=campaign_category + " Well Projects",
                startrow=start_row,
                startcol=0,
                index=False,
            )
            start_row += len(wells_df) + 2  # Add one line spacing after each table

        # add the campaign summary
        self.get_campaign_summary().to_excel(
            excel_writer, sheet_name=campaign_category + "Project Scores", index=False
        )

    def set_efficiency_weights(self, eff_metrics: EfficiencyMetrics):
        """
        Wrapper for the EfficiencyCalculator efficiency_weights property
        """
        self.efficiency_calculator.efficiency_weights = eff_metrics

    def compute_efficiency_scores(self):
        """
        Wrapper for the EfficiencyCalculator compute efficiency scores method
        """
        self.efficiency_calculator.compute_efficiency_scores()

    def validate_efficiency_model_scores(self, efficiency_model_scores):
        """
        Verifies if the computed efficiency scores and the scores obtained
        from the optimization model are the same. Raises an error if they
        are not.
        """
        for project in self:
            calculated_scores = project.efficiency_metric_scores
            model_scores = efficiency_model_scores[project.project_id]

            for key in calculated_scores:
                if not key in model_scores:
                    raise KeyError(
                        "The keys from the efficiency model score computation "
                        "and the keys of the project efficiency score computation"
                        " do not match. Please contact PRIMO developers."
                    )
                if not np.isclose(model_scores[key], calculated_scores[key], atol=1e-4):
                    raise RuntimeError(
                        "Calculated and model efficiency scores did not match. "
                        "Please contact PRIMO developers."
                    )

    def _remove_wells(
        self,
        updated_clusters_dict: dict[int, list[int]],
        remove_wells_dict: dict[int, list[Union[int, str]]],
    ) -> dict[int, list[int]]:
        """
        Removes wells from projects.

        Parameters
        ----------
        updated_clusters_dict : dict[int, list[int]]
            Dictionary where keys are project IDs and values are lists of well
            row indices in each project.

        remove_wells_dict : dict[int, list[Union[int, str]]]
            Dictionary where keys are project IDs and values are lists of well
            IDs to remove from each project.
        """
        well_id_col = self.wd.col_names.well_id
        for project_id, well_ids in remove_wells_dict.items():
            if project_id not in updated_clusters_dict:
                raise_exception(
                    f"Project ID {project_id} is not in the campaign.", ValueError
                )

            for well_id in well_ids:
                try:
                    well_index = self.wd.data.index[
                        self.wd.data[well_id_col] == well_id
                    ][0]
                except IndexError:
                    raise_exception(
                        f"{well_id} is not found in the dataset. Check if there is an error",
                        ValueError,
                    )
                if well_index not in updated_clusters_dict[project_id]:
                    raise_exception(
                        f"Well ID {well_id} is not in project {project_id}.",
                        ValueError,
                    )
                updated_clusters_dict[project_id].remove(well_index)

        return updated_clusters_dict

    def _add_wells(
        self,
        updated_clusters_dict: dict[int, list[int]],
        add_wells_dict: dict[int, list[Union[int, str]]],
    ) -> dict[int, list[int]]:
        """
        Adds wells to projects.

        Parameters
        ----------
        updated_clusters_dict : dict[int, list[int]]
            Dictionary where keys are project IDs and values are lists of well
            row indices in each project.

        add_wells_dict : dict[int, list[Union[int, str]]]
            Dictionary where keys are project IDs and values are lists of well
            IDs to add to each project.
        """
        well_id_col = self.wd.col_names.well_id
        for project_id, well_ids in add_wells_dict.items():
            if project_id not in updated_clusters_dict:
                raise_exception(
                    f"Project ID {project_id} is not in the campaign.", ValueError
                )

            for well_id in well_ids:
                well_index = self.wd.data.index[self.wd.data[well_id_col] == well_id][0]
                existing_project_id = None
                for cluster_id, cluster_wells in updated_clusters_dict.items():
                    if well_index in cluster_wells:
                        existing_project_id = cluster_id
                        break

                if existing_project_id:
                    if existing_project_id == project_id:
                        raise_exception(
                            f"Well ID {well_id} is already in project {project_id}.",
                            ValueError,
                        )
                    updated_clusters_dict[existing_project_id].remove(well_index)
                    LOGGER.info(
                        "Well ID %s moved from project %s to project %s.",
                        well_id,
                        existing_project_id,
                        project_id,
                    )

                updated_clusters_dict[project_id].append(well_index)

        return updated_clusters_dict

    def _update_cost(
        self,
        updated_clusters_dict: dict[int, list[int]],
    ) -> Optional[dict[int, float]]:
        """
        Updates plugging costs for the updated campaign.

        Parameters
        ----------
        updated_clusters_dict : dict[int, list[int]]
            Dictionary where keys are project IDs and values are lists of well
            row indices in each project.
        """
        mob_cost = self.opt_model_inputs.config.mobilization_cost
        user_plugging_cost = self.wd.col_names.cost_of_plugging
        if mob_cost is not None:
            updated_cost_dict = {}
            mobilization_cost = self.opt_model_inputs.get_mobilization_cost
            for project_id, wells in updated_clusters_dict.items():
                updated_cost_dict[project_id] = mobilization_cost[len(wells)]
            return updated_cost_dict

        updated_cost_dict = {}
        beta = self.opt_model_inputs.config.discount_cost_factor
        for project_id, wells in updated_clusters_dict.items():
            cost_scale = (len(wells) ** beta) / len(wells)
            updated_cost_dict[project_id] = (
                self.wd.data.loc[wells, user_plugging_cost].sum() * 1e-6 * cost_scale
            )
        return updated_cost_dict

    def update_campaign(
        self,
        add_wells_dict: Optional[dict[int, list[Union[int, str]]]] = None,
        remove_wells_dict: Optional[dict[int, list[Union[int, str]]]] = None,
    ):
        """
        Updates a campaign by adding and removing wells from projects.

        Parameters
        ----------
        add_wells_dict : Optional[dict[int, list[Union[int, str]]]]
            Dictionary where keys are project IDs and values are lists of well
            IDs to add to each project.

        remove_wells_dict : Optional[dict[int, list[Union[int, str]]]]
            Dictionary where keys are project IDs and values are lists of well
            IDs to remove from each project.

        Returns
        -------
        Campaign
            A new Campaign object with updated wells, costs, and scores.
        """
        updated_clusters_dict = {
            project_id: list(wells) for project_id, wells in self.clusters_dict.items()
        }
        if remove_wells_dict:
            updated_clusters_dict = self._remove_wells(
                updated_clusters_dict, remove_wells_dict
            )

        if add_wells_dict:
            updated_clusters_dict = self._add_wells(
                updated_clusters_dict, add_wells_dict
            )

        for project_id, wells in list(updated_clusters_dict.items()):
            if len(wells) == 0:
                LOGGER.info(
                    "Project %s has no wells after campaign update. "
                    "Removing project.",
                    project_id,
                )
                del updated_clusters_dict[project_id]

        if len(updated_clusters_dict) == 0:
            raise_exception(
                "Updated campaign must contain at least one project.", ValueError
            )

        return Campaign(
            wd=self.wd,
            clusters_dict=updated_clusters_dict,
            plugging_cost=self._update_cost(updated_clusters_dict),
            opt_model_inputs=self.opt_model_inputs,
        )


class EfficiencyCalculator:
    """
    A class to compute efficiency scores for projects.
    """

    def __init__(self, campaign: Campaign, efficiency_weights: EfficiencyMetrics):
        """
        Constructs the object for all of the efficiency computations for a campaign

        Parameters
        ----------

        campaign : Campaign
            The final campaign for efficiencies to be computed

        efficiency_weights : EfficiencyMetrics
            List of efficiency metrics

        """
        self.campaign = campaign
        self._efficiency_weights = efficiency_weights

    @property
    def efficiency_weights(self):
        """Returns the pointer to efficiency_metrics"""
        return self._efficiency_weights

    @efficiency_weights.setter
    def efficiency_weights(self, efficiency_weights: EfficiencyMetrics):
        """
        Sets the attribute containing the efficiency weights
        """
        if self._efficiency_weights is not None:
            raise RuntimeError("Attempting to overwrite efficiency_weights")

        # Code reaches here only if the efficiency weights are not specified
        # at the time the instance is created. This implies that efficiency weights
        # are not available in the WellData object. So, after updating
        # weights here, add a pointer in the WellData object as well
        self._efficiency_weights = efficiency_weights
        wd = self.campaign.opt_model_inputs.config.well_data
        wd.set_impact_and_efficiency_metrics(efficiency_metrics=efficiency_weights)

    def _compute_efficiency_attributes_for_project(self, project: Project):
        """
        Adds attributes to each project object with the metric efficiency score

        Parameters
        ----------
        project : Project
            project in an Campaign

        """
        for metric in self.efficiency_weights:
            if metric.weight == 0 or hasattr(metric, "submetrics"):
                # Metric/submetric is not chosen, or
                # This is a parent metric, so no data assessment is required
                continue

            LOGGER.debug(
                f"Computing scores for metric/submetric {metric.name}/{metric.full_name}."
            )

            scaling_factor = getattr(
                self.campaign.opt_model_inputs.config, "max_" + metric.name
            )

            assert getattr(project, metric.score_attribute, None) is None
            if metric.name == "num_wells":
                score = (
                    getattr(project, metric.name) / scaling_factor
                ) * metric.effective_weight
            elif metric.name == "num_unique_owners":
                if scaling_factor == 1:
                    score = metric.effective_weight
                else:
                    score = (
                        1 - (getattr(project, metric.name) - 1) / (scaling_factor - 1)
                    ) * metric.effective_weight
            elif "range" in metric.name and project.num_wells == 1:
                score = 0
            else:
                score = (
                    1 - getattr(project, metric.name) / scaling_factor
                ) * metric.effective_weight
            setattr(
                project,
                metric.score_attribute,
                min(max(score, 0), metric.effective_weight),
            )

    def compute_efficiency_scores(self):
        """
        Function to compute the efficiency scores for the campaign
        """
        if self.efficiency_weights is None:
            # Efficiency metrics are not available, so return
            LOGGER.warning(
                "Efficiency metrics are not specified, so skipping "
                "efficiency calculations."
            )
            return

        # Compute efficiency scaling factors
        self.campaign.opt_model_inputs.compute_efficiency_scaling_factors()
        for project in self.campaign:
            LOGGER.debug(
                f"Computing efficiency scores for project {project.project_id}"
            )
            self._compute_efficiency_attributes_for_project(project)


def export_data_to_excel(
    output_file_path: str,
    campaigns: List[Campaign],
    campaign_categories: List[str],
):
    """
    Exports the data from campaigns to an excel file

    Parameters
    ----------
    output_file_path : str
        The path to the output file
    campaigns : List[Campaign]
        A list of campaigns to output data for
    campaign_categories : List[str]
        A list of labels corresponding to the campaigns in the campaigns argument
    """
    excel_writer = pd.ExcelWriter(output_file_path, engine="xlsxwriter")
    for idx, campaign in enumerate(campaigns):
        campaign.export_data(excel_writer, campaign_categories[idx])
    excel_writer.close()
