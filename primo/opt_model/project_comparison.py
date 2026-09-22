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
import pathlib
from typing import List, Union

# Installed libs
import pandas as pd

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.opt_model.result_parser import Campaign, Project
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


class ProjectComparison:
    """
    A class to manage and sort a list of Project objects based on specified sorting metrics.
    """

    def __init__(self, projects: List[Project], sorting_metric: str):
        """
        Initializes the ProjectComparison with a list of Project objects and a sorting metric.

        Parameters
        ----------
        projects : List[Project]
            A list of Project objects to be compared and sorted.
        sorting_metric : str
            The metric by which the Project objects should be sorted.
        """
        self.projects = projects
        self.sorting_metric = sorting_metric
        self.sort()

    def __str__(self):
        """
        Returns a summary of project comparison including total wells,
        total cost, and sorted project IDs.

        Returns
        -------
        str
            A summary string of the project comparison.
        """
        total_wells = sum(project.num_wells for project in self.projects)
        valid_cost = [
            project.plugging_cost
            for project in self.projects
            if project.plugging_cost != "N/A"
        ]

        if valid_cost:
            total_cost = sum(valid_cost)
            total_cost_str = f"${total_cost:,.2f}"
        else:
            total_cost_str = "not available"

        project_ids = [project.project_id for project in self.projects]

        return (
            f"Total Wells: {total_wells}"
            f"Total Plugging Cost: {total_cost_str}"
            f"Project IDs (Sorted): {project_ids}"
        )

    def __getitem__(self, project_rank: int):
        """
        Returns a project based on its position in the sorted list.

        Parameters
        ----------
        project_rank : int
            The 1-based rank of the project to retrieve.

        Returns
        -------
        Project
            The Project object at the specified rank.
        """
        return self.projects[project_rank - 1]

    def _repr_html_(self):
        """
        Returns a detailed HTML representation of the ProjectComparison instance,
        including a summary of key metrics and individual project details.

        Returns
        -------
        str
            HTML content representing project summary and details for Jupyter display.
        """
        total_wells = sum(project.num_wells for project in self.projects)
        valid_cost = [
            project.plugging_cost
            for project in self.projects
            if project.plugging_cost != "N/A"
        ]

        if valid_cost:
            total_cost = sum(valid_cost)
            total_cost_str = f"${total_cost:,.2f}"
        else:
            total_cost_str = "not available"

        highest_impact = max(
            self.projects, key=lambda project: project.impact_score, default=None
        )
        highest_efficiency = max(
            self.projects, key=lambda project: project.efficiency_score, default=None
        )

        df = pd.DataFrame(
            [
                {
                    "Project ID": project.project_id,
                    "Number of Wells": project.num_wells,
                    "Impact Score": project.impact_score,
                    "Efficiency Score": project.efficiency_score,
                    "Plugging Cost": project.plugging_cost,
                    # pylint: disable = protected-access
                    "Num Wells Missing Data": (
                        project._get_missing_data_information()[0]
                        if project._get_missing_data_information() is not None
                        else 0
                    ),
                }
                for project in self.projects
            ]
        )

        warning_html = ""
        if (df["Num Wells Missing Data"] > 0).any():
            warning_html = (
                "<pre><b>********** WARNING **********\n"
                "Projects in campaign contain wells for which priority factor data is missing.\n"
                "The missing data has been filled in with default values.</b></pre>"
            )

        summary_html = f"""
        <h3><b>Campaign Summary</b></h3>

        <p><i>Total Number of Projects: </i> {len(self.projects)}</p>
        <p><i>Total Number of Wells: </i> {total_wells}</p>
        <p><i>Total Plugging Cost: </i> {total_cost_str}</p>
        <p><i>Project with Highest Impact Score: Project </i> 
            {highest_impact.project_id if highest_impact else 'N/A'}
            (Score: {highest_impact.impact_score:.2f})
        </p>

        <p><i>Project with Highest Efficiency Score: Project </i> 
            {highest_efficiency.project_id if highest_efficiency else 'N/A'}
            (Score: {highest_efficiency.efficiency_score:.2f})
        </p>
        
        {warning_html}
        
        {df.to_html(index=False,
            formatters={"Plugging Cost": lambda val: f'{val:,.2f}' if val != "N/A" else val} )}
        """

        project_details_html = "<h3><b>Detailed Project Summary</b></h3>"
        for project in self.projects:
            project_details_html += f"""
            <h4><i>Project {project.project_id}</i></h4>
            <p>
                Well information for Project {project.project_id} with
                Efficiency Score: {project.efficiency_score:.2f} and
                Average Impact Score: {project.average_impact_score:.2f}
            </p>
            {project.get_well_info_dataframe().to_html(index=False)}
            """

        return summary_html + project_details_html

    def add_project(self, project: Project):
        """
        Adds a new Project object to the comparison list.

        Parameters
        ----------
        project : Project
            The Project object to add to the current list.
        """
        self.projects.append(project)

    def remove_project(self, project_id: int):
        """
        Removes a Project from the comparison list based on its project ID.

        Parameters
        ----------
        project_id : int
            The ID of the project to remove.
        """
        self.projects.remove(self.get_project(project_id))

    def sort(self, reverse: bool = True):
        """
        Sorts the projects in-place based on the sorting metric.

        Parameters
        ----------
        reverse : bool, optional
            Sort in descending order (default is True).
        """
        for project in self.projects:
            project.sorting_metric = self.sorting_metric
        self.projects.sort(reverse=reverse)

    def get_top_k_projects(self, k: int = None):
        """
        Get a new ProjectComparison object with only the top-k projects.

        Parameters
        ----------
        k : int, optional
            Number of top-ranked projects to include. If None, includes all projects.

        Returns
        -------
        ProjectComparison
            A new ProjectComparison object with the top-k projects.
        """
        if k is not None and k >= 1:
            return ProjectComparison(self.projects[:k], self.sorting_metric)
        return self

    def display(self, k: int = None):
        """
        Display the sorted project list, optionally limited to top-k projects.

        Parameters
        ----------
        k : int, optional
            The number of top-ranked projects to display. If None, all projects are displayed.

        Returns
        -------
        ProjectComparison or str
            A ProjectComparison instance if in Jupyter, else a plain string.
        """
        # pylint: disable=undefined-variable
        obj = self.get_top_k_projects(k)
        return obj if get_ipython() else str(obj)

    def get_project(self, project_id: int):
        """
        Retrieves a project from the list based on its project ID.

        Parameters
        ----------
        project_id : int
            The ID of the project to retrieve.

        Returns
        -------
        Project or None
            The Project object with the given ID, or None if not found.
        """
        return next(
            (project for project in self.projects if project.project_id == project_id),
            None,
        )


class WellProject:
    """
    A class to construct Project object that can be used for performing the project comparison.
    """

    def __init__(
        self,
        project_data: Union[str, pd.DataFrame],
        well_data: WellData,
        well_column: str,
        project_column: str,
        **kwargs,
    ):
        """
        Initialize the WellProject with well and project information.

        Parameters
        ----------
        project_data : Union[str, pandas Data frame]
            File path to the project information uploaded by the user or
            Data frame of data with the information

        well_data : WellData
            A WellData object containing the full set of valid well information.

        well_column : str
            Name of the column in the project data file containing well identifiers.

        project_column : str
            Name of the column in the project data file containing project identifiers.
        """
        self.wd = well_data
        self.well_column = well_column
        self.project_column = project_column
        self.project_dict = {}
        self.project_map = {}

        if isinstance(project_data, str):
            LOGGER.info("Reading the project information from input file.")
            extension = pathlib.Path(project_data).suffix
            if extension in [".xlsx", ".xls"]:
                self.project_data = pd.read_excel(
                    project_data,
                    sheet_name=kwargs.pop("sheet", 0),
                    dtype=str,
                )
            elif extension == ".csv":
                self.project_data = pd.read_csv(
                    project_data,
                    dtype=str,
                )

            else:
                raise_exception(
                    "Unsupported input file format. Only .xlsx, .xls, and .csv are supported.",
                    TypeError,
                )
        else:
            LOGGER.info("Received Data frame of information")
            self.project_data = project_data

        project_data_check(self.project_data, self.project_column, self.well_column)
        self.project_data_consistency()
        self.project_mapping()

    def project_data_consistency(self):
        """
        Validate that all wells in the project data exist in the well data file.
        """

        project_wells = self.project_data[self.well_column]
        well_missing = []
        # pylint: disable = protected-access
        for well in project_wells:
            if well not in self.wd.data[self.wd._col_names.well_id].values:
                well_missing.append(well)

        if len(well_missing) > 0:
            msg = (
                f"The following wells are not in the well data input: {well_missing}."
                f"Please check your input file."
            )
            raise_exception(msg, ValueError)

    def project_mapping(self):
        """
        Create mappings from project id to well indices.
        """

        for _, row in self.project_data.iterrows():
            project = int(row[self.project_column])
            well_id = row[self.well_column]

            well_index = self.wd.data[
                self.wd.data[self.wd.column_names.well_id] == well_id
            ].index.item()
            self.project_dict.setdefault(project, []).append(well_index)

        wells_not_selected = [
            self.wd.data[
                self.wd.data[self.wd.column_names.well_id] == well
            ].index.item()
            for well in self.wd.data[self.wd.column_names.well_id]
            if well not in self.project_data[self.well_column].values
        ]
        self.project_map = copy.deepcopy(self.project_dict)

        # Create a project_map which serves as cluster_mapping when setting up
        # OptModelInputs to skip the clustering step.
        # Assigning all wells not selected for any project to project ID 0 supports
        # implementing overrides under project comparison–only scenarios in the future.
        if wells_not_selected:
            self.project_map.setdefault(0, []).extend(wells_not_selected)

    def campaign_creation(
        self,
        opt_model_inputs,
    ):
        """
        Construct a Campaign object based on user provided project data

        Parameters
        ----------
        opt_model_inputs : OptModelInputs
            A OptModelInputs object for model inputs

        Returns
        -------
        campaign: Campaign
            A Campaign object built from project and well mappings.
        """
        plugging_cost = {}
        if opt_model_inputs.config.mobilization_cost:
            mob_cost = opt_model_inputs.get_mobilization_cost
            for project, wells in self.project_dict.items():
                plugging_cost[project] = mob_cost[len(wells)]
        elif self.wd.col_names.cost_of_plugging is not None:
            for project, wells in self.project_dict.items():
                project_df = self.wd.data.loc[wells]
                num_wells = len(wells)
                beta = opt_model_inputs.config.discount_cost_factor
                cost_scale = (num_wells**beta) / num_wells
                plugging_cost[project] = (
                    (project_df[self.wd.col_names.cost_of_plugging].sum())
                    * 1e-6
                    * cost_scale
                )
        else:
            plugging_cost = None

        LOGGER.info("Constructing projects.")
        campaign = Campaign(self.wd, self.project_dict, plugging_cost, opt_model_inputs)

        return campaign


# This function is defined outside the WellProject class to ensure it can be used
# as a separate check on the API side when users upload only the project input file
# and have not specified the well data file.
def project_data_check(project_data, project_column, well_column):
    """
    Perform validation checks on the project data input.

    Checks:
        1. Wells without associated project numbers.
        2. Projects missing well id information.
        3. Duplicate well IDs assigned to multiple projects.
        4. All project IDs are provided in integer format.

    Parameters
    ----------
    project_data : pd.DataFrame
        DataFrame containing user-uploaded project information and corresponding wells.

    project_column : str
        Name of the column containing project ids.

    well_column : str
        Name of the column containing well ids.

    Raises
    ------
        ValueError: If any of the above issues are found.
    """
    msg = ""

    # Wells with no associated project
    wells_no_project = list(
        project_data[project_data[project_column].isna()][well_column]
    )

    # Projects with well id missing
    projects_no_well = list(
        set(project_data[project_data[well_column].isna()][project_column])
    )

    # Duplicate well ids
    duplicates = (
        project_data[well_column]
        .loc[project_data[well_column].duplicated()]
        .dropna()
        .tolist()
    )

    no_missing_project_data = project_data[project_data[project_column].notna()]
    flag = no_missing_project_data[project_column].apply(
        lambda x: not isinstance(x, int) and not (isinstance(x, str) and x.isdigit())
    )
    wells_with_non_integer_project_ids = no_missing_project_data.loc[
        flag, well_column
    ].tolist()

    if len(wells_no_project) > 0:
        msg += (
            "The following wells do not have an associated project number: "
            f"{wells_no_project}.\n"
        )

    if len(projects_no_well) > 0:
        msg += (
            "The following projects have missing well information: "
            f"{sorted(projects_no_well)}.\n"
        )

    if len(duplicates) > 0:
        msg += f"The following wells are assigned to multiple projects: {duplicates}.\n"

    if len(wells_with_non_integer_project_ids) > 0:
        msg += (
            "The following wells have non-integer project number: "
            f"{wells_with_non_integer_project_ids}.\n"
        )

    if msg:
        msg += "Please check your input file."
        raise_exception(msg, ValueError)
