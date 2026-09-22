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
import json
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List

# Installed libs
import ipywidgets as widgets
import pandas as pd
import requests
from ipyfilechooser import FileChooser
from IPython.display import display

# User-defined libs
from primo.utils.census_utils import get_census_key
from primo.utils.geo_utils import get_nearest_neighbors

LOGGER = logging.getLogger(__name__)


def file_path_widget(
    description: str, file_type: str, on_upload_change_str: str, default: str = None
):
    """
    Displays a widget on a Jupyter Notebook to provide a path for a file.

    Parameters
    ----------
    description : str
        The description to be displayed on the FileChooser widget

    file_type : str
        The file types that will be allowable by the widget

    on_upload_change_str : str
        The string to be displayed when the file is uploaded

    default : str, default=None
        The default path suggested by the widget

    Returns
    -------
    A FileChooser widget
    """
    kwargs = {"filter_pattern": f"*{file_type}", "title": f"<b> {description}</b>"}

    default_exists = False
    if default:
        if os.path.exists(default):
            default_exists = True
            default_path = default
        else:
            # check if the issue is using Windows style path on Linux or vice versa
            # 1: Test as windows path
            win_path = Path(PureWindowsPath(default))
            linux_path = Path(PurePosixPath(default))
            if os.path.exists(win_path):
                default_path = str(win_path)
                default_exists = True
            elif os.path.exists(linux_path):
                default_path = str(linux_path)
                default_exists = True
            else:
                LOGGER.warning(f"Default file path: {default} doesn't exist. Ignoring")

    if default_exists:
        # pylint: disable = used-before-assignment
        file_dir, file_path = os.path.split(default_path)
        kwargs["path"] = file_dir
        kwargs["filename"] = file_path
        kwargs["select_default"] = True
        kwargs["title"] = f"<b> Default {description} Selected </b>"

    def change_title(chooser):
        chooser.title = f"<b> {on_upload_change_str} </b>"

    file_widget = FileChooser(**kwargs)
    file_widget.register_callback(change_title)
    display(file_widget)
    return file_widget


def file_upload_widget(
    description: str, file_type: str, on_upload_change_str: str, multiple: bool = False
):
    """
    Displays a widget on a Jupyter Notebook to upload a file_type,

    Parameters
    ----------
    description : str
        The description to be displayed on the file upload widget

    file_type : str
        The file types that will be allowable by the widget


    on_upload_change_str : str
        The string to be displayed when the file is uploaded

    multiple : bool
        Whether multiple files are allowed to be uploaded

    Returns
    -------
    A FileUpload widget
    """

    input_file = widgets.FileUpload(
        accept=file_type, multiple=multiple, description=description
    )
    label = widgets.Label("")

    def on_upload_change(_):
        label.value = on_upload_change_str

    input_file.observe(on_upload_change, names="value")

    display(input_file, label)
    return input_file


def sort_by_nearest_wells(df: pd.DataFrame, distance: float) -> pd.DataFrame:
    """
    Sort a DataFrame by the distance to nearest wells.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    distance : float
        The distance threshold

    Returns
    -------
    pd.DataFrame
        The DataFrame sorted by nearest wells
    """
    well_locations = list(zip(df["latitude"], df["longitude"]))
    new_col = f"Wells within {distance} miles"
    df[new_col] = get_nearest_neighbors(well_locations, distance, "MILES")
    df = df.sort_values(by=new_col, ascending=True).reset_index(drop=True)
    return df


def get_population_by_state(state_code: int) -> pd.DataFrame:
    """
    Retrieves population data by state from the U.S. Census API.

    Parameters
    ----------
    state_code : int
        The state code

    Returns
    -------
    pd.DataFrame
        Population data by census tract in the specified state
    """
    CENSUS_KEY = get_census_key()
    url = "https://api.census.gov/data/2020/dec/dhc"
    params = {
        "get": "NAME,P1_001N",
        "for": "tract:*",
        "in": f"state:{state_code} county:*",
        "key": CENSUS_KEY,
    }
    session = requests.session()
    resp = session.get(url, params=params)
    assert resp.status_code == 200
    data = resp.json()
    pop = pd.DataFrame(data[1:], columns=data[0])
    pop = pop.rename(columns={"P1_001N": "Total Population"})
    for col in ["state", "county", "tract", "Total Population"]:
        pop[col] = pop[col].astype("int")
    return pop


def generate_configurations(
    weights_file_path: str,
    config_path: str,
    output_folder_path: str,
    life_prod_values: List,
    owner_well_counts: List,
):
    """
    Generate configuration files based on scenarios defined in an Excel file.

    Parameters
    ----------
    weights_file_path : str
        Path to the Excel file containing weight scenarios
    config_path : str
        Path to the base configuration JSON file for reference
    output_folder_path : str
        Path to the output folder where the generated configuration files will be saved
    life_prod_values: List
        List of values to be used for the lifelong production filters
    owner_well_counts: List
        List of values to be used for the owner well-count constraint in the opt model

    Returns
    -------
    None
    """

    # Load the Excel file with two sheets
    try:
        df_scenarios = pd.read_excel(weights_file_path, sheet_name="default")
    except ValueError as e:
        raise ValueError(f"Error loading sheet 'default': {e}")

    try:
        df_scenarios_owc = pd.read_excel(weights_file_path, sheet_name="owc constraint")
    except ValueError as e:
        raise ValueError(f"Error loading sheet 'owc constraint': {e}")

    num_scenarios = (
        df_scenarios.shape[1] - 3
    )  # Scenarios start after first three columns in weights excel file
    num_scenarios_owc = (
        df_scenarios_owc.shape[1] - 3
    )  # Scenarios start after first three columns in weights excel file

    # Load the base configuration
    with open(config_path, "r") as file:
        base_config = json.load(file)

    # Function to update the configuration with default values from a scenario
    def update_config_with_scenario(
        config: dict, df: pd.DataFrame, scenario_index: int
    ) -> dict:
        """
        Update the configuration dictionary with default values
        from a given scenario in the DataFrame.

        Parameters
        ----------
        config : dict
            The base configuration dictionary to be updated.
        df : pd.DataFrame
            The DataFrame containing scenario values. It should have
            columns 'Major Priority', 'Sub priority',
            and 'Scenario {scenario_index}'.
        scenario_index : int
            The index of the scenario to use for updating the configuration.

        Returns
        -------
        dict
            The updated configuration dictionary with values from the specified scenario.
        """
        current_major_priority = None

        for _, row in df.iterrows():
            major_priority = row["Major Priority"]
            sub_priority = row["Sub Priority"]
            default_value = row[f"Priority Weight for Scenario {scenario_index}"]

            if pd.notna(major_priority):
                current_major_priority = major_priority.strip()
                if current_major_priority not in config["impact_weights"]:
                    config["impact_weights"][current_major_priority] = {
                        "default": default_value,
                        "min_val": 0,
                        "max_val": 100,
                        "incr": 5,
                        "sub_weights": {},
                    }
                else:
                    config["impact_weights"][current_major_priority][
                        "default"
                    ] = default_value

            if pd.notna(sub_priority):
                sub_key = sub_priority.strip()
                if (
                    "sub_weights"
                    not in config["impact_weights"][current_major_priority]
                ):
                    config["impact_weights"][current_major_priority]["sub_weights"] = {}
                config["impact_weights"][current_major_priority]["sub_weights"][
                    sub_key
                ] = {
                    "default": default_value,
                    "min_val": 0,
                    "max_val": 100,
                    "incr": 5,
                }

        return config

    # Ensure output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Iterate over 9 scenarios for default scenarios
    for scenario_index in range(1, num_scenarios + 1):
        config = copy.deepcopy(base_config)
        config = update_config_with_scenario(config, df_scenarios, scenario_index)

        del config["program_constraints"]["Owner Well Count Constraint"]
        del config["program_constraints"]["Lifelong Production Constraint"]

        output_path = os.path.join(
            output_folder_path, f"config_scenario_{scenario_index}.json"
        )
        with open(output_path, "w") as outfile:
            json.dump(config, outfile, indent=4)

    # Generate configurations for the second sheet with owc constraints

    for scenario_index in range(1, num_scenarios_owc + 1):
        for owc in owner_well_counts:
            config = copy.deepcopy(base_config)
            config = update_config_with_scenario(
                config, df_scenarios_owc, scenario_index
            )

            # Update the Owner Well Count Constraint
            config["program_constraints"]["Owner Well Count Constraint"][
                "default"
            ] = owc

            output_path = os.path.join(
                output_folder_path, f"config_scenario_{scenario_index}_owc_{owc}.json"
            )
            with open(output_path, "w") as outfile:
                json.dump(config, outfile, indent=4)

    # Generate configurations for the prod cases

    for scenario_index in range(1, num_scenarios_owc + 1):
        for prod in life_prod_values:
            config = copy.deepcopy(base_config)
            config = update_config_with_scenario(
                config, df_scenarios_owc, scenario_index
            )

            # Update the 5-year Production Constraint
            config["program_constraints"]["Lifelong Production Constraint"][
                "default"
            ] = prod

            output_path = os.path.join(
                output_folder_path, f"config_scenario_{scenario_index}_prod_{prod}.json"
            )
            with open(output_path, "w") as outfile:
                json.dump(config, outfile, indent=4)
