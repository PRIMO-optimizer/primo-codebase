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
# pylint: disable = too-many-lines
# pylint: disable = too-many-arguments, too-many-positional-arguments

# Standard libs
import logging
import os

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser import ImpactMetrics, ScenarioType, WellDataColumnNames
from primo.data_parser.well_data import WellData

LOGGER = logging.getLogger(__name__)


INCOMPLETE_ROWS = {
    "API Well Number": [2, 3, 4, 5, 6],
    "x": [7, 8, 9, 10, 11],
    "y": [12, 13, 14, 15, 16],
    "Operator": [17, 18, 19, 20, 21],
    "Unknown Operator": [22, 23, 24, 25, 26],
    "Age": [27, 28, 29, 30, 31],
    "Depth": [32, 33, 34, 35, 36],
    "Leak": [37, 38, 39, 40, 41],
    "Violation": [42, 43, 44, 45, 46],
    "Incident": [47, 48, 49, 50, 51],
    "Compliance": [52, 53, 54, 55, 56],
    "Oil": [57, 58, 59, 60, 61],
    "Gas": [62, 63, 64, 65, 66],
    "Hospitals": [67, 68, 69, 70, 71],
    "Schools": [72, 73, 74, 75, 76],
    "Life Gas Fill": [77, 78, 79, 80, 81],
    "Life Oil Fill": [82, 83, 84, 85, 86],
    "Life Gas Remove": [87, 88, 89, 90, 91],
    "Life Oil Remove": [92, 93, 94, 95, 96],
}


# pylint: disable = missing-function-docstring, protected-access
# pylint: disable = unused-variable, no-member
@pytest.fixture(name="get_column_names", scope="function")
def get_column_names_fixture():
    """Returns well data from a csv file and scenario type"""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        hospitals="Hospitals",
        schools="Schools",
        life_gas_production="Lifelong Gas [Mcf]",
        life_oil_production="Lifelong Oil [bbl]",
        placeholder_one="Placeholder 1",
        placeholder_two="Placeholder 2",
        placeholder_three="Placeholder 3",
        placeholder_four="Placeholder 4",
        placeholder_five="Placeholder 5",
        placeholder_six="Placeholder 6",
        placeholder_seven="Placeholder 7",
        placeholder_eight="Placeholder 8",
        placeholder_nine="Placeholder 9",
        placeholder_ten="Placeholder 10",
        placeholder_eleven="Placeholder 11",
        placeholder_twelve="Placeholder 12",
        placeholder_thirteen="Placeholder 13",
        placeholder_fourteen="Placeholder 14",
        placeholder_fifteen="Placeholder 15",
        placeholder_sixteen="Placeholder 16",
        placeholder_seventeen="Placeholder 17",
        placeholder_eighteen="Placeholder 18",
        placeholder_nineteen="Placeholder 19",
        placeholder_twenty="Placeholder 20",
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data.csv",
    )

    return filename, col_names


@pytest.fixture(name="get_random_generator", scope="function")
def get_random_generator_fixture():
    """
    Return a random generator with a known seed
    """
    return np.random.default_rng(42)


@pytest.fixture(name="get_random_lat_long_bounds", scope="function")
def get_random_lat_long_bounds_fixture():
    """
    Provides a block randomly selected across the United States
    """
    return (-113, -83, 33, 41)


def test_well_data(caplog, get_column_names):
    """Returns well data from a csv file"""

    filename, col_names = get_column_names
    wd = WellData(
        data=filename,
        column_names=col_names,
        fill_age=99,
        fill_depth=999,
        fill_life_gas_production=1.5,
        fill_life_oil_production=1.5,
        min_lifetime_oil_production=0,
        max_lifetime_oil_production=2,
        min_lifetime_gas_production=0,
        max_lifetime_gas_production=2,
        well_depth_limit=2000,
    )

    # Test column_names alias
    assert wd.col_names is wd.column_names

    # Well ID checks
    assert (
        f"Removed wells because {col_names.well_id} "
        f"information is not available for them."
    ) in caplog.text
    assert (
        wd.get_removed_wells_with_reason["missing well_id"]
        == INCOMPLETE_ROWS["API Well Number"]
    )

    # Latitude checks
    assert (
        f"Removed wells because {col_names.latitude} "
        f"information is not available for them."
    ) in caplog.text
    assert wd.get_removed_wells_with_reason["missing latitude"] == INCOMPLETE_ROWS["x"]

    # Longitude checks
    assert (
        f"Removed wells because {col_names.longitude} "
        f"information is not available for them."
    ) in caplog.text
    assert wd.get_removed_wells_with_reason["missing longitude"] == INCOMPLETE_ROWS["y"]

    # Operator name checks
    assert (
        f"Removed wells because {col_names.operator_name} "
        f"information is not available for them."
    ) in caplog.text
    assert (
        wd.get_removed_wells_with_reason["missing operator_name"]
        == INCOMPLETE_ROWS["Operator"]
    )

    assert (
        "Owner name for some wells is listed as unknown."
        "Treating these wells as if the owner name is not provided, "
        "so removing them from the dataset."
    ) in caplog.text
    assert (
        wd.get_removed_wells_with_reason["unknown operator_name"]
        == INCOMPLETE_ROWS["Unknown Operator"]
    )

    # Age checks
    assert "Found empty cells in column Age [Years]" in caplog.text
    assert (
        "Filling any empty cells in column Age [Years] with value 99."
    ) in caplog.text

    # Ensure that wells are not deleted because of missing age
    assert "age" not in wd.get_removed_wells_with_reason
    for row in wd:
        if row in INCOMPLETE_ROWS["Age"]:
            assert wd.data.loc[row, "Age [Years]_flag"] == 1
            assert wd.data.loc[row, col_names.age] == 99
        else:
            assert wd.data.loc[row, "Age [Years]_flag"] == 0

    # Depth checks
    assert "Found empty cells in column Depth [ft]" in caplog.text
    assert (
        "Filling any empty cells in column Depth [ft] with value 999."
    ) in caplog.text

    # Ensure that wells are not deleted because of missing depth
    assert "depth" not in wd.get_removed_wells_with_reason
    for row in wd:
        if row in INCOMPLETE_ROWS["Depth"]:
            assert wd.data.loc[row, "Depth [ft]_flag"] == 1
            assert wd.data.loc[row, col_names.depth] == 999
        else:
            assert wd.data.loc[row, "Depth [ft]_flag"] == 0

    # Filter volume checks
    assert (
        wd.data.loc[INCOMPLETE_ROWS["Life Gas Fill"], col_names.life_gas_production]
        == 1.5
    ).all()  # Check if the missing data is filled correctly
    assert (
        wd.data.loc[INCOMPLETE_ROWS["Life Oil Fill"], col_names.life_oil_production]
        == 1.5
    ).all()  # Check if the missing data is filled correctly
    assert wd.get_removed_wells_with_reason[
        "lifetime gas production above the maximum threshold"
    ] == (INCOMPLETE_ROWS["Life Gas Remove"])
    assert wd.get_removed_wells_with_reason[
        "lifetime oil production above the maximum threshold"
    ] == (INCOMPLETE_ROWS["Life Oil Remove"])
    assert (
        "Some wells have been removed for being above the maximum specified lifetime gas "
        "production volume."
    ) in caplog.text
    assert (
        "Some wells have been removed for being above the maximum specified lifetime oil "
        "production volume."
    ) in caplog.text

    # Threshold depth is specified, so the processing step categorizes wells as
    # shallow and deep
    assert isinstance(wd._well_types["shallow"], set)
    assert isinstance(wd._well_types["deep"], set)

    # Data needed to classify oil and gas wells is available
    assert isinstance(wd._well_types["oil"], set)
    assert isinstance(wd._well_types["gas"], set)

    # Warning on number of wells removed
    assert (
        "Preliminary processing removed 35 "
        "wells (11.67% wells in the input data set) \n\tbecause of missing "
        "information. To include these wells in the analysis, \n\tplease provide the missing "
        "information. \n\tList of wells removed can be queried using the get_removed_wells "
        "\n\tand get_removed_wells_with_reason properties."
    ) in caplog.text

    new_column = np.ones(len(wd.data))
    badly_sized_column = np.ones(len(wd.data) - 1)
    with pytest.raises(AttributeError):
        wd.add_new_column_ordered("ones", "col_of_ones", badly_sized_column)

    # scenario_type is set to its default
    assert wd.config.scenario_type.well_ranking
    assert wd.config.scenario_type.project_recommendation
    assert not wd.config.scenario_type.project_comparison

    # no priority_score attribute when priority info is not provided directly
    assert not hasattr(wd._col_names, "priority_score")

    # test the branch on get_maximum_priority_score where there is no priority
    # score column
    assert wd.get_maximum_priority_score is None


def test_no_warnings(caplog, get_column_names):
    """
    This function:
    1. Checks if no warnings are thrown if there are no issues with data.
    2. Passing the data as DataFrame instead of a file.
    3. No categorization is performed if the required data is not
        available.
    4. A warning is printed if the required input for filter production
        volume function is not available
    5. Raises an error if the Well API is not unique
    """
    filename, col_names = get_column_names

    # Doing this to avoid oil-gas classification
    col_names.ann_gas_production = None
    col_names.ann_oil_production = None
    col_names.life_gas_production = None
    col_names.life_oil_production = None

    wd_df = pd.read_csv(filename, usecols=col_names.values())
    # Removing 100 wells with incomplete data
    wd_df = wd_df.drop(index=list(range(100)))
    assert wd_df.shape[0] == 200

    wd = WellData(
        data=wd_df,
        column_names=col_names,
        fill_age=99,
        fill_depth=999,
        fill_life_gas_production=1.5,
        fill_life_oil_production=1.5,
        min_lifetime_oil_production=0,
        max_lifetime_oil_production=2,
        min_lifetime_gas_production=0,
        max_lifetime_gas_production=2,
    )

    # Check no warning messages related to empty are printed
    assert "Found empty cells in column" not in caplog.text
    assert (
        "information. To include these wells in the analysis, please provide the missing "
        "information. List of wells removed can be queried using the get_removed_wells "
        "and get_removed_wells_with_reason properties."
    ) not in caplog.text

    # Filter production volume is active, but the required data is not available
    # This should print a warning.
    assert (
        "Unable to filter wells based on production volume because both lifelong gas "
        "production [in Mcf] and lifelong oil production [in bbl] are missing in the "
        "input data. Please provide at least one of the columns via attributes "
        "life_gas_production and life_oil_production in the WellDataColumnNames object."
    ) in caplog.text

    # Categorization is not performed because data is not available
    assert wd._well_types["oil"] is None
    assert wd._well_types["gas"] is None
    assert wd._well_types["shallow"] is None
    assert wd._well_types["deep"] is None

    assert wd.get_gas_oil_wells is None
    assert (
        "Insufficient information for well categorization. Either specify "
        "well_type in the input data, or provide both ann_gas_production "
        "and ann_oil_production"
    ) in caplog.text

    assert wd.get_shallow_deep_wells is None
    assert (
        "Insufficient information for well categorization. Either specify "
        "well_type_by_depth in the input data, or specify well_depth_limit "
        "while instantiating the WellData object."
    ) in caplog.text

    assert wd.get_fully_partitioned_data is None
    assert "Insufficient information for well categorization." in caplog.text

    # Raises an error if the Well API is not unique
    wd_df.loc[103, col_names.well_id] = wd_df.loc[102, col_names.well_id]
    with pytest.raises(
        ValueError,
        match="Well ID must be unique. Found multiple wells with the same Well ID.",
    ):
        wd = WellData(data=wd_df, column_names=col_names)


def test_empty_well_data(get_column_names):
    "Tests that the correct error is raised if all wells are removed during preliminary processing"

    filename, col_names = get_column_names
    # The first 35 rows have incomplete data, so all wells will be removed during processing
    df = pd.read_csv(filename).head(35)

    msg = "All wells in the data set have been removed during preliminary processing."
    msg += "\n5 wells were removed for missing well_id."
    msg += "\n5 wells were removed for missing latitude."
    msg += "\n5 wells were removed for missing longitude."
    msg += "\n5 wells were removed for missing operator_name."
    msg += "\n5 wells were removed for unknown operator_name."
    msg += "\n5 wells were removed for missing age."
    msg += "\n5 wells were removed for missing depth."

    with pytest.raises(ValueError, match=msg):
        wd = WellData(
            data=df,
            column_names=col_names,
            missing_age="remove",
            missing_depth="remove",
        )


def test_age_depth_remove(caplog, get_column_names):
    """Tests the remove method for missing age and depth"""
    filename, col_names = get_column_names

    wd = WellData(
        data=filename,
        column_names=col_names,
        missing_age="remove",
        missing_depth="remove",
    )

    assert (
        f"Removed wells because {col_names.age} "
        f"information is not available for them."
    ) in caplog.text
    assert wd.get_removed_wells_with_reason["missing age"] == INCOMPLETE_ROWS["Age"]

    assert (
        f"Removed wells because {col_names.depth} "
        f"information is not available for them."
    ) in caplog.text
    assert wd.get_removed_wells_with_reason["missing depth"] == INCOMPLETE_ROWS["Depth"]


def test_age_depth_estimation(caplog, get_column_names):
    """Tests the estimate method for missing age and depth"""
    filename, col_names = get_column_names

    with pytest.raises(
        NotImplementedError,
        match="age estimation feature is not supported currently.",
    ):
        wd = WellData(
            data=filename,
            column_names=col_names,
            missing_age="estimate",
        )

    assert "Estimating the age of a well if it is missing." in caplog.text

    with pytest.raises(
        NotImplementedError,
        match="depth estimation feature is not supported currently.",
    ):
        wd = WellData(
            data=filename,
            column_names=col_names,
            missing_depth="estimate",
        )

    assert "Estimating the depth of a well if it is missing." in caplog.text


def test_both_production_type(
    get_column_names,
):
    filename, col_names = get_column_names
    # Add well types to the input data
    wd_df = pd.read_csv(filename, usecols=col_names.values())
    # Removing 100 wells with incomplete data
    wd_df = wd_df.drop(index=list(range(100))).reset_index()
    assert wd_df.shape[0] == 200

    # Initialize columns
    col_names.well_type = "Well Type"
    wd_df[col_names.well_type] = "OIL"
    # wd_df[col_names.ann_gas_production] = 10000000000

    for i in range(0, 100, 2):
        # Test if categorization is successful when well type
        # is both
        wd_df.loc[i, col_names.well_type] = "both"

    for i in range(1, 100, 2):
        # Test if categorization is successful when well type
        # is both
        wd_df.loc[i, col_names.well_type] = "oil"
    # Construct the WellData object
    wd = WellData(data=wd_df, column_names=col_names)

    gas_oil_wells = wd.get_gas_oil_wells
    assert isinstance(gas_oil_wells["oil"], WellData)
    assert gas_oil_wells["oil"].data.shape[0] == 179
    assert isinstance(gas_oil_wells["gas"], WellData)
    assert gas_oil_wells["gas"].data.shape[0] == 21
    assert len(gas_oil_wells) == 2


def test_get_high_priority_wells(get_column_names, caplog):
    """Test the get_high_priority_wells method"""
    filename, col_names = get_column_names
    df = pd.read_csv(filename)
    wd = WellData(data=df, column_names=col_names)
    wd.add_new_column_ordered(
        "priority_score", "Priority Score [0-100]", np.linspace(100, 0, len(wd))
    )
    top_wells = wd.get_high_priority_wells(5)

    assert not wd.config.priority_score_reverse
    assert top_wells is not None
    assert isinstance(top_wells, WellData)
    assert len(top_wells) == 5
    assert (
        top_wells[col_names.priority_score].tolist()
        == sorted(wd[col_names.priority_score], reverse=True)[:5]
    )

    assert "Returning None, since priority scores are not available!" not in caplog.text

    # Test the warning
    delattr(col_names, "priority_score")
    wd = WellData(data=df, column_names=col_names)
    top_wells = wd.get_high_priority_wells(5)

    assert top_wells is None
    assert "Returning None, since priority scores are not available!" in caplog.text


def test_well_partitioning(
    tmp_path, get_column_names
):  # pylint: disable=too-many-statements
    filename, col_names = get_column_names
    # Doing this to avoid oil-gas classification
    col_names.ann_gas_production = None
    col_names.ann_oil_production = None
    col_names.life_gas_production = None
    col_names.life_oil_production = None

    # Add well types to the input data
    wd_df = pd.read_csv(filename, usecols=col_names.values())
    # Removing 100 wells with incomplete data
    wd_df = wd_df.drop(index=list(range(100))).reset_index()
    assert wd_df.shape[0] == 200

    # Initialize columns
    col_names.well_type = "Well Type"
    col_names.well_type_by_depth = "Well Type by depth"
    wd_df[col_names.well_type] = "OIL"
    wd_df[col_names.well_type_by_depth] = "Deep"

    for i in range(100):
        wd_df.loc[i, col_names.well_type] = "Gas"

    for i in range(50):
        wd_df.loc[i, col_names.well_type_by_depth] = "Shallow"

    for i in range(100, 150):
        wd_df.loc[i, col_names.well_type_by_depth] = "Shallow"

    # Construct the WellData object
    wd = WellData(data=wd_df, column_names=col_names)

    assert wd._well_types["oil"] == set(range(100, 200))
    assert wd._well_types["gas"] == set(range(100))
    assert wd._well_types["shallow"] == set(range(50)).union(set(range(100, 150)))
    assert wd._well_types["deep"] == set(range(50, 100)).union(set(range(150, 200)))

    gas_oil_wells = wd.get_gas_oil_wells
    assert isinstance(gas_oil_wells["oil"], WellData)
    assert gas_oil_wells["oil"].data.shape[0] == 100
    assert isinstance(gas_oil_wells["gas"], WellData)
    assert gas_oil_wells["gas"].data.shape[0] == 100
    assert len(gas_oil_wells) == 2

    shallow_deep_wells = wd.get_shallow_deep_wells
    assert isinstance(shallow_deep_wells["shallow"], WellData)
    assert shallow_deep_wells["shallow"].data.shape[0] == 100
    assert isinstance(shallow_deep_wells["deep"], WellData)
    assert shallow_deep_wells["deep"].data.shape[0] == 100
    assert len(shallow_deep_wells) == 2

    full_partition_wells = wd.get_fully_partitioned_data
    assert len(full_partition_wells) == 4
    assert isinstance(full_partition_wells["deep_oil"], WellData)
    assert full_partition_wells["deep_oil"].data.shape[0] == 50
    assert list(full_partition_wells["deep_oil"].data.index) == list(range(150, 200))

    assert isinstance(full_partition_wells["shallow_oil"], WellData)
    assert full_partition_wells["shallow_oil"].data.shape[0] == 50
    assert list(full_partition_wells["shallow_oil"].data.index) == list(range(100, 150))

    assert isinstance(full_partition_wells["deep_gas"], WellData)
    assert full_partition_wells["deep_gas"].data.shape[0] == 50
    assert list(full_partition_wells["deep_gas"].data.index) == list(range(50, 100))

    assert isinstance(full_partition_wells["shallow_gas"], WellData)
    assert full_partition_wells["shallow_gas"].data.shape[0] == 50
    assert list(full_partition_wells["shallow_gas"].data.index) == list(range(50))

    # Write the data files to the temp directory
    deep_oil_file = str(tmp_path / "deep_oil.xlsx")
    full_partition_wells["deep_oil"].save_to_file(deep_oil_file)
    shallow_gas_file = str(tmp_path / "shallow_gas.csv")
    full_partition_wells["shallow_gas"].save_to_file(shallow_gas_file)

    assert os.path.exists(deep_oil_file)
    assert os.path.exists(shallow_gas_file)

    deep_gas_file = str(tmp_path / "deep_gas.foo")
    with pytest.raises(ValueError, match="Format .foo is not supported."):
        full_partition_wells["deep_gas"].save_to_file(deep_gas_file)

    # Check unrecognized well-type errors
    wd_df.loc[0, col_names.well_type] = "Foo"
    with pytest.raises(
        ValueError,
        match=(
            "Well-type must be either oil or gas or both. Received " "Foo in row 0."
        ),
    ):
        wd = WellData(data=wd_df, column_names=col_names)

    wd_df.loc[0, col_names.well_type] = "gas"
    wd_df.loc[0, col_names.well_type_by_depth] = "Foo"
    with pytest.raises(
        ValueError,
        match=(
            "Well-type by depth must be either shallow or deep. Received "
            "Foo in row 0."
        ),
    ):
        wd = WellData(data=wd_df, column_names=col_names)


def test_filter_annual_production(caplog, get_column_names):
    """Tests the _filter_ann_production method"""
    filename, col_names = get_column_names

    wd_df = pd.read_csv(filename, usecols=col_names.values())
    # Removing 100 wells with incomplete data
    wd_df = wd_df.drop(index=list(range(100))).reset_index()
    assert wd_df.shape[0] == 200

    # Ensure that nothing happens in the default dataset
    wd = WellData(
        data=wd_df, column_names=col_names, threshold_daily_production_boe=1000
    )
    assert (
        "wells whose annual production exceeds 1000.0 BOE per day."
    ) not in caplog.text
    assert "wells from the dataset." not in caplog.text
    assert (
        "Unable to filter wells based on threshold_daily_production_boe "
        "because the required data is not available."
    ) not in caplog.text
    assert len(wd) == 200
    assert 0 in wd._well_types["gas"]
    assert len(wd._removed_rows) == 0

    # Modify the dataset to remove the first well
    wd.data.loc[0, col_names.ann_oil_production] = 1e6
    wd._filter_ann_production()
    assert (
        "Found 1 wells whose annual production exceeds 1000.0 BOE per day."
    ) in caplog.text
    assert "Removed 1 wells from the dataset." in caplog.text
    assert (
        "Unable to filter wells based on threshold_daily_production_boe "
        "because the required data is not available."
    ) not in caplog.text
    assert len(wd) == 199
    assert 0 not in wd.data.index
    assert 0 not in wd._well_types["gas"]
    assert wd._removed_rows == {"exceeding daily production threshold": [0]}

    # Test data not available warning
    oil_col_name = col_names.ann_oil_production
    col_names.ann_oil_production = None
    wd = WellData(
        data=wd_df.drop(oil_col_name, axis=1),
        column_names=col_names,
        threshold_daily_production_boe=1000,
    )
    assert (
        "Unable to filter wells based on threshold_daily_production_boe "
        "because the required data is not available."
    ) in caplog.text
    assert len(wd) == 200


def test_compute_priority_scores(
    caplog, get_column_names
):  # pylint: disable=too-many-statements
    filename, col_names = get_column_names
    im_metrics = ImpactMetrics()
    im_metrics.set_weight(
        primary_metrics={
            "well_history": 30,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_age": 20,
            "well_count": 10,
        },
        submetrics={
            "well_history": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )
    wd = WellData(
        data=filename,
        column_names=col_names,
        fill_age=99,
        fill_depth=999,
        fill_life_gas_production=1.5,
        fill_life_oil_production=1.5,
        min_lifetime_oil_production=0,
        max_lifetime_oil_production=2,
        min_lifetime_gas_production=0,
        max_lifetime_gas_production=2,
        impact_metrics=im_metrics,
    )

    wd.compute_priority_scores()

    # Check warning messages
    assert f"Found empty cells in column {col_names.leak}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.leak} with value False."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.compliance}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.compliance} with value True."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.violation}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.violation} with value False."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.incident}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.incident} with value False."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.hospitals}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.hospitals} with value 0."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.schools}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.schools} with value 0."
        in caplog.text
    )

    # Annual gas and oil production data has been filled in the preliminary step
    assert f"Found empty cells in column {col_names.ann_gas_production}" in caplog.text
    assert f"Found empty cells in column {col_names.ann_oil_production}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.ann_gas_production} "
        f"with value 0.0"
    ) in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.ann_oil_production} "
        f"with value 0.0"
    ) in caplog.text

    # All cells in the leak column must be zero
    assert (wd.data[im_metrics.leak.score_col_name] == 0).all()

    # Compliance has inverse priority. Check if the score is calculated correctly
    compliant_row = 289  # Row that has a compliant well
    non_compliant_row = 290  # Row that has a non-compliant well
    assert np.isclose(
        wd.data.loc[compliant_row, im_metrics.compliance.score_col_name],
        0,
    )
    assert np.isclose(
        wd.data.loc[non_compliant_row, im_metrics.compliance.score_col_name],
        9.0,
    )

    # Violation is proportional to priority.
    # Check if the score is calculated correctly
    violation_row = 274  # Row that has a well that is in violation
    non_violation_row = 275  # Row that has a well that is not in violation
    assert np.isclose(
        wd.data.loc[violation_row, im_metrics.violation.score_col_name],
        6.0,
    )
    assert np.isclose(
        wd.data.loc[non_violation_row, im_metrics.violation.score_col_name],
        0,
    )

    # Num. Hospitals are proportional to priority
    hospital_row = 296
    assert np.isclose(
        wd.data.loc[hospital_row, im_metrics.hospitals.score_col_name],
        ((4 - 0) / (6 - 0)) * im_metrics.hospitals.effective_weight,
    )

    # Num. Schools are proportional to priority
    school_row = 297
    assert np.isclose(
        wd.data.loc[school_row, im_metrics.schools.score_col_name],
        ((3 - 0) / (6 - 0)) * im_metrics.schools.effective_weight,
    )

    # Annual gas and oil production volumes have inverse priority
    oil_production_row = 300
    gas_production_row = 301
    assert np.isclose(
        wd.data.loc[gas_production_row, im_metrics.ann_gas_production.score_col_name],
        ((981.6 - 63.01) / 981.6) * im_metrics.ann_gas_production.effective_weight,
    )
    assert np.isclose(
        wd.data.loc[oil_production_row, im_metrics.ann_oil_production.score_col_name],
        ((998.22 - 511.4) / 998.22) * im_metrics.ann_oil_production.effective_weight,
    )

    # Test errors thrown by the method
    # with efficiency refactoring, the priority score is
    # added as an attribute to the well column name object
    wd_df = pd.read_csv(
        filename,
        usecols=[
            col
            for col in col_names.values()
            if "Priority" not in col and "Rank" not in col
        ],
    )

    # Set a non-numeric value
    wd_df[col_names.hospitals] = wd_df[col_names.hospitals].astype("object")
    wd_df.loc[250, col_names.hospitals] = "NULL"
    # we now add the priority score column to the object when we compute
    col_names.priority_score = None
    col_names.well_rank = None

    with pytest.raises(
        ValueError,
        match=(
            f"Unable to compute scores for metric {im_metrics.hospitals.name}/"
            f"{im_metrics.hospitals.full_name},  because the column "
            f"{im_metrics.hospitals.data_col_name} "
            f"contains non-numeric values in rows \\[250\\]."
        ),
    ):
        wd = WellData(
            data=wd_df,
            column_names=col_names,
            impact_metrics=im_metrics,
        )
        wd.compute_priority_scores()

    # Error raised when the data for an unsupported metric is missing
    im_metrics.register_new_metric("my_metric", "My Custom Metric")
    im_metrics.well_age.weight = 15
    im_metrics.my_metric.weight = 5
    wd_df = pd.read_csv(
        filename, usecols=[col for col in col_names.values() if "Priority" not in col]
    )
    # have to reset the column added by computing the priority score
    col_names.priority_score = None
    with pytest.raises(
        ValueError,
        match=(
            "data_col_name attribute for metric my_metric/My Custom Metric "
            "is not provided."
        ),
    ):
        wd = WellData(data=wd_df, column_names=col_names, impact_metrics=im_metrics)
        wd.compute_priority_scores()


def test_compute_priority_scores_placeholders(
    caplog, get_column_names
):  # pylint: disable=too-many-statements
    with caplog.at_level(logging.INFO):
        filename, col_names = get_column_names
        im_metrics = ImpactMetrics()
        im_metrics.set_weight(
            primary_metrics={
                "Placeholder 1": 5,
                "Placeholder 2": 5,
                "Placeholder 3": 5,
                "Placeholder 4": 5,
                "Placeholder 5": 5,
                "Placeholder 6": 5,
                "Placeholder 7": 5,
                "Placeholder 8": 5,
                "Placeholder 9": 5,
                "Placeholder 10": 5,
                "Placeholder 11": 5,
                "Placeholder 12": 5,
                "Placeholder 13": 5,
                "Placeholder 14": 5,
                "Placeholder 15": 5,
                "Placeholder 16": 5,
                "Placeholder 17": 5,
                "Placeholder 18": 5,
                "Placeholder 19": 5,
                "Placeholder 20": 5,
            },
        )
        wd = WellData(
            data=filename,
            column_names=col_names,
            fill_age=99,
            fill_depth=999,
            fill_life_gas_production=1.5,
            fill_life_oil_production=1.5,
            min_lifetime_oil_production=0,
            max_lifetime_oil_production=2,
            min_lifetime_gas_production=0,
            max_lifetime_gas_production=2,
            impact_metrics=im_metrics,
        )

        wd.compute_priority_scores()

    # select random wells to use as anchors
    priority_scores = {
        19212: 47.130415892975115,
        90622: 58.85208210407261,
        86849: 56.06847745565535,
        18898: 43.096345707212976,
        56609: 51.3592130219126,
    }

    for index, row in wd.data.iterrows():
        if int(row[col_names.well_id]) in priority_scores:
            assert priority_scores[int(row[col_names.well_id])] == pytest.approx(
                row["Priority Score [0-100]"]
            )


def test_compute_priority_scores_custom_basis(get_column_names):
    filename, col_names = get_column_names
    im_metrics = ImpactMetrics(basis=1000)
    im_metrics.set_weight(
        primary_metrics={
            "well_history": 300,
            "sensitive_receptors": 200,
            "ann_production_volume": 200,
            "well_age": 200,
            "well_count": 100,
        },
        submetrics={
            "well_history": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )
    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
    )

    wd.compute_priority_scores()

    assert wd.col_names.priority_score == "Priority Score [0-1000]"
    assert wd.data["Priority Score [0-1000]"].max() <= 1000
    assert (wd.data["Priority Score [0-1000]"] > 100).any()

    assert wd.col_names.priority_score_normalized == "Priority Score [0-100]"
    assert wd.data["Priority Score [0-100]"].max() <= 100
    assert wd.data["Priority Score [0-100]"].min() >= 0


@pytest.fixture(name="get_column_names_recommendation_only", scope="function")
def get_column_names_recommendation_only_fixture():
    """Returns well data from a csv file and scenario type"""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        additional_columns={"priority_score": "Priority Score"},
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_recommendation_only.csv",
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=True, project_comparison=False
    )

    return filename, col_names, scenario_type


def test_compute_priority_score(get_column_names_recommendation_only):
    """Test the WellData class under the ranking only scenario"""

    filename, col_names, scenario_type = get_column_names_recommendation_only

    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=None,
        scenario_type=scenario_type,
    )

    # Test column_names alias
    assert wd.col_names is wd.column_names

    assert not hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score == "Priority Score"

    wd.compute_priority_scores()

    assert hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score_normalized == "Priority Score [0-100]"

    priority_scores = {
        19212: 35,
        90622: 74,
        86849: 95,
        18898: 94,
        56609: 27,
    }

    for index, row in wd.data.iterrows():
        if int(row[col_names.well_id]) in priority_scores:
            assert priority_scores[int(row[col_names.well_id])] == pytest.approx(
                row["Priority Score [0-100]"]
            )


def test_compute_priority_score_reverse(get_column_names_recommendation_only):
    """Test the WellData class under the ranking only scenario where wells with
    lower priority score receive higher ranking"""

    filename, col_names, scenario_type = get_column_names_recommendation_only

    wd = WellData(
        data=filename,
        column_names=col_names,
        scenario_type=scenario_type,
        priority_score_reverse=True,
    )

    # Test column_names alias
    assert wd.col_names is wd.column_names

    assert not hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score == "Priority Score"

    wd.compute_priority_scores()

    assert hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score_normalized == "Priority Score [0-100]"

    priority_scores_raw = {
        19212: 35,
        90622: 74,
        86849: 95,
        18898: 94,
        56609: 27,
    }

    priority_scores_normalized = {
        19212: 65,
        90622: 26,
        86849: 5,
        18898: 6,
        56609: 73,
    }

    for index, row in wd.data.iterrows():
        if int(row[col_names.well_id]) in priority_scores_raw:
            assert priority_scores_raw[int(row[col_names.well_id])] == pytest.approx(
                row["Priority Score"], 1e-2
            )
            assert priority_scores_normalized[
                int(row[col_names.well_id])
            ] == pytest.approx(row[col_names.priority_score_normalized], 1e-2)


@pytest.fixture(name="get_column_names_recommendation_only_over_100", scope="function")
def get_column_names_recommendation_only_over_100_fixture():
    """Returns well data from a csv file and scenario type"""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        additional_columns={"priority_score": "Priority Score Raw"},
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_recommendation_only.csv",
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=True, project_comparison=False
    )

    return filename, col_names, scenario_type


def test_set_priority_score_over_100(get_column_names_recommendation_only_over_100):
    """Test the WellData class under the ranking-only scenario where
    user-provided priority scores are outside the [0.0, 100.0] range"""

    filename, col_names, scenario_type = get_column_names_recommendation_only_over_100

    wd = WellData(
        data=filename,
        column_names=col_names,
        scenario_type=scenario_type,
    )

    # Test column_names alias
    assert wd.col_names is wd.column_names

    assert not hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score == "Priority Score Raw"

    wd.compute_priority_scores()

    assert hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score_normalized == "Priority Score [0-100]"
    assert col_names.priority_score == "Priority Score Raw"

    priority_scores_raw = {
        19212: 45.5,
        90622: 96.2,
        86849: 123.5,
        18898: 122.2,
        56609: 35.1,
    }

    priority_scores_normalized = {
        19212: 35,
        90622: 74,
        86849: 95,
        18898: 94,
        56609: 27,
    }

    # Test if the provided priority score are normalized
    for index, row in wd.data.iterrows():
        if int(row[col_names.well_id]) in priority_scores_normalized:
            assert priority_scores_normalized[
                int(row[col_names.well_id])
            ] == pytest.approx(row[col_names.priority_score_normalized], 1e-2)
            assert priority_scores_raw[int(row[col_names.well_id])] == pytest.approx(
                row[col_names.priority_score], 1e-2
            )


def test_compute_priority_score_over_100_reverse(
    get_column_names_recommendation_only_over_100,
):
    """Test the WellData class under the ranking-only scenario where
    user-provided priority scores are outside the [0.0, 100.0] range and
    a higher score indicates a lower-priority well."""

    filename, col_names, scenario_type = get_column_names_recommendation_only_over_100

    wd = WellData(
        data=filename,
        column_names=col_names,
        scenario_type=scenario_type,
        priority_score_reverse=True,
    )

    # Test column_names alias
    assert wd.col_names is wd.column_names

    assert not hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score == "Priority Score Raw"

    wd.compute_priority_scores()

    assert hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score_normalized == "Priority Score [0-100]"
    assert col_names.priority_score == "Priority Score Raw"

    priority_scores_raw = {
        19212: 45.5,
        90622: 96.2,
        86849: 123.5,
        18898: 122.2,
        56609: 35.1,
        77459: 130,
        45595: 0,
    }

    priority_scores_normalized = {
        19212: 65,
        90622: 26,
        86849: 5,
        18898: 6,
        56609: 73,
        77459: 0,
        45595: 100,
    }

    # Test the priority score calculation in the reverse pattern
    for index, row in wd.data.iterrows():
        if int(row[col_names.well_id]) in priority_scores_normalized:
            assert priority_scores_normalized[
                int(row[col_names.well_id])
            ] == pytest.approx(row[col_names.priority_score_normalized], 1e-2)
            assert priority_scores_raw[int(row[col_names.well_id])] == pytest.approx(
                row[col_names.priority_score], 1e-2
            )

    # Test priority score calculation with reverse ranking logic
    top_wells = wd.get_high_priority_wells(5)
    assert (
        top_wells[col_names.priority_score].tolist()
        == sorted(wd[col_names.priority_score], reverse=False)[:5]
    )
    assert (
        top_wells[col_names.priority_score_normalized].tolist()
        == sorted(wd[col_names.priority_score_normalized], reverse=True)[:5]
    )


def test_missing_priority_score():
    """This intends to test the scenario where well ranking is not performed
    and the priority scores provided by users have missing scores."""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        additional_columns={"priority_score": "Priority Score Missing"},
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_recommendation_only.csv",
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=True, project_comparison=False
    )

    well_priority_score_missing = [84139, 58006, 28545, 17853, 41399, 90932]

    assert col_names.priority_score == "Priority Score Missing"

    with pytest.raises(
        ValueError,
        match=(
            "The priority score of the following wells are missing:"
            f"{well_priority_score_missing}."
        ),
    ):
        wd = WellData(
            data=filename,
            column_names=col_names,
            scenario_type=scenario_type,
        )


def test_no_priority_score():
    """This intends to test the scenario where well ranking is not performed
    and priority scores are not provided."""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_recommendation_only.csv",
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=True, project_comparison=False
    )

    assert not hasattr(col_names, "priority_score")

    with pytest.raises(
        ValueError,
        match=(
            "Priority scores for wells were not provided. Since the well ranking "
            "capability is not enabled, please provide the priority scores."
        ),
    ):
        wd = WellData(
            data=filename,
            column_names=col_names,
            scenario_type=scenario_type,
        )


def test_negative_priority_score():
    """This intends to test the scenario where priority scores provide by users
    contain negative values."""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        additional_columns={"priority_score": "Priority Score Negative"},
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_recommendation_only.csv",
    )

    scenario_type = ScenarioType(
        well_ranking=False, project_recommendation=True, project_comparison=False
    )

    wd = WellData(
        data=filename,
        column_names=col_names,
        scenario_type=scenario_type,
        priority_score_reverse=True,
    )

    assert wd.col_names is wd.column_names

    assert not hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score == "Priority Score Negative"

    with pytest.raises(ValueError, match="Priority scores must be >= 0."):
        wd.compute_priority_scores()


@pytest.mark.parametrize(
    "column_name,raw_value,expected_value,expected_max_value,scenario_flags",
    [
        ("Priority Score Zeros", 0, 0, 0, [False, True, False]),
        ("Priority Score Uniform Less Than 100", 56, 100, 56, [False, True, False]),
        (
            "Priority Score Uniform Greater Than 100",
            128,
            100,
            128,
            [False, True, False],
        ),
        ("Priority Score Zeros", 0, 0, 0, [True, True, False]),
        ("Priority Score Uniform Less Than 100", 56, 100, 56, [False, False, True]),
    ],
)
def test_uniform_priority_score(
    column_name,
    raw_value,
    expected_value,
    expected_max_value,
    scenario_flags,
    caplog,
):
    """This intends to test the scenario where priority scores provide by users
    are the same for all wells."""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        additional_columns={"priority_score": column_name},
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_recommendation_only.csv",
    )

    well_ranking = scenario_flags[0]
    project_recommendation = scenario_flags[1]
    project_comparison = scenario_flags[2]

    scenario_type = ScenarioType(
        well_ranking=well_ranking,
        project_recommendation=project_recommendation,
        project_comparison=project_comparison,
    )

    wd = WellData(
        data=filename,
        column_names=col_names,
        scenario_type=scenario_type,
    )

    assert wd.col_names is wd.column_names

    assert not hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score == column_name

    wd.compute_priority_scores()

    assert wd.get_maximum_priority_score == expected_max_value

    assert hasattr(col_names, "priority_score_normalized")
    assert col_names.priority_score_normalized == "Priority Score [0-100]"
    assert col_names.priority_score == column_name

    for _, row in wd.data.iterrows():
        assert row[col_names.priority_score] == pytest.approx(raw_value, 1e-2)
        assert row[col_names.priority_score_normalized] == pytest.approx(
            expected_value, 1e-2
        )

    if (project_recommendation and not well_ranking) or project_comparison:
        assert (
            "The provided priority scores are identical across all wells."
        ) in caplog.text
    else:
        assert (
            "The provided priority scores are identical across all wells."
        ) not in caplog.text


def test_fully_incomplete_data_check(get_column_names):
    """
    This test method is to check to make sure the fully_incomplete_data_check
    is working properly
    """
    filename, col_names = get_column_names

    df = pd.read_csv(filename)
    df["Placeholder 20"] = np.nan
    df.loc[223, "Placeholder 19"] = np.nan
    wd = WellData(data=df, column_names=col_names)

    assert wd.has_fully_incomplete_data("Placeholder 20") is True

    assert wd.has_fully_incomplete_data("Placeholder 19") is False

    assert wd.has_fully_incomplete_data("Placeholder 18") is False


@pytest.mark.parametrize(
    "scenario_combination,status",
    [
        ((True, False, False), True),
        ((False, True, False), True),
        ((True, True, False), True),
        ((True, False, True), False),
        ((False, True, True), False),
        ((True, True, True), False),
    ],
)
def test_scenario_type_combination(scenario_combination, status):
    if status:
        ScenarioType(*scenario_combination)
    else:
        with pytest.raises(
            ValueError,
            match=(
                r"Project comparison cannot be used together with well "
                r"ranking or project recommendation."
            ),
        ):
            ScenarioType(*scenario_combination)


@pytest.fixture(name="get_column_names_custom_impact_factor", scope="function")
def get_column_names_fixture_custom_impact_factor():
    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="y",
        longitude="x",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        hospitals="Hospitals",
        schools="Schools",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        elevation_delta="Elevation Delta [m]",
        dist_to_road="Distance to Road [miles]",
    )
    col_names.register_new_columns(
        {
            "endangered_species": "Endangered Species [Yes/No]",
            "number_of_heavy_equipment": "Number of Heavy Equipment",
        }
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_custom_impact_factors.csv",
    )

    return col_names, filename


@pytest.fixture(
    params=[
        {"endangered_species_inverse": False, "equipment_inverse": True},
        {"endangered_species_inverse": True, "equipment_inverse": False},
    ],
    name="im_metric",
)
def im_metric_fixture(request):
    config = request.param

    im_metrics = ImpactMetrics()

    im_metrics.register_new_metric(
        name="endangered_species",
        full_name="Endangered Species [Yes/No]",
        is_binary_type=True,
        has_inverse_priority=config["endangered_species_inverse"],
        data_col_name="Endangered Species [Yes/No]",
    )
    im_metrics.register_new_metric(
        name="number_of_heavy_equipment",
        full_name="Number of Heavy Equipment",
        is_binary_type=False,
        has_inverse_priority=config["equipment_inverse"],
        data_col_name="Number of Heavy Equipment",
    )

    # Set weights
    im_metrics.set_weight(
        primary_metrics={
            "well_history": 35,
            "sensitive_receptors": 10,
            "ann_production_volume": 10,
            "well_age": 15,
            "well_count": 10,
            "endangered_species": 10,
            "number_of_heavy_equipment": 10,
        },
        submetrics={
            "well_history": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )

    return im_metrics, config["endangered_species_inverse"], config["equipment_inverse"]


def test_filling_missing_data_when_no_default(
    get_column_names_custom_impact_factor, im_metric
):

    col_names, filename = get_column_names_custom_impact_factor
    im_metrics, endangered_species_label, heavy_equipment_label = im_metric

    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
    )

    if heavy_equipment_label:
        max_number_of_heavy_equipment_value = wd.data["Number of Heavy Equipment"].max()
        assert wd.data[wd.data["API Well Number"] == "15586"][
            "Number of Heavy Equipment"
        ].iloc[0] == pytest.approx(max_number_of_heavy_equipment_value)
        assert wd.data[wd.data["API Well Number"] == "67270"][
            "Number of Heavy Equipment"
        ].iloc[0] == pytest.approx(max_number_of_heavy_equipment_value)
    else:
        min_number_of_heavy_equipment_value = wd.data["Number of Heavy Equipment"].min()
        assert wd.data[wd.data["API Well Number"] == "15586"][
            "Number of Heavy Equipment"
        ].iloc[0] == pytest.approx(min_number_of_heavy_equipment_value)
        assert wd.data[wd.data["API Well Number"] == "67270"][
            "Number of Heavy Equipment"
        ].iloc[0] == pytest.approx(min_number_of_heavy_equipment_value)

    if endangered_species_label:
        assert wd.data[wd.data["API Well Number"] == "76973"][
            "Endangered Species [Yes/No]"
        ].iloc[0] == pytest.approx(1)
        assert wd.data[wd.data["API Well Number"] == "61821"][
            "Endangered Species [Yes/No]"
        ].iloc[0] == pytest.approx(1)
    else:
        assert wd.data[wd.data["API Well Number"] == "76973"][
            "Endangered Species [Yes/No]"
        ].iloc[0] == pytest.approx(0)
        assert wd.data[wd.data["API Well Number"] == "61821"][
            "Endangered Species [Yes/No]"
        ].iloc[0] == pytest.approx(0)
