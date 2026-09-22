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
import os

# Installed libs
import pytest

# User-defined libs
from primo.utils.demo_utils import (
    file_path_widget,
    file_upload_widget,
    generate_configurations,
)


def test_file_path_widget():
    file_path_widget(
        "Test Widget",
        ".csv",
        "File Selected",
        "primo\\utils\\tests\\screening_data.csv",
    )


def test_file_upload_widget():
    file_upload_widget(
        "Test Widget",
        ".csv",
        "File Uploaded",
    )


@pytest.mark.parametrize(
    "weights_file_path, config_path, output_folder_path, life_prod_values,"
    "owner_well_counts, status",
    [  # Case 1: Pass case
        (
            "weights_toy_example.xlsx",
            "config_test.json",
            "..\\config_output",
            [1000],
            [5],
            True,
        ),
        # Case 2: Wrong path to the weight scenario Excel file
        (
            "wrong_file.xlsx",
            "config_test.json",
            "..\\config_output",
            [1000],
            [5],
            False,
        ),
        # Case 3: Wrong path to the base config file
        (
            "weights_toy_example.xlsx",
            "config_wrong.json",
            "..\\config_output",
            [1000],
            [5],
            False,
        ),
        # Case 4: Pass case - no lifelong production constraint
        (
            "weights_toy_example.xlsx",
            "config_test.json",
            "..\\config_output",
            [],
            [5],
            True,
        ),
    ],
)
def test_generate_configurations(
    weights_file_path,
    config_path,
    output_folder_path,
    life_prod_values,
    owner_well_counts,
    status,
):
    directory = os.path.dirname(os.path.abspath(__file__))
    weights_file_full_path = os.path.join(directory, weights_file_path)
    config_full_path = os.path.join(directory, config_path)
    if status:
        generate_configurations(
            weights_file_full_path,
            config_full_path,
            output_folder_path,
            life_prod_values,
            owner_well_counts,
        )
    elif status is None:
        with pytest.raises(ValueError):
            generate_configurations(
                weights_file_full_path,
                config_full_path,
                output_folder_path,
                life_prod_values,
                owner_well_counts,
            )
    else:
        with pytest.raises(FileNotFoundError):
            generate_configurations(
                weights_file_full_path,
                config_full_path,
                output_folder_path,
                life_prod_values,
                owner_well_counts,
            )
