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

# Installed libs
import matplotlib.pyplot as plt
import pandas as pd

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils.map_utils import get_cluster_colors


def visualize_gas_oil_wells(gas_wells: WellData, oil_wells: WellData):
    """Generates an x-y plot for oil wells and gas wells separately"""

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot gas wells
    gas_cn = gas_wells.column_names  # Column names
    ax1.scatter(
        gas_wells[gas_cn.longitude], gas_wells[gas_cn.latitude], c="r", marker="o"
    )
    ax1.set_title("Gas Wells")
    ax1.set(xlabel="x-coordinate of wells", ylabel="y-coordinate of wells")

    # Plot oil wells
    oil_cn = oil_wells.column_names  # Column names
    ax2.scatter(
        oil_wells[oil_cn.longitude], oil_wells[oil_cn.latitude], c="b", marker="X"
    )
    ax2.set_title("Oil Wells")
    ax2.set(xlabel="x-coordinate of wells", ylabel="y-coordinate of wells")
    plt.show()


def visualize_clusters(wells: WellData, well_type: str = "Gas"):
    """Generates an x-y plot for clusters of wells"""
    cluster_col_name = "Clusters"
    col_names = wells.column_names

    color_list = get_cluster_colors(pd.unique(wells[cluster_col_name]))

    # Scatter plot of clusters
    for cluster in color_list:
        well_cluster = wells[wells[cluster_col_name] == cluster]
        plt.scatter(
            well_cluster[col_names.longitude],
            well_cluster[col_names.latitude],
            c=color_list[cluster],
        )

    plt.title(f"Clusters of {well_type} wells")
    plt.xlabel("x-coordinate of wells")
    plt.ylabel("y-coordinate of wells")
    plt.show()
