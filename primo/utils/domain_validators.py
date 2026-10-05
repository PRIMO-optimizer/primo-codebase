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

"""
This module contains functions that can be used for domain validation
with the `ConfigDict()` data structure from Pyomo.
"""

# Standard libs
from types import SimpleNamespace
from typing import Dict

# Installed libs
import numpy as np
from pyomo.common.config import NonNegativeFloat

# User-defined libs
from primo.utils.opt_utils import in_bounds


# pylint: disable-next = invalid-name
def InRange(lb, ub):
    """
    Domain validator for 1D compact sets.

    Parameters
    ----------
    lb : float
        Lower bound

    ub : float
        Upper bound

    Returns
    -------
    _in_range :
        Pointer to a domain validator function
    """

    def _in_range(val):
        if in_bounds(val, lb, ub, 0.0):
            return val
        raise ValueError(f"Value {val} lies outside the admissible range [{lb}, {ub}]")

    return _in_range


def validate_mobilization_cost(data: Dict[int, float]):
    """
    Validates the mobilization cost data

    Parameters
    ----------
    data : dict
        key => number of wells, value => cost of plugging

    Returns
    -------
    data
        Validated mobilization cost data.
    """
    for k in range(1, len(data) + 1):
        if k not in data:
            raise KeyError(f"Mobilization cost for {k} wells is not provided.")
    for key, val in data.items():
        data[key] = NonNegativeFloat(val)
    return data


def is_valid_zone_data(data: dict):
    """
    Domain validator for zone data for efficiency metrics

    Parameters
    ----------
    data : dict
        Keys correspond to the zone's outer boundary, and the values
        correspond to the corresponding zonal efficiency (as a fraction)
    """
    points = list(data.keys())
    efficiency = list(data.values())

    # Check if zones are in ascending order or not
    if not np.allclose(points, sorted(points)):
        raise ValueError(f"Zones in {data} are not in ascending order")

    # Raise an error if the zone length is zero
    for index in range(len(points) - 1):
        if np.isclose(points[index], points[index + 1]):
            raise ValueError(f"Zones {index + 1} and {index + 2} are identical!")

    # Check if all efficiencies are less than 1
    if any(np.array(efficiency) < 0) or any(np.array(efficiency) > 1):
        raise ValueError("Received an efficiency value outside [0, 1] interval")

    # Check if all efficiencies are arranged in descending order
    if not np.allclose(efficiency, sorted(efficiency, reverse=True)):
        raise ValueError("Zonal efficiencies are not in descending order")

    # Efficiency of the first zone must be 1
    if not np.isclose(efficiency[0], 1):
        raise ValueError(f"Efficiency of the first zone {efficiency[0]} is not 1")

    # If the zero efficiency zone is not specified, then set it to infinity.
    if efficiency[-1] != 0:
        data[float("inf")] = 0

    set_of_zones = list(range(1, len(data) + 1))

    zone_data = {
        "data": data,
        "zones": set_of_zones,
        "points": [0] + list(data.keys()),
        "efficiency": dict(zip(set_of_zones, data.values())),
    }

    return SimpleNamespace(**zone_data)


def is_valid_inverse_priority_zone_data(data: dict):
    """
    Domain validator for zone data for efficiency metrics
    with inverse priority i.e., a higher value ==> higher efficiency

    Parameters
    ----------
    data : dict
        Keys correspond to the zone's outer boundary, and the values
        correspond to the corresponding zonal efficiency (as a fraction)
    """

    # Take complement of the data and use is_valid_zone_data
    comp_data = {k: 1 - v for k, v in data.items()}
    zone_data = is_valid_zone_data(comp_data)

    # Take another complement to get the correct values
    zone_data.data = {k: 1 - v for k, v in zone_data.data.items()}
    zone_data.efficiency = {k: 1 - v for k, v in zone_data.efficiency.items()}

    return zone_data
