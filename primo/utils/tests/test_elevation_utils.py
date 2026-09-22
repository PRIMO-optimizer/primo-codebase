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

# pylint: disable=missing-function-docstring
# Installed libs
import numpy as np
import pytest
from routingpy.direction import Direction

# User-defined libs
from primo.utils.elevation_utils import (
    accessibility,
    get_nearest_road_point,
    get_route,
)


@pytest.mark.parametrize(
    "start_coord,end_coord, status",
    [  # Case 1: pass case
        (
            [-79.715395, 40.642704],
            [-79.715295, 40.642804],
            True,
        ),
        # Case 2: missing data
        (
            [-79.715395, 40.642704],
            [-79.715295, np.nan],
            False,
        ),
        # Case 3: missing data
        (
            [-79.715395, np.nan],
            [-79.715295, 40.642804],
            False,
        ),
        # Add more test cases as needed
    ],
)
def test_get_route(start_coord, end_coord, status):
    if status:
        assert isinstance(get_route(start_coord, end_coord), Direction)
        assert hasattr(get_route(start_coord, end_coord), "geometry")
    else:
        with pytest.raises(ValueError):
            get_route(start_coord, end_coord)


@pytest.mark.parametrize(
    "lat, lon, return_tuple, status",
    [  # Case 1: pass case
        (
            40.589335,
            -79.92741,
            (40.589202, -79.927406),
            True,
        ),
        # Case 2: missing data
        (
            np.nan,
            -79.92741,
            "Error",
            False,
        ),
        # Add more test cases as needed
    ],
)
def test_get_nearest_road_point(lat, lon, return_tuple, status):
    if status:
        assert np.allclose(
            get_nearest_road_point(lat, lon),
            return_tuple,
            rtol=1e-5,
            atol=1e-8,
        )
    else:
        with pytest.raises(ValueError):
            get_nearest_road_point(lat, lon)


@pytest.mark.parametrize(
    "lat, lon, return_value, status",
    [  # Case 1: pass case
        (40.4309, -79.739699, 0.02113937013190853, True),
        # Case 2: missing data
        (
            np.nan,
            -79.92741,
            "Error",
            False,
        ),
        # Add more test cases as needed
    ],
)
def test_accessibility(lat, lon, return_value, status):
    if status:
        assert np.isclose(
            accessibility(lat, lon),
            return_value,
            rtol=1e-5,
            atol=1e-8,
        )
    else:
        with pytest.raises(ValueError):
            accessibility(lat, lon)
