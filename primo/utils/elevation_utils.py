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
from typing import Optional, Tuple

# Installed libs
import numpy as np
import pandas as pd
import rasterio
from pyproj import Proj, transform
from routingpy import Valhalla
from routingpy.direction import Direction

# User-defined libs
from primo.data_parser.default_data import COORDINATE_ADJUSTMENT as coordinate_adj
from primo.utils.geo_utils import get_distance
from primo.utils.raise_exception import raise_exception


# pylint: disable = too-many-arguments,too-many-positional-arguments
def get_route(
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
    profile: str = "auto",
    preference: str = "shortest",
    retry_over_query_limit: bool = True,
    retry_timeout: int = 120,
) -> Direction:
    """
    Get the route between two coordinates using the routingpy package to send requests
    to the Valhalla routing engine.
    For more details on routingpy see: https://routingpy.readthedocs.io/en/latest/
    For more details on Valhalla see: https://valhalla.github.io/valhalla/mjolnir/data_sources/

    Parameters
    ----------
    start_coord : Tuple[float, float]
        The coordinates of the starting point
    end_coord : Tuple[float, float]
        The coordinates of the ending point
    profile: str
        Specifies the mode of transport to use when calculating directions.
        One of ["auto", "auto_shorter" (deprecated), "bicycle", "bus", "hov",
        "motor_scooter", "motorcycle", "multimodal", "pedestrian"].
    preferences: str
        Specifies the routing preference.
        One of ["shortest", "fastest"].
    retry_over_query_limit: Bool
        If True, client will not raise an exception
        on HTTP 429 (request limit error), but instead
        jitter a sleeping timer to pause between
        requests until HTTP 200 or retry_timeout is
        reached.
    retry_timeout: int
        Time in seconds for the client will pause
        for between requests when it receives a HTTP 429
        error.

    Returns
    -------
    Route: Direction
        Detailed route information if available; otherwise None

    Raises
    ------
    ValueError
        In case of invalid input arguments
    """
    if any(np.isnan(x) for x in start_coord):
        raise ValueError("Start coordinates received contain None values")

    if any(np.isnan(x) for x in end_coord):
        raise ValueError("End coordinates received contain None values")

    locations = [start_coord, end_coord]  # Define the locations for the route.

    client = Valhalla(
        retry_over_query_limit=retry_over_query_limit, retry_timeout=retry_timeout
    )  # Initialize the class for sending requests to the Valhalla client.

    route = client.directions(  # Calculate the route
        locations=locations,
        profile=profile,
        preference=preference,
    )

    return route


def get_nearest_road_point(
    lat: float,
    long: float,
) -> Tuple[float, float]:
    """
    Get the nearest road point to a given latitude and longitude using the Bing Maps API.

    Parameters
    ----------
    lat : float
        Latitude of the point
    long : float
        Longitude of the point

    Returns
    -------
    Tuple[float, float]
        The latitude and longitude of the nearest road point

    Raises
    ------
    ValueError
        In case of invalid input arguments
    """
    end_coordinates = [long, lat]  # well

    if any(np.isnan(x) for x in end_coordinates):
        raise ValueError("Coordinates received contain None values")

    # Create arbitrary starting point
    start_coord = [end_coordinates[0], end_coordinates[1] + coordinate_adj]
    # Get detailed route information
    route = get_route(start_coord, end_coordinates)
    # Reversing the order of the coordinates, (long, lat) -> (lat,long),
    # so it's in a form that other functions (e.g. accessibility) is expecting.
    return tuple(reversed(route.geometry[-1]))


def accessibility(lat: float, long: float) -> float:
    """
    Calculate the accessibility quotient for a given latitude and longitude.

    Parameters
    ----------
    lat : float
        Latitude of the location
    long : float
        Longitude of the location

    Returns
    -------
    float
        The accessibility quotient
    """

    closest_road_point = get_nearest_road_point(lat, long)
    accessibility_quotient = get_distance(
        (closest_road_point[0], closest_road_point[1]),
        (lat, long),
        "haversine",
        "MILES",
    )

    return accessibility_quotient


def get_elevation(lat: float, long: float, tif_file_path: str) -> Optional[float]:
    """
    Get the elevation at a given latitude and longitude using a specified GeoTIFF file.

    Parameters
    ----------
    lat : float
        Latitude of the location
    long : float
        Longitude of the location
    tif_file_path : str
        The path to the GeoTIFF file containing elevation data

    Returns
    -------
    Optional[float]
        The elevation at the specified location if available; otherwise None
    """
    try:
        # Define the source and destination coordinate systems
        src_crs = Proj(init="epsg:4326")  # WGS84
        dst_crs = Proj(init="epsg:5070")  # EPSG:5070

        # Transform latitude and longitude to the projected CRS
        long_transform, lat_transform = transform(src_crs, dst_crs, long, lat)

        with rasterio.open(tif_file_path) as src:
            # Transform latitude and longitude to pixel coordinates
            row, col = map(int, src.index(long_transform, lat_transform))

            # Read elevation data at the specified location
            elevation = src.read(1, window=((row, row + 1), (col, col + 1)))
            # Check if the value is NoData
            if elevation[0][0] == src.nodatavals[0]:
                return None  # NoData value, elevation not available

            return elevation[0][0]  # Return the elevation value

    except IndexError:
        msg = (
            "Empty array after reading elevation data from source file. "
            "Latitude and longitude values are not within bounds of the region of "
            "the raster file given. "
            f"Returning None for these coordinates - Latitude:{lat}, Longitude:{long}"
        )
        raise_exception(msg, IndexError)
        return None


def get_elevation_delta(df: pd.DataFrame, tif_file_path: str) -> pd.DataFrame:
    """
    Calculate the elevation delta for each location in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing latitude and longitude coordinates
    tif_file_path : str
        The path to the GeoTIFF file containing elevation data

    Returns
    -------
    pd.DataFrame
        The DataFrame with an added column 'elevation_delta' indicating the elevation delta
        for each location
    """
    deltas = []
    for _, row in df.iterrows():
        closest_road_point_lat, closest_road_point_long = get_nearest_road_point(
            row["Latitude"], row["Longitude"]
        )
        elevation = get_elevation(row["Latitude"], row["Longitude"], tif_file_path)
        closest_road_elevation = get_elevation(
            closest_road_point_lat, closest_road_point_long, tif_file_path
        )
        if elevation is None or closest_road_elevation is None:
            delta = np.nan
        else:
            delta = elevation - closest_road_elevation
        deltas.append(delta)

    df["Elevation Delta [m]"] = deltas
    return df
