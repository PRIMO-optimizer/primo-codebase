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
import datetime
import logging
from dataclasses import dataclass
from typing import Optional, Union

# Installed libs
from pyomo.common.config import Bool, NonNegativeFloat, NonNegativeInt

LOGGER = logging.getLogger(__name__)

# This file should contain all the constants
FEASIBILITY_TOLERANCE = 1e-6  # Optimization Feasibility tolerance
EARTH_RADIUS = 3959.0  # Earth's radius in Miles
CENSUS_YEAR = 2020
CONVERSION_FACTOR = 5.614583  # convert Bbl of oil to Mcf of gas
# The adjust the end coordinates by to make an arbitrary starting coordinate 2 miles away.
COORDINATE_ADJUSTMENT = 0.0289659
# default value for max num wells and max num unique owners scaling factor in efficiency model
DEFAULT_MAX_NUM_WELLS = 25
DEFAULT_MAX_NUM_UNIQUE_OWNERS = 5

# Ensure that census year is as recent as possible
if datetime.date.today().year - CENSUS_YEAR > 10:
    LOGGER.warning(f"Package is using {CENSUS_YEAR} CENSUS Data by default")
    LOGGER.warning("Consider upgrading to newer version")

# sorting metrics for projects
SORTING_METRICS = [
    "average_impact_score",
    "efficiency_score",
    "total_impact_score",
    "num_wells",
    "plugging_cost",
]

# list of efficiency metrics accessed in the efficiency model
WELL_BASED_METRICS = [
    "elevation_delta",
    "dist_to_road",
    "population_density",
    "record_completeness",
]

# list of pairwise efficiency metrics accessed in the efficiency model
WELL_PAIR_METRICS = ["age_range", "depth_range", "dist_range"]

# List of columns for which the user has an option to
# either remove/fill if the data is missing
MISSING_INPUT_DATA_COLUMNS = [
    "age",
    "depth",
    "well_type",
    "ann_gas_production",
    "ann_oil_production",
    "life_gas_production",
    "life_oil_production",
]


# Set of supported impact metrics along with
# the required data for the analysis.
# pylint: disable = too-many-instance-attributes
@dataclass()
class _SupportedContent:
    name: str
    full_name: str
    has_submetrics: bool = False
    is_submetric: bool = False
    parent_metric: Optional[str] = None
    # required_data: keys in WellDataColumnNames class
    # This will be used to check if the input data has
    # required columns or not.
    required_data: Optional[Union[str, list]] = None
    # Is the value of this metric inversely proportional to plugging priority?
    # E.g., Compliance, production volume, etc.
    has_inverse_priority: bool = False
    fill_missing_value: Optional[dict] = None


WELL_HISTORY_METRICS = {
    "well_history": _SupportedContent(
        name="well_history",
        full_name="Well History",
        has_submetrics=True,
    ),
    "compliance": _SupportedContent(
        name="compliance",
        full_name="Compliance [Yes/No]",
        is_submetric=True,
        parent_metric="well_history",
        required_data="compliance",
        # Priority should be higher if the well is not compliant
        has_inverse_priority=True,
        # Assuming well is compliant if it is not specified
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "incident": _SupportedContent(
        name="incident",
        full_name="Incident [Yes/No]",
        is_submetric=True,
        parent_metric="well_history",
        required_data="incident",
        # Incident is assumed to be False if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "leak": _SupportedContent(
        name="leak",
        full_name="Leak [Yes/No]",
        is_submetric=True,
        parent_metric="well_history",
        required_data="leak",
        # Assume that the well is not leaking if it not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "violation": _SupportedContent(
        name="violation",
        full_name="Violation [Yes/No]",
        is_submetric=True,
        parent_metric="well_history",
        required_data="violation",
        # Assuming that the well is not in violation
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "hydrocarbon_losses": _SupportedContent(
        name="hydrocarbon_losses",
        full_name="Hydrocarbon Loss Rate [Ton/Year]",
        is_submetric=True,
        parent_metric="well_history",
        required_data="hydrocarbon_losses",
        # Assuming that there is no hydrocarbon loss
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
}

ECOLOGICAL_RECEPTORS = {
    "ecological_receptors": _SupportedContent(
        name="ecological_receptors",
        full_name="Ecological Receptors",
        has_submetrics=True,
    ),
    "endangered_species_on_site": _SupportedContent(
        name="endangered_species_on_site",
        full_name="Endangered Species on Site [Yes/No]",
        is_submetric=True,
        parent_metric="ecological_receptors",
        required_data="endangered_species_on_site",
        fill_missing_value={"domain": Bool, "default": False},
    ),
}

ENVIRONMENT_METRICS = {
    "environment": _SupportedContent(
        name="environment",
        full_name="Environment",
        has_submetrics=True,
    ),
    "fed_wetlands_near": _SupportedContent(
        name="fed_wetlands_near",
        full_name="Federal Wetlands (Close Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="fed_wetlands_near",
        # Assuming no nearby federal wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "fed_wetlands_far": _SupportedContent(
        name="fed_wetlands_far",
        full_name="Federal Wetlands (Distant Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="fed_wetlands_far",
        # Assuming no distant federal wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "known_soil_or_water_impact": _SupportedContent(
        name="known_soil_or_water_impact",
        full_name="Known Soil or Water Impact [Yes/No]",
        is_submetric=True,
        parent_metric="environment",
        required_data="known_soil_or_water_impact",
        # Assume that the well does not impact soil and water if not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "state_wetlands_far": _SupportedContent(
        name="state_wetlands_far",
        full_name="State Wetlands (Distant Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="state_wetlands_far",
        # Assuming no distant state wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "state_wetlands_near": _SupportedContent(
        name="state_wetlands_near",
        full_name="State Wetlands (Close Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="state_wetlands_near",
        # Assuming no nearby state wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "water_source_nearby": _SupportedContent(
        name="water_source_nearby",
        full_name="Water Source Nearby [Yes/No]",
        is_submetric=True,
        parent_metric="environment",
        required_data="water_source_nearby",
        # Assume that the well is not near water source if not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
}

LEAK_METRICS = {
    "other_losses": _SupportedContent(
        name="other_losses",
        full_name="Other Losses",
        has_submetrics=True,
    ),
    "brine_leak": _SupportedContent(
        name="brine_leak",
        full_name="Brine Leak [Yes/No]",
        is_submetric=True,
        parent_metric="other_losses",
        required_data="brine_leak",
        # Assuming no brine leak if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "h2s_leak": _SupportedContent(
        name="h2s_leak",
        full_name="H2S Leak [Yes/No]",
        is_submetric=True,
        parent_metric="other_losses",
        required_data="h2s_leak",
        # Assuming no H2S leak if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
}

# these are impact metrics without parent metrics
OTHER_IMPACT_METRICS = {
    "cost_of_plugging": _SupportedContent(
        name="cost_of_plugging",
        full_name="Cost of Plugging [USD]",
        required_data="cost_of_plugging",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 999999},
    ),
    "high_pressure_observed": _SupportedContent(
        name="high_pressure_observed",
        full_name="High Pressure Observed [Yes/No]",
        required_data="high_pressure_observed",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "idle_status_duration": _SupportedContent(
        name="idle_status_duration",
        full_name="Idle Status Duration [Years]",
        required_data="idle_status_duration",
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "in_tribal_land": _SupportedContent(
        name="in_tribal_land",
        full_name="In Tribal Land [Yes/No]",
        required_data="in_tribal_land",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "likely_to_be_orphaned": _SupportedContent(
        name="likely_to_be_orphaned",
        full_name="Likely to be Orphaned [Yes/No]",
        required_data="likely_to_be_orphaned",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "mechanical_integrity_test": _SupportedContent(
        name="mechanical_integrity_test",
        full_name="Mechanical Integrity Test [#]",
        required_data="mechanical_integrity_test",
        fill_missing_value={"domain": NonNegativeFloat, "default": False},
    ),
    "number_of_mcws_nearby": _SupportedContent(
        name="number_of_mcws_nearby",
        full_name="Number of MCWs Nearby [#]",
        required_data="number_of_mcws_nearby",
        fill_missing_value={"domain": NonNegativeInt, "default": 0},
    ),
    "otherwise_incentivized_well": _SupportedContent(
        name="otherwise_incentivized_well",
        full_name="Otherwise Incentivized Well [Yes/No]",
        required_data="otherwise_incentivized_well",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "well_age": _SupportedContent(
        name="well_age",
        full_name="Well Age",
        required_data="age",
    ),
    "well_count": _SupportedContent(
        name="well_count",
        full_name="Owner Well Count",
        required_data="operator_name",
    ),
    "well_integrity": _SupportedContent(
        name="well_integrity",
        full_name="Well Integrity Issues [Yes/No]",
        required_data="well_integrity",
        # If it is not specified, assume no well integrity issues
        fill_missing_value={"domain": Bool, "default": False},
    ),
}

PLACEHOLDERS = {
    "placeholder_one": _SupportedContent(
        name="placeholder_one",
        full_name="Placeholder 1",
        required_data="placeholder_one",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "placeholder_two": _SupportedContent(
        name="placeholder_two",
        full_name="Placeholder 2",
        required_data="placeholder_two",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "placeholder_three": _SupportedContent(
        name="placeholder_three",
        full_name="Placeholder 3",
        required_data="placeholder_three",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "placeholder_four": _SupportedContent(
        name="placeholder_four",
        full_name="Placeholder 4",
        required_data="placeholder_four",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "placeholder_five": _SupportedContent(
        name="placeholder_five",
        full_name="Placeholder 5",
        required_data="placeholder_five",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "placeholder_six": _SupportedContent(
        name="placeholder_six",
        full_name="Placeholder 6",
        required_data="placeholder_six",
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "placeholder_seven": _SupportedContent(
        name="placeholder_seven",
        full_name="Placeholder 7",
        required_data="placeholder_seven",
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "placeholder_eight": _SupportedContent(
        name="placeholder_eight",
        full_name="Placeholder 8",
        required_data="placeholder_eight",
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "placeholder_nine": _SupportedContent(
        name="placeholder_nine",
        full_name="Placeholder 9",
        required_data="placeholder_nine",
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "placeholder_ten": _SupportedContent(
        name="placeholder_ten",
        full_name="Placeholder 10",
        required_data="placeholder_ten",
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "placeholder_eleven": _SupportedContent(
        name="placeholder_eleven",
        full_name="Placeholder 11",
        required_data="placeholder_eleven",
        has_inverse_priority=True,
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "placeholder_twelve": _SupportedContent(
        name="placeholder_twelve",
        full_name="Placeholder 12",
        required_data="placeholder_twelve",
        has_inverse_priority=True,
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "placeholder_thirteen": _SupportedContent(
        name="placeholder_thirteen",
        full_name="Placeholder 13",
        required_data="placeholder_thirteen",
        has_inverse_priority=True,
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "placeholder_fourteen": _SupportedContent(
        name="placeholder_fourteen",
        full_name="Placeholder 14",
        required_data="placeholder_fourteen",
        has_inverse_priority=True,
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "placeholder_fifteen": _SupportedContent(
        name="placeholder_fifteen",
        full_name="Placeholder 15",
        required_data="placeholder_fifteen",
        has_inverse_priority=True,
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "placeholder_sixteen": _SupportedContent(
        name="placeholder_sixteen",
        full_name="Placeholder 16",
        required_data="placeholder_sixteen",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 1e10},
    ),
    "placeholder_seventeen": _SupportedContent(
        name="placeholder_seventeen",
        full_name="Placeholder 17",
        required_data="placeholder_seventeen",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 1e10},
    ),
    "placeholder_eighteen": _SupportedContent(
        name="placeholder_eighteen",
        full_name="Placeholder 18",
        required_data="placeholder_eighteen",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 1e10},
    ),
    "placeholder_nineteen": _SupportedContent(
        name="placeholder_nineteen",
        full_name="Placeholder 19",
        required_data="placeholder_nineteen",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 1e10},
    ),
    "placeholder_twenty": _SupportedContent(
        name="placeholder_twenty",
        full_name="Placeholder 20",
        required_data="placeholder_twenty",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 1e10},
    ),
}

PRODUCTION_VOLUME_METRICS = {
    "ann_production_volume": _SupportedContent(
        name="ann_production_volume",
        full_name="Annual Production Volume",
        has_submetrics=True,
    ),
    "ann_gas_production": _SupportedContent(
        name="ann_gas_production",
        full_name="Annual Gas Production [in Mcf/Year]",
        is_submetric=True,
        parent_metric="ann_production_volume",
        required_data="ann_gas_production",
        # Higher gas production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing gas, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "ann_oil_production": _SupportedContent(
        name="ann_oil_production",
        full_name="Annual Oil Production [in bbl/Year]",
        is_submetric=True,
        parent_metric="ann_production_volume",
        required_data="ann_oil_production",
        # Higher oil production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing oil, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "five_year_production_volume": _SupportedContent(
        name="five_year_production_volume",
        full_name="Five-year Production Volume",
        has_submetrics=True,
    ),
    "five_year_gas_production": _SupportedContent(
        name="five_year_gas_production",
        full_name="Five-year Gas Production [in Mcf]",
        is_submetric=True,
        parent_metric="five_year_production_volume",
        required_data="five_year_gas_production",
        # Higher gas production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing gas, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "five_year_oil_production": _SupportedContent(
        name="five_year_oil_production",
        full_name="Five-year Oil Production [in bbl]",
        is_submetric=True,
        parent_metric="five_year_production_volume",
        required_data="five_year_oil_production",
        # Higher oil production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing oil, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "lifelong_production_volume": _SupportedContent(
        name="lifelong_production_volume",
        full_name="Lifelong Production Volume",
        has_submetrics=True,
    ),
    "lifelong_gas_production": _SupportedContent(
        name="lifelong_gas_production",
        full_name="Lifelong Gas Production [in Mcf]",
        is_submetric=True,
        parent_metric="lifelong_production_volume",
        required_data="life_gas_production",
        # Higher gas production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing gas, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "lifelong_oil_production": _SupportedContent(
        name="lifelong_oil_production",
        full_name="Lifelong Oil Production [in bbl]",
        is_submetric=True,
        parent_metric="lifelong_production_volume",
        required_data="life_oil_production",
        # Higher oil production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing oil, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
}

SENSITIVE_RECEPTORS_METRICS = {
    "sensitive_receptors": _SupportedContent(
        name="sensitive_receptors",
        full_name="Sensitive Receptors",
        has_submetrics=True,
    ),
    "agriculture_area_nearby": _SupportedContent(
        name="agriculture_area_nearby",
        full_name="Agriculture Area Nearby [Yes/No]",
        required_data="agriculture_area_nearby",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "buildings_far": _SupportedContent(
        name="buildings_far",
        full_name="Buildings (Distant Range)",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="buildings_far",
        # Assuming no distant buildings
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "buildings_near": _SupportedContent(
        name="buildings_near",
        full_name="Buildings (Close Range)",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="buildings_near",
        # Assuming that no buildings are nearby if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "hospitals": _SupportedContent(
        name="hospitals",
        full_name="Hospitals",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="hospitals",
        fill_missing_value={"domain": NonNegativeInt, "default": 0},
    ),
    "schools": _SupportedContent(
        name="schools",
        full_name="Schools",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="schools",
        fill_missing_value={"domain": NonNegativeInt, "default": 0},
    ),
}

SITE_CONSIDERATION_METRICS = {
    "site_considerations": _SupportedContent(
        name="site_considerations",
        full_name="Site Considerations",
        has_submetrics=True,
    ),
    "historical_preservation_site": _SupportedContent(
        name="historical_preservation_site",
        full_name="Historical Preservation Site [Yes/No]",
        required_data="historical_preservation_site",
        parent_metric="site_considerations",
        is_submetric=True,
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "home_use_gas_well": _SupportedContent(
        name="home_use_gas_well",
        full_name="Home Use Gas Well [Yes/No]",
        required_data="home_use_gas_well",
        parent_metric="site_considerations",
        is_submetric=True,
        has_inverse_priority=True,
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "post_plugging_land_use": _SupportedContent(
        name="post_plugging_land_use",
        full_name="Post-Plugging Land Use [Yes/No]",
        required_data="post_plugging_land_use",
        parent_metric="site_considerations",
        is_submetric=True,
        fill_missing_value={"domain": Bool, "default": False},
    ),
    # "proximity_to_geologic_faults": _SupportedContent(
    #     name="proximity_to_geologic_faults",
    #     full_name="Proximity to Geologic Faults [miles]",
    #     required_data="proximity_to_geologic_faults",
    #     parent_metric="site_considerations",
    #     is_submetric=True,
    #     fill_missing_value={"domain": NonNegativeFloat, "default": 1e5},
    # ),
    "surface_equipment_on_site": _SupportedContent(
        name="surface_equipment_on_site",
        full_name="Surface Equipment on Site [Yes/No]",
        required_data="surface_equipment_on_site",
        parent_metric="site_considerations",
        is_submetric=True,
        fill_missing_value={"domain": Bool, "default": False},
    ),
}


SUPP_IMPACT_METRICS = {
    **WELL_HISTORY_METRICS,
    **ECOLOGICAL_RECEPTORS,
    **ENVIRONMENT_METRICS,
    **LEAK_METRICS,
    **OTHER_IMPACT_METRICS,
    **PLACEHOLDERS,
    **PRODUCTION_VOLUME_METRICS,
    **SENSITIVE_RECEPTORS_METRICS,
    **SITE_CONSIDERATION_METRICS,
}

# note the names of these metrics must be identical to
# a property of the Optimal Project class for the
# efficiency calculation
SUPP_EFF_METRICS = {
    "num_wells": _SupportedContent(
        name="num_wells", full_name="Number of Wells", required_data="well_id"
    ),
    "num_unique_owners": _SupportedContent(
        name="num_unique_owners",
        full_name="Number of Unique Owners",
        required_data="operator_name",
        has_inverse_priority=True,
    ),
    "elevation_delta": _SupportedContent(
        name="elevation_delta",
        full_name="Elevation Delta [m]",
        required_data="elevation_delta",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "age_range": _SupportedContent(
        name="age_range",
        full_name="Age Range [Years]",
        required_data="age",
        has_inverse_priority=True,
    ),
    "depth_range": _SupportedContent(
        name="depth_range",
        full_name="Depth Range [ft]",
        required_data="depth",
        has_inverse_priority=True,
    ),
    "dist_range": _SupportedContent(
        name="dist_range",
        full_name="Distance Range [miles]",
        has_inverse_priority=True,
        required_data="latitude",
    ),
    "record_completeness": _SupportedContent(
        name="record_completeness",
        full_name="Record Completeness",
    ),
    "dist_to_road": _SupportedContent(
        name="dist_to_road",
        full_name="Distance to Road [miles]",
        required_data="dist_to_road",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "population_density": _SupportedContent(
        name="population_density",
        full_name="Total Population Density [#/km2]",
        required_data="population_density",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
}
