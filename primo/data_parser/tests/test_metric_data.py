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
import logging
from types import SimpleNamespace

# Installed libs
import pytest
from pyomo.common.config import Bool, NonNegativeFloat

# User-defined libs
from primo.data_parser.metric_data import (
    EfficiencyMetrics,
    ImpactMetrics,
    Metric,
    SetOfMetrics,
    SubMetric,
)

LOGGER = logging.getLogger(__name__)


# pylint: disable = no-member, missing-function-docstring
# pylint: disable = protected-access
# pylint: disable = too-many-statements
def test_metric_class(caplog):
    z = Metric(name="met_1", weight=50, full_name="Metric One")

    assert z.name == "met_1"
    assert z.full_name == "Metric One"
    assert (z.min_weight, z.weight, z.max_weight) == (0, 50, 100)
    assert z.effective_weight == 50
    assert not z.is_submetric
    print_format = (
        "Metric name: Metric One, Metric weight: 50 \n    Admissible range: [0, 100]"
    )
    assert str(z) == print_format

    assert z.data_col_name is None
    assert z.score_col_name is None
    assert z.score_attribute is None

    assert z.is_binary_type is False
    assert z.has_inverse_priority is False
    assert z.fill_missing_value is None
    assert z._fill_missing_value.name() == "met_1"
    z.fill_missing_value = "foo"
    assert z.fill_missing_value == "foo"

    z.data_col_name = "Metric One"
    assert z.score_col_name == "Metric One Score [0-50]"
    assert z.score_attribute == "met_1_eff_score_0_50"

    # Add test to capture the name modification error
    with pytest.raises(
        ValueError,
        match=("Metric's key name cannot be modified after it is defined."),
    ):
        z.name = "modified_met_1"

    # Add a test to catch a metric name that is not a valid Python variable name
    with pytest.raises(
        ValueError,
        match=(
            "Received met 2 for Metric's key name, which is not a valid python variable name!"
        ),
    ):
        print(Metric("met 2", 100))

    # Add test to capture the weight out-of-range error
    with pytest.raises(
        ValueError,
        match=(
            "Attempted to assign -50 for metric met_1, which "
            "lies outside the admissible range \\[0, 100\\]."
        ),
    ):
        z.weight = -50

    # Test warning for assigning non-integer value
    z.weight = 50.7
    assert (
        "Received 50.7, a non-integer value for weight. "
        "Rounding it to 51, the nearest integer value."
    ) in caplog.text

    # Test the _configure_fill_missing_value method
    z._configure_fill_missing_value(domain=NonNegativeFloat, default=3)
    assert z.fill_missing_value == 3.0
    z.fill_missing_value = 5
    assert z.fill_missing_value == 5.0
    assert not z.is_binary_type

    with pytest.raises(
        ValueError,
        match=(
            "invalid value for configuration 'met_1':\n"
            "\tFailed casting foo\n\tto NonNegativeFloat\n"
            "\tError: could not convert string to float: 'foo'"
        ),
    ):
        z.fill_missing_value = "foo"

    z._configure_fill_missing_value(domain=Bool, default="Yes")
    assert z.is_binary_type
    assert z.fill_missing_value is True


def test_submetric_class():
    z_par = Metric("par_metric", 40, full_name="Parent Metric One")
    z_sub = SubMetric("sub_metric", z_par, 50, full_name="SubMetric One")

    assert z_sub.is_submetric
    assert z_sub.parent_metric is z_par
    assert z_sub.effective_weight == 20
    print_format = (
        "Submetric name: SubMetric One, Submetric weight: 50 \n"
        "    Admissible range: [0, 100] \n"
        "    Is a submetric of Parent Metric One"
    )
    assert str(z_sub) == print_format


def test_set_of_metrics_class():
    z = SetOfMetrics()

    z.register_new_metric("met_1", "Metric One")
    z.register_new_metric("met_2", "Metric Two")
    z.register_new_metric("met_3", "Metric Three")
    z.set_weight({"met_1": 33, "met_2": 33, "met_3": 34})

    print_format = (
        "        Metric Name  Metric weight\n"
        "met_1    Metric One             33\n"
        "met_2    Metric Two             33\n"
        "met_3  Metric Three             34\n"
        "              Total            100"
    )

    assert str(z) == print_format
    weights = z.get_weights
    assert weights.met_1 == 33
    assert weights.met_2 == 33
    assert weights.met_3 == 34

    z.register_new_submetric("sub_met_1_1", z.met_1, "Sub Metric One-One")
    z.register_new_submetric("sub_met_1_2", z.met_1, "Sub Metric One-Two")
    z.register_new_submetric("sub_met_3_1", z.met_3, "Sub Metric Three-One")
    z.register_new_submetric("sub_met_3_2", z.met_3, "Sub Metric Three-Two")
    z.set_weight(
        {"sub_met_1_1": 50, "sub_met_1_2": 50, "sub_met_3_1": 25, "sub_met_3_2": 75}
    )

    print_format = (
        "        Metric Name  Metric weight\n"
        "met_1    Metric One             33\n"
        "met_2    Metric Two             33\n"
        "met_3  Metric Three             34\n"
        "              Total            100\n\n\n"
        "Primary metric Metric One, with weight 33, has submetrics:\n"
        "================================================================================\n"
        "                 Submetric Name  Submetric weight\n"
        "sub_met_1_1  Sub Metric One-One                50\n"
        "sub_met_1_2  Sub Metric One-Two                50\n"
        "                          Total               100\n\n\n"
        "Primary metric Metric Three, with weight 34, has submetrics:\n"
        "================================================================================\n"
        "                   Submetric Name  Submetric weight\n"
        "sub_met_3_1  Sub Metric Three-One                25\n"
        "sub_met_3_2  Sub Metric Three-Two                75\n"
        "                            Total               100"
    )

    assert str(z) == print_format
    weights = z.get_weights
    assert isinstance(weights, SimpleNamespace)
    assert weights.sub_met_1_1 == 16.5
    assert weights.sub_met_1_2 == 16.5
    assert weights.met_2 == 33
    assert weights.sub_met_3_1 == 8.5
    assert weights.sub_met_3_2 == 25.5
    assert not hasattr(weights, "met_1")
    assert not hasattr(weights, "met_3")

    assert z.check_validity() is None
    assert isinstance(z.met_1, Metric)
    assert isinstance(z.met_2, Metric)
    assert isinstance(z.met_3, Metric)
    assert isinstance(z.sub_met_1_1, SubMetric)
    assert isinstance(z.sub_met_1_2, SubMetric)
    assert isinstance(z.sub_met_3_1, SubMetric)
    assert isinstance(z.sub_met_3_2, SubMetric)

    assert z.met_1.submetrics == {
        "sub_met_1_1": z.sub_met_1_1,
        "sub_met_1_2": z.sub_met_1_2,
    }
    assert not hasattr(z.met_2, "submetrics")
    assert z.met_3.submetrics == {
        "sub_met_3_1": z.sub_met_3_1,
        "sub_met_3_2": z.sub_met_3_2,
    }

    _all_metrics = {
        "met_1": z.met_1,
        "met_2": z.met_2,
        "met_3": z.met_3,
        "sub_met_1_1": z.sub_met_1_1,
        "sub_met_1_2": z.sub_met_1_2,
        "sub_met_3_1": z.sub_met_3_1,
        "sub_met_3_2": z.sub_met_3_2,
    }
    assert dict(z.items()) == _all_metrics
    assert z.get_primary_metrics == {
        "met_1": z.met_1,
        "met_2": z.met_2,
        "met_3": z.met_3,
    }
    assert z.get_submetrics == {
        "met_1": {"sub_met_1_1": z.sub_met_1_1, "sub_met_1_2": z.sub_met_1_2},
        "met_3": {"sub_met_3_1": z.sub_met_3_1, "sub_met_3_2": z.sub_met_3_2},
    }
    assert z._get_all_metrics_extended == {
        "met_1": z.met_1,
        "met_2": z.met_2,
        "met_3": z.met_3,
        "sub_met_1_1": z.sub_met_1_1,
        "sub_met_1_2": z.sub_met_1_2,
        "sub_met_3_1": z.sub_met_3_1,
        "sub_met_3_2": z.sub_met_3_2,
        "Metric One": z.met_1,
        "Metric Two": z.met_2,
        "Metric Three": z.met_3,
        "Sub Metric One-One": z.sub_met_1_1,
        "Sub Metric One-Two": z.sub_met_1_2,
        "Sub Metric Three-One": z.sub_met_3_1,
        "Sub Metric Three-Two": z.sub_met_3_2,
    }

    # Test iterator method and contains method
    for obj in z:
        assert obj in _all_metrics.values()
        assert obj.name in z
        assert obj.full_name in z

    # Test for receiving update error: Enter a typo in the input dict
    with pytest.raises(
        KeyError,
        match=("Metrics/submetrics \\['met 1'\\] are not recognized/registered."),
    ):
        z.set_weight({"met 1": 33, "Metric Two": 33})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError, match=("Sum of weights of primary metrics does not add up to 100")
    ):
        z.set_weight({"met_1": 33, "met_2": 30})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError,
        match=(
            "Weight of the primary metric met_1 is zero, but the sum of "
            "weights of its submetrics is 100, which is nonzero."
        ),
    ):
        z.set_weight({"met_1": 0, "met_2": 66})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError,
        match=(
            "Sum of weights of submetrics of the primary metric met_1 does not add up to 100."
        ),
    ):
        z.set_weight(
            {"met_1": 33, "met_2": 33}, submetrics={"met_1": {"sub_met_1_1": 0}}
        )

    z.set_weight({"sub_met_1_1": 50})

    # Test for receiving error raised when registering non-Metric instances
    with pytest.raises(
        TypeError,
        match=(
            "Attributes of SetOfMetrics must be instances of Metric. "
            "Attempted to register Foo."
        ),
    ):
        z.test = "Foo"

    # Test for receiving error when an existing metric is overwritten
    with pytest.raises(
        ValueError,
        match=(
            "Metric/submetric met_2 has already been registered. "
            "Attempting to register a new metric with the same name."
        ),
    ):
        z.register_new_metric("met_2")

    # Try deleting primary metric on z.met_1. This should automatically delete submetrics
    z.delete_metric("met_1")
    assert not hasattr(z, "met_1")
    assert not hasattr(z, "sub_met_1_1")
    assert not hasattr(z, "sub_met_1_2")
    assert len(list(z)) == 4

    # Test for receiving error when deleting a metric that does not exist
    with pytest.raises(
        AttributeError, match=("Metric/submetric met_1 does not exist.")
    ):
        z.delete_metric("met_1")

    # Try deleting submetric for z.sub_met_3_1.
    z.delete_submetric("sub_met_3_1")
    assert not hasattr(z, "sub_met_3_1")

    # Test for receiving error when deleting a metric that does not exist
    with pytest.raises(AttributeError, match="Submetric sub_met_3_1 does not exist."):
        z.delete_submetric("sub_met_3_1")


def test_impact_metrics_class():
    im_wt = ImpactMetrics()

    # re-organized to match organization in default_data.py
    assert hasattr(im_wt, "well_history")
    assert hasattr(im_wt, "leak")
    assert hasattr(im_wt, "compliance")
    assert hasattr(im_wt, "violation")
    assert hasattr(im_wt, "incident")
    assert hasattr(im_wt, "hydrocarbon_losses")
    assert im_wt.well_history.submetrics == {
        "leak": im_wt.leak,
        "compliance": im_wt.compliance,
        "violation": im_wt.violation,
        "incident": im_wt.incident,
        "hydrocarbon_losses": im_wt.hydrocarbon_losses,
    }

    assert hasattr(im_wt, "ecological_receptors")
    assert hasattr(im_wt, "endangered_species_on_site")
    assert im_wt.ecological_receptors.submetrics == {
        "endangered_species_on_site": im_wt.endangered_species_on_site
    }

    assert hasattr(im_wt, "environment")
    assert hasattr(im_wt, "fed_wetlands_near")
    assert hasattr(im_wt, "fed_wetlands_far")
    assert hasattr(im_wt, "state_wetlands_near")
    assert hasattr(im_wt, "state_wetlands_far")
    assert hasattr(im_wt, "known_soil_or_water_impact")
    assert hasattr(im_wt, "water_source_nearby")
    assert im_wt.environment.submetrics == {
        "fed_wetlands_near": im_wt.fed_wetlands_near,
        "state_wetlands_near": im_wt.state_wetlands_near,
        "fed_wetlands_far": im_wt.fed_wetlands_far,
        "state_wetlands_far": im_wt.state_wetlands_far,
        "known_soil_or_water_impact": im_wt.known_soil_or_water_impact,
        "water_source_nearby": im_wt.water_source_nearby,
    }

    assert hasattr(im_wt, "other_losses")
    assert hasattr(im_wt, "h2s_leak")
    assert hasattr(im_wt, "brine_leak")
    assert im_wt.other_losses.submetrics == {
        "h2s_leak": im_wt.h2s_leak,
        "brine_leak": im_wt.brine_leak,
    }

    assert hasattr(im_wt, "cost_of_plugging")
    assert hasattr(im_wt, "high_pressure_observed")
    assert hasattr(im_wt, "idle_status_duration")
    assert hasattr(im_wt, "in_tribal_land")
    assert hasattr(im_wt, "likely_to_be_orphaned")
    assert hasattr(im_wt, "mechanical_integrity_test")
    assert hasattr(im_wt, "number_of_mcws_nearby")
    assert hasattr(im_wt, "otherwise_incentivized_well")
    assert hasattr(im_wt, "well_count")
    assert hasattr(im_wt, "well_integrity")
    assert hasattr(im_wt, "well_age")

    assert not hasattr(im_wt.well_age, "submetrics")
    assert not hasattr(im_wt.well_count, "submetrics")
    assert not hasattr(im_wt.well_integrity, "submetrics")
    assert not hasattr(im_wt.cost_of_plugging, "submetrics")
    assert not hasattr(im_wt.high_pressure_observed, "submetrics")
    assert not hasattr(im_wt.in_tribal_land, "submetrics")
    assert not hasattr(im_wt.likely_to_be_orphaned, "submetrics")
    assert not hasattr(im_wt.number_of_mcws_nearby, "submetrics")
    assert not hasattr(im_wt.otherwise_incentivized_well, "submetrics")

    assert hasattr(im_wt, "placeholder_one")
    assert hasattr(im_wt, "placeholder_two")
    assert hasattr(im_wt, "placeholder_three")
    assert hasattr(im_wt, "placeholder_four")
    assert hasattr(im_wt, "placeholder_five")
    assert hasattr(im_wt, "placeholder_six")
    assert hasattr(im_wt, "placeholder_seven")
    assert hasattr(im_wt, "placeholder_eight")
    assert hasattr(im_wt, "placeholder_nine")
    assert hasattr(im_wt, "placeholder_ten")
    assert hasattr(im_wt, "placeholder_eleven")
    assert hasattr(im_wt, "placeholder_twelve")
    assert hasattr(im_wt, "placeholder_thirteen")
    assert hasattr(im_wt, "placeholder_fourteen")
    assert hasattr(im_wt, "placeholder_fifteen")
    assert hasattr(im_wt, "placeholder_sixteen")
    assert hasattr(im_wt, "placeholder_seventeen")
    assert hasattr(im_wt, "placeholder_eighteen")
    assert hasattr(im_wt, "placeholder_nineteen")
    assert hasattr(im_wt, "placeholder_twenty")

    assert not hasattr(im_wt.placeholder_one, "submetrics")
    assert not hasattr(im_wt.placeholder_two, "submetrics")
    assert not hasattr(im_wt.placeholder_three, "submetrics")
    assert not hasattr(im_wt.placeholder_four, "submetrics")
    assert not hasattr(im_wt.placeholder_five, "submetrics")
    assert not hasattr(im_wt.placeholder_six, "submetrics")
    assert not hasattr(im_wt.placeholder_seven, "submetrics")
    assert not hasattr(im_wt.placeholder_eight, "submetrics")
    assert not hasattr(im_wt.placeholder_nine, "submetrics")
    assert not hasattr(im_wt.placeholder_ten, "submetrics")
    assert not hasattr(im_wt.placeholder_eleven, "submetrics")
    assert not hasattr(im_wt.placeholder_twelve, "submetrics")
    assert not hasattr(im_wt.placeholder_thirteen, "submetrics")
    assert not hasattr(im_wt.placeholder_fourteen, "submetrics")
    assert not hasattr(im_wt.placeholder_fifteen, "submetrics")
    assert not hasattr(im_wt.placeholder_sixteen, "submetrics")
    assert not hasattr(im_wt.placeholder_seventeen, "submetrics")
    assert not hasattr(im_wt.placeholder_eighteen, "submetrics")
    assert not hasattr(im_wt.placeholder_nineteen, "submetrics")
    assert not hasattr(im_wt.placeholder_twenty, "submetrics")

    assert hasattr(im_wt, "ann_production_volume")
    assert hasattr(im_wt, "ann_gas_production")
    assert hasattr(im_wt, "ann_oil_production")
    assert im_wt.ann_production_volume.submetrics == {
        "ann_gas_production": im_wt.ann_gas_production,
        "ann_oil_production": im_wt.ann_oil_production,
    }

    assert hasattr(im_wt, "five_year_production_volume")
    assert hasattr(im_wt, "five_year_gas_production")
    assert hasattr(im_wt, "five_year_oil_production")
    assert im_wt.five_year_production_volume.submetrics == {
        "five_year_gas_production": im_wt.five_year_gas_production,
        "five_year_oil_production": im_wt.five_year_oil_production,
    }

    assert hasattr(im_wt, "lifelong_production_volume")
    assert hasattr(im_wt, "lifelong_gas_production")
    assert hasattr(im_wt, "lifelong_oil_production")
    assert im_wt.lifelong_production_volume.submetrics == {
        "lifelong_gas_production": im_wt.lifelong_gas_production,
        "lifelong_oil_production": im_wt.lifelong_oil_production,
    }

    assert hasattr(im_wt, "sensitive_receptors")
    assert hasattr(im_wt, "hospitals")
    assert hasattr(im_wt, "schools")
    assert hasattr(im_wt, "buildings_near")
    assert hasattr(im_wt, "buildings_far")
    assert hasattr(im_wt, "agriculture_area_nearby")

    assert im_wt.sensitive_receptors.submetrics == {
        "hospitals": im_wt.hospitals,
        "schools": im_wt.schools,
        "buildings_near": im_wt.buildings_near,
        "buildings_far": im_wt.buildings_far,
        "agriculture_area_nearby": im_wt.agriculture_area_nearby,
    }

    assert hasattr(im_wt, "site_considerations")
    assert hasattr(im_wt, "historical_preservation_site")
    assert hasattr(im_wt, "home_use_gas_well")
    assert hasattr(im_wt, "post_plugging_land_use")
    assert hasattr(im_wt, "surface_equipment_on_site")
    assert im_wt.site_considerations.submetrics == {
        "post_plugging_land_use": im_wt.post_plugging_land_use,
        "historical_preservation_site": im_wt.historical_preservation_site,
        "home_use_gas_well": im_wt.home_use_gas_well,
        "surface_equipment_on_site": im_wt.surface_equipment_on_site,
    }

    assert im_wt.compliance.is_binary_type
    assert im_wt.compliance.has_inverse_priority
    assert im_wt.compliance._fill_missing_value._domain is Bool
    assert im_wt.compliance.fill_missing_value

    im_wt.set_weight(
        primary_metrics={
            "well_history": 10,
            "sensitive_receptors": 10,
            "environment": 20,
            "ecological_receptors": 10,
            "other_losses": 1,
            "ann_production_volume": 5,
            "five_year_production_volume": 2,
            "lifelong_production_volume": 2,
            "site_considerations": 9,
            "cost_of_plugging": 1,
            "high_pressure_observed": 1,
            "idle_status_duration": 1,
            "in_tribal_land": 1,
            "likely_to_be_orphaned": 1,
            "mechanical_integrity_test": 1,
            "number_of_mcws_nearby": 1,
            "otherwise_incentivized_well": 1,
            "well_count": 1,
            "well_integrity": 1,
            "well_age": 1,
            "placeholder_one": 1,
            "placeholder_two": 1,
            "placeholder_three": 1,
            "placeholder_four": 1,
            "placeholder_five": 1,
            "placeholder_six": 1,
            "placeholder_seven": 1,
            "placeholder_eight": 1,
            "placeholder_nine": 1,
            "placeholder_ten": 1,
            "placeholder_eleven": 1,
            "placeholder_twelve": 1,
            "placeholder_thirteen": 1,
            "placeholder_fourteen": 1,
            "placeholder_fifteen": 1,
            "placeholder_sixteen": 1,
            "placeholder_seventeen": 1,
            "placeholder_eighteen": 1,
            "placeholder_nineteen": 1,
            "placeholder_twenty": 1,
        },
        submetrics={
            "well_history": {
                "leak": 20,
                "compliance": 20,
                "violation": 20,
                "incident": 20,
                "hydrocarbon_losses": 20,
            },
            "environment": {
                "fed_wetlands_near": 15,
                "state_wetlands_near": 15,
                "fed_wetlands_far": 15,
                "state_wetlands_far": 15,
                "known_soil_or_water_impact": 15,
                "water_source_nearby": 25,
            },
            "other_losses": {
                "h2s_leak": 50,
                "brine_leak": 50,
            },
            "sensitive_receptors": {
                "hospitals": 20,
                "schools": 20,
                "buildings_near": 20,
                "buildings_far": 20,
                "agriculture_area_nearby": 20,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
            "five_year_production_volume": {
                "five_year_gas_production": 50,
                "five_year_oil_production": 50,
            },
            "lifelong_production_volume": {
                "lifelong_gas_production": 50,
                "lifelong_oil_production": 50,
            },
            "site_considerations": {
                "post_plugging_land_use": 25,
                "historical_preservation_site": 25,
                "home_use_gas_well": 25,
                "surface_equipment_on_site": 25,
            },
            "ecological_receptors": {
                "endangered_species_on_site": 100,
            },
        },
    )

    assert im_wt.check_validity() is None

    im_widget = im_wt.build_widget()
    im_widget.confirm_weights(None)
    im_wt.set_weight_from_widget(im_widget)
    assert im_wt.check_validity() is None


def test_efficiency_metrics_class():
    ef_wt = EfficiencyMetrics()

    assert hasattr(ef_wt, "num_wells")
    assert hasattr(ef_wt, "num_unique_owners")
    assert hasattr(ef_wt, "elevation_delta")
    assert hasattr(ef_wt, "age_range")
    assert hasattr(ef_wt, "depth_range")
    assert hasattr(ef_wt, "dist_range")
    assert hasattr(ef_wt, "population_density")
    assert hasattr(ef_wt, "record_completeness")
    assert hasattr(ef_wt, "dist_to_road")
    assert len(ef_wt.get_primary_metrics) == 9

    ef_wt.set_weight(
        {
            "num_wells": 20,
            "age_range": 20,
            "depth_range": 60,
        }
    )
    assert ef_wt.check_validity() is None


def test_custom_basis():
    im_wt = ImpactMetrics(basis=120)

    assert im_wt.basis == 120

    im_wt.set_weight(
        {
            "placeholder_one": 20,
            "placeholder_two": 20,
            "placeholder_three": 20,
            "placeholder_four": 20,
            "placeholder_five": 20,
            "placeholder_six": 20,
        }
    )

    assert im_wt.check_validity() is None
