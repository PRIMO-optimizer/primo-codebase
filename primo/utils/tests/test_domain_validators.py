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

# Installed libs
import pytest

# User-defined libs
from primo.utils.domain_validators import (
    InRange,
    is_valid_inverse_priority_zone_data,
    is_valid_zone_data,
    validate_mobilization_cost,
)


def test_inrange():
    """Tests InRange domain validator"""
    val = 5.0
    assert val == InRange(0.0, 10.0)(val)

    val = -5.0
    with pytest.raises(
        ValueError,
        match=("Value -5.0 lies outside the admissible range \\[0.0, 10.0\\]"),
    ):
        InRange(0.0, 10.0)(val)


def test_validate_mobilization_cost():
    """Tests the validate_mobilization_cost function"""
    data = {1: 100, 2: 200, 3: 300}
    result = validate_mobilization_cost(data)
    assert result == {1: 100.0, 2: 200.0, 3: 300.0}

    data[5] = 500
    with pytest.raises(
        KeyError, match="Mobilization cost for 4 wells is not provided."
    ):
        validate_mobilization_cost(data)


def test_is_valid_zone_data():
    """Tests the is_valid_zone_data function"""
    # Test ascending order of zones
    data = {1: 1, 5: 0.5, 3: 0.25}
    with pytest.raises(
        ValueError,
        match=f"Zones in {data} are not in ascending order",
    ):
        is_valid_zone_data(data)

    # Test zero zone length error
    data = {1: 1, 5.0: 0.5, 5.00001: 0.25}
    with pytest.raises(ValueError, match="Zones 2 and 3 are identical!"):
        is_valid_zone_data(data)

    # Test efficiency > 1 error
    data = {1: 1, 3: 5, 5: 0.25}
    with pytest.raises(
        ValueError,
        match="Received an efficiency value outside \\[0, 1\\] interval",
    ):
        is_valid_zone_data(data)

    # Test descending order of efficiencies
    data = {1: 1, 3: 0.25, 5: 0.5}
    with pytest.raises(
        ValueError,
        match="Zonal efficiencies are not in descending order",
    ):
        is_valid_zone_data(data)

    # Test missing 100% efficient zone
    data = {3: 0.5, 5: 0.25}
    with pytest.raises(
        ValueError,
        match="Efficiency of the first zone 0.5 is not 1",
    ):
        is_valid_zone_data(data)

    # Test the output
    data = {1: 1, 3: 0.5, 5: 0.25}
    zd = is_valid_zone_data(data)

    assert zd.data == {1: 1, 3: 0.5, 5: 0.25, float("inf"): 0}
    assert zd.zones == [1, 2, 3, 4]
    assert zd.points == [0, 1, 3, 5, float("inf")]
    assert zd.coeff == {1: 0, 2: 0.5, 3: 0.25, 4: 0.25}


def test_is_valid_inverse_priority_zone_data():
    """Tests the is_valid_inverse_priority_zone_data validator"""
    # Since it uses is_valid_zone_data, most functionalities
    # are not tested here again to avoid duplication
    data = {5: 0, 10: 0.1, 15: 0.3, 20: 0.6, 25: 0.95}
    zd = is_valid_inverse_priority_zone_data(data)

    assert zd.zones == [1, 2, 3, 4, 5, 6]
    assert zd.points == [0, 5, 10, 15, 20, 25, float("inf")]
    for k, v in data.items():
        assert v == pytest.approx(zd.data[k])

    coeffs = {1: 0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.35, 6: 0.05}
    for k, v in zd.coeff.items():
        assert v == pytest.approx(coeffs[k])
