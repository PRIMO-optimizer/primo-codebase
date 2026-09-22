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
GitHub secrets are not passed to pull requests: Therefore all tests that rely on secrets
must be turned off when Actions are run for an incoming PR. This in turn leads to
codecov complaining about reduced coverage with a patch.

Thus all tests that rely on secrets are implemented in this single file which is
ignored for codecov analysis by suitably setting ignore paths in codecov.yml
"""
# pylint: disable=missing-function-docstring
# Installed libs
import pytest

# User-defined libs
from primo.utils.census_utils import CensusClient, get_census_key
from primo.utils.demo_utils import get_population_by_state

# Sample state code for testing. 37 stands for North Carolina
STATE_CODE = 37
STATE_CODE_FAKE = 60


# pylint: disable=protected-access
@pytest.mark.secrets
def test_generate_geo_identifiers():
    census_key = get_census_key()
    client = CensusClient(census_key)
    generate_identifiers = client._generate_geo_identifiers
    assert generate_identifiers("42") == ("state:42", "")
    assert generate_identifiers("") == ("", "")
    assert generate_identifiers("42079") == ("county:079", "state:42")
    assert generate_identifiers("42079216601") == (
        "tract:216601",
        "state:42 county:079",
    )


@pytest.mark.secrets
def test_get_population_by_state():
    result_df = get_population_by_state(STATE_CODE)
    assert "Total Population" in result_df.columns
    assert result_df["state"].dtypes == int
    assert result_df["county"].dtypes == int
    assert result_df["tract"].dtypes == int
    assert result_df["Total Population"].dtypes == int

    with pytest.raises(AssertionError):
        get_population_by_state(STATE_CODE_FAKE)
