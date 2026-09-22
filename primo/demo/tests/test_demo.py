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
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.parametrize(
    "file_path",
    [
        "PRIMO - Example_Workflow.ipynb",
        "PRIMO - Example_Project_Recommendation.ipynb",
        "PRIMO - Example_Project_Comparison.ipynb",
        "PRIMO - Example_Well_Ranking.ipynb",
    ],
)
def test_example(file_path):
    """Run example demo notebooks and raise an error
    if the notebook fails to run"""

    notebook_path = os.path.join("primo", "demo", file_path)
    with open(notebook_path, encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    ep = ExecutePreprocessor()
    try:
        ep.preprocess(notebook, {"metadata": {"path": os.path.dirname(notebook_path)}})
        assert True, "Running demo notebook succeeded"
    except:  # pylint: disable=bare-except
        assert False, "Demo notebook did not run successfully."
