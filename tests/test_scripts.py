"""
The run scripts in scripts/ are used in class: they must run as they are,
from an empty folder, as a student would run them.

Each script is copied to a temporary folder and run there; on the first run it
copies its example model next to itself, solves it with HiGHS, saves the
results and draws the plots.
"""
import os
import runpy
import shutil

import cvxpy as cp
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

pytestmark = pytest.mark.skipif("HIGHS" not in cp.installed_solvers(),
                                reason="the scripts use HiGHS")


@pytest.mark.parametrize("script, example", [
    ("run_operation.py", "Operation_1Region"),
    ("run_planning.py", "Planning_1Region"),
])
def test_script_runs_from_an_empty_folder(script, example, tmp_path, capsys):
    shutil.copy(os.path.join(SCRIPTS, script), tmp_path)
    runpy.run_path(str(tmp_path / script), run_name="__main__")

    model = tmp_path / example
    assert (model / "results" / "Single" / "tech_production.csv").exists()
    assert (model / "plots" / "total_capacity.html").exists()
    assert "Plots saved" in capsys.readouterr().out
