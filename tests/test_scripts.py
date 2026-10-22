"""
The run scripts in scripts/ are used in class: they must run as they are,
from an empty folder, as a student would run them.

Each script is copied to a temporary folder and run there; on the first run it
copies its example model next to itself, solves it with HiGHS, saves the
results and draws the plots.
"""
import os
import re
import runpy
import shutil

import cvxpy as cp
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

needs_highs = pytest.mark.skipif("HIGHS" not in cp.installed_solvers(),
                                 reason="the scripts use HiGHS")


@pytest.mark.parametrize("script", ["run_operation.py", "run_planning.py"])
def test_script_lists_every_optional_constraint(script):
    """Each script lists every optional constraint, commented out, so that a
    student sees what can be switched on; a new option must be added there."""
    from hypatia.backend.constraints.ConstraintList import OPTIONAL_CONSTRAINTS

    with open(os.path.join(SCRIPTS, script), encoding="utf-8") as f:
        listed = re.findall(r'^\s*#\s*"(\w+)",', f.read(), flags=re.M)
    assert listed == list(OPTIONAL_CONSTRAINTS)


@needs_highs
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
