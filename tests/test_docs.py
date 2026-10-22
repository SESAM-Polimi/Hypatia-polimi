"""
The documentation must keep up with the code.

* Every parameter sheet the code can ask for is described in
  doc/tools/parameter_docs.py, and no description is left for a sheet that no
  longer exists. The input tables of the documentation are generated from the
  same data, so a new sheet cannot go undocumented.
* Every constraint class is listed in parameter_docs.CONSTRAINTS with the
  labels of its equations, and those labels exist in the formulation pages.
* The scripts in doc/examples, whose code and output the documentation shows,
  still run.
"""
import os
import re
import runpy
import sys

import cvxpy as cp
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "doc", "tools")
FORMULATION = os.path.join(ROOT, "doc", "source", "formulation")
EXAMPLES = os.path.join(ROOT, "doc", "examples")

sys.path.insert(0, TOOLS)
import parameter_docs  # noqa: E402
import reference  # noqa: E402


@pytest.fixture(scope="module")
def records():
    return reference.collect()


def test_every_parameter_sheet_is_documented(records):
    missing, stale = reference.undocumented(records)
    assert missing == [], f"sheets without a description in parameter_docs.PARAMETERS: {missing}"
    assert stale == [], f"descriptions of sheets the code no longer creates: {stale}"


def test_every_documented_parameter_has_symbol_unit_and_text():
    incomplete = [key for key, doc in parameter_docs.PARAMETERS.items()
                  if not (doc.get("symbol") and doc.get("unit") and doc.get("text"))]
    assert incomplete == []


def test_units_use_the_documented_symbols():
    """Units are written with the symbols of parameter_docs.UNITS (e.g. [E/(P·yr)])."""
    known = {symbol.strip("[]") for symbol in parameter_docs.UNITS}
    unknown = {}
    for key, doc in parameter_docs.PARAMETERS.items():
        for bracket in re.findall(r"\[([^\]]*)\]", doc["unit"]):
            for token in re.split(r"[/·()^0-9\s-]+", bracket):
                if token and token not in known:
                    unknown.setdefault(token, []).append(key)
    assert unknown == {}, f"unit symbols not in parameter_docs.UNITS: {unknown}"


def _equation_labels():
    labels = set()
    for name in os.listdir(FORMULATION):
        if name.endswith(".rst"):
            with open(os.path.join(FORMULATION, name), encoding="utf-8") as f:
                labels |= set(re.findall(r":label:\s*(\S+)", f.read()))
    return labels


def test_every_constraint_class_is_documented():
    from hypatia.backend.constraints.ConstraintList import CORE_CONSTRAINTS, OPTIONAL_CONSTRAINTS
    classes = {cls.__name__ for cls in CORE_CONSTRAINTS + list(OPTIONAL_CONSTRAINTS.values())}
    assert sorted(classes - set(parameter_docs.CONSTRAINTS)) == [], "constraint classes missing"
    assert sorted(set(parameter_docs.CONSTRAINTS) - classes) == [], "documented classes that no longer exist"


def test_equations_of_every_constraint_exist():
    labels = _equation_labels()
    missing = {name: [label for label in eq_labels if label not in labels]
               for name, (eq_labels, _) in parameter_docs.CONSTRAINTS.items()}
    missing = {name: m for name, m in missing.items() if m}
    assert missing == {}, f"equation labels not found in doc/source/formulation: {missing}"


needs_highs = pytest.mark.skipif("HIGHS" not in cp.installed_solvers(),
                                 reason="the documentation examples use HiGHS")


@needs_highs
@pytest.mark.parametrize("script", ["first_run.py", "planning_tutorial.py", "operation_tutorial.py"])
def test_documentation_example_runs(script, tmp_path, monkeypatch, capsys):
    """Run the script in an empty folder, as a reader would."""
    monkeypatch.chdir(tmp_path)
    runpy.run_path(os.path.join(EXAMPLES, script), run_name="__main__")
    assert capsys.readouterr().out  # every script prints what the page shows
