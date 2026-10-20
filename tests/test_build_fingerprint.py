"""
The built model (every constraint, objective and result expression) must be
the same as in the stored reference, evaluated at a random point without
solving. See tests/fingerprint.py for the method.

This is the test that guards refactorings such as the cvxpy upgrade: it sees
changes in constraints that are not binding in the examples, and in parts of
the code that only the feature model builds.

When a change is *meant* to alter the built model, regenerate the reference
of the cases concerned in the same PR and say why:

    python -m tests.fingerprint update <case> [<case> ...]
"""
import contextlib
import io
import os

import pytest

from tests.fingerprint import compare, fingerprint, load
from tests.fingerprint_cases import CASES, load_case, reference_path

MAX_SHOWN = 25


def quiet(function, *args):
    # Some constraints print messages while being built.
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


@pytest.mark.parametrize("case", sorted(CASES))
def test_built_model_matches_reference(case):
    path = reference_path(case)
    assert os.path.exists(path), f"no reference for {case}: python -m tests.fingerprint update {case}"
    current = quiet(lambda: fingerprint(load_case(case)))
    diffs = compare(load(path), current)
    shown = "\n".join(diffs[:MAX_SHOWN])
    more = f"\n... and {len(diffs) - MAX_SHOWN} more" if len(diffs) > MAX_SHOWN else ""
    assert not diffs, f"{case}: the built model changed\n{shown}{more}"


def test_every_constraint_class_is_covered():
    """Each constraint class must produce rows in at least one case; otherwise
    nothing above would notice a change in it."""
    from hypatia.backend.constraints.ConstraintList import CORE_CONSTRAINTS, OPTIONAL_CONSTRAINTS
    expected = {cls.__name__ for cls in CORE_CONSTRAINTS + list(OPTIONAL_CONSTRAINTS.values())}
    covered = set()
    for case in CASES:
        items = load(reference_path(case))["items"]
        covered |= {name.split("/")[1] for name, item in items.items()
                    if name.startswith("constraint/") and item["L"]["n"] > 0}
    assert sorted(expected - covered) == []
