"""
Regression tests: every example must solve to optimality with the same
objective value as the reference run.

Reference values: Gurobi, cvxpy 1.1.18, 19 October 2026 (examples as merged in
PR #24). The six "fast" examples gave the same objective with HiGHS to within
1e-15 relative, so CI checks them with HiGHS (no licence needed).

When a change is *meant* to alter a result, update the value here in the same
PR and say why in the PR description: a moved reference should never be a
surprise.

Run:
    pytest tests/test_examples.py             fast examples (HiGHS)
    pytest tests/test_examples.py -m slow     slow ones (Gurobi if installed)
Solver override: HYPATIA_TEST_SOLVER=GUROBI pytest ...
"""
import os

import cvxpy as cp
import pytest

from hypatia import load_example

REL_TOL = 1e-6

REFERENCE = {
    # fast: seconds with HiGHS
    "Atlantis": 6528476949.951972,  # lifetimes fixed (1 -> 20 years), HiGHS, 21 October 2026
    "Neverland": 15146817372.791317,
    "Operation_1Region": 111831200.0,
    "Operation_1Region_UnMetDemand": 3.1933229033394428e16,
    "Operation_2Regions": 397340484.88107675,
    "Operation_2Regions_UnMetDemand": 2.6372606710655234e19,
    # slow: minutes with HiGHS (Planning_*), or hourly models (CHP, Minigrid)
    "Planning_1Region": 4138021288.021925,
    "Planning_1Region_UnMetDemand": 1.0480337455578235e19,
    "Planning_2Regions": 16802325902.899176,
    "Planning_2Regions_UnMetDemand": 2.1140915446761746e19,
    "Operation_1Region_CHP-Variable-range": 473311254.77475625,
    "Minigrid": 274991.343580798,
}
FAST = ["Atlantis", "Neverland", "Operation_1Region", "Operation_1Region_UnMetDemand",
        "Operation_2Regions", "Operation_2Regions_UnMetDemand"]
SLOW = [name for name in REFERENCE if name not in FAST]


def solver_options(slow):
    """Solver name and options: env override, else HiGHS for fast tests and Gurobi for slow ones."""
    name = os.environ.get("HYPATIA_TEST_SOLVER")
    if name is None:
        name = "GUROBI" if slow else "SCIPY"
    name = name.upper()
    if name not in cp.installed_solvers():
        pytest.skip(f"solver {name} not installed")
    options = {"scipy_options": {"method": "highs"}} if name == "SCIPY" else {}
    return name, options


@pytest.fixture
def solved_problems(monkeypatch):
    """Record (status, objective) of every cvxpy problem solved during the test.

    Hypatia does not return the objective value, so we wrap cvxpy's
    Problem.solve. monkeypatch puts the original back after the test.
    """
    record = []
    original = cp.Problem.solve

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        record.append((self.status, self.value))
        return result

    monkeypatch.setattr(cp.Problem, "solve", spy)
    return record


# Examples an old solver cannot solve, with the reason (skipped there).
OLD_SCIPY_FAILS = {
    # Atlantis mixes placeholder values of 1e10 (residual capacity, annual
    # production) with small costs. The HiGHS in SciPy 1.7 (legacy stack)
    # stops with "optimal in the scaled model but not in the unscaled one";
    # ECOS and OSQP fail too. SciPy 1.17 solves it.
    "Atlantis": (1, 8),
}


def check_example(name, slow, solved_problems):
    solver, options = solver_options(slow)
    if solver == "SCIPY" and name in OLD_SCIPY_FAILS:
        import scipy
        if tuple(int(part) for part in scipy.__version__.split(".")[:2]) < OLD_SCIPY_FAILS[name]:
            pytest.skip(f"{name}: badly scaled for the HiGHS of SciPy {scipy.__version__}")
    model = load_example(name)
    model.run(solver=solver, verbosity=False, force_rewrite=True, **options)

    assert solved_problems, "no optimisation problem was solved"
    status, objective = solved_problems[-1]
    assert status == "optimal"
    assert objective == pytest.approx(REFERENCE[name], rel=REL_TOL)


def test_reference_covers_every_example():
    """A new example must get a reference value (and an entry in FAST or SLOW)."""
    from hypatia import available_examples
    assert sorted(REFERENCE) == available_examples()


@pytest.mark.parametrize("name", FAST)
def test_example_objective(name, solved_problems):
    check_example(name, slow=False, solved_problems=solved_problems)


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW)
def test_slow_example_objective(name, solved_problems):
    check_example(name, slow=True, solved_problems=solved_problems)
