"""
Which solver outcomes give results, and the minimum-emission runs.

check_solution keeps an optimal solution, keeps an inaccurate one or one
stopped at a limit (with a warning), and raises NoSolution otherwise. Until
October 2026 anything but "optimal" printed a message and left the results
empty, without an error.

The second step of run_MinEm and the last point of run_MO bound the emissions
by their minimum. With the exact minimum, HiGHS reports Atlantis infeasible,
so run_MinEm returned None values and run_MO lost its last solution; the
bound now has a relative slack of 1e-6 (emission_bound).
"""
import contextlib
import io
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from hypatia import load_example
from hypatia.backend.Build import EMISSION_SLACK, check_solution, emission_bound
from hypatia.error_log.Exceptions import NoSolution

needs_highs = pytest.mark.skipif("HIGHS" not in cp.installed_solvers(), reason="uses HiGHS")


def solved(status, value=1.0, has_values=True):
    """A stand-in for a solved cvxpy problem."""
    variable = SimpleNamespace(value=np.ones(2) if has_values else None)
    return SimpleNamespace(status=status, value=value, variables=lambda: [variable])


def test_optimal_is_kept_silently(recwarn):
    check_solution(solved(cp.OPTIMAL), "run")
    assert not recwarn.list


@pytest.mark.parametrize("status", [cp.OPTIMAL_INACCURATE, cp.USER_LIMIT])
def test_usable_but_not_optimal_is_kept_with_a_warning(status):
    with pytest.warns(UserWarning, match=status):
        check_solution(solved(status), "run")


@pytest.mark.parametrize("problem", [
    solved(cp.USER_LIMIT, value=None, has_values=False),  # stopped before any feasible point
    solved(cp.INFEASIBLE, value=np.inf, has_values=False),
    solved(cp.UNBOUNDED, value=-np.inf, has_values=False),
])
def test_no_usable_solution_raises(problem):
    with pytest.raises(NoSolution, match=problem.status):
        check_solution(problem, "run")


def test_infeasible_cvxpy_problem_raises():
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
    problem.solve()
    with pytest.raises(NoSolution, match="infeasible"):
        check_solution(problem, "The model")


@pytest.mark.parametrize("minimum", [5.6e6, 0.0, -3.0e4])
def test_emission_bound_is_slightly_above_the_minimum(minimum):
    bound = emission_bound(minimum)
    assert bound > minimum
    assert bound - minimum == pytest.approx(EMISSION_SLACK * max(abs(minimum), 1))


@needs_highs
def test_min_em_returns_values_on_atlantis():
    model = load_example("Atlantis", optimization="MinEm")
    with contextlib.redirect_stdout(io.StringIO()):
        model.run_MinEm(solver="highs", verbosity=False)
    emissions = float(np.ravel(model.results.tot_emissions.value)[0])
    assert emissions == pytest.approx(5623333.33, rel=1e-6)


@needs_highs
def test_run_mo_saves_every_point_on_atlantis(tmp_path):
    model = load_example("Atlantis", optimization="Multi")
    with contextlib.redirect_stdout(io.StringIO()):
        model.run_MO(solver="highs", number_solutions=3, path=str(tmp_path), verbosity=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1", "2", "3"]


def test_run_min_em_needs_min_em_or_multi():
    model = load_example("Atlantis")  # optimization="Single"
    with pytest.raises(ValueError, match="'MinEm' or 'Multi'"):
        model.run_MinEm(solver="highs")
