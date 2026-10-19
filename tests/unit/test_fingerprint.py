"""
Tests of the fingerprint method itself (tests/fingerprint.py), on tiny
hand-made problems: a check that never fails is worse than no check, so we
verify that harmless rewrites pass and real changes are caught.

Each builder creates fresh variables, as a new build of the model would.
"""
import cvxpy as cp
import numpy as np

from tests.fingerprint import compare, fingerprint_items

T = 6
DEMAND = np.arange(1.0, T + 1)
COST = np.linspace(2.0, 3.0, T)


def balance_model(demand=DEMAND, cost=COST, split=False, flipped=False):
    production = cp.Variable(T, nonneg=True)
    if split:  # same rows, generated in two pieces and in reverse order
        rows = [production[3:] >= demand[3:], production[:3] >= demand[:3]]
    else:
        rows = [production >= demand]
    storage = cp.Variable(T)
    equality = (storage == production - demand) if not flipped else (production - demand == storage)
    return {
        "constraint/Balance/Inequality": rows,
        "constraint/Storage/Equality": [equality],
        "model/cost": cost @ production,
    }


def differences(reference_items, new_items):
    return compare(fingerprint_items(reference_items), fingerprint_items(new_items))


def test_same_model_built_twice():
    assert differences(balance_model(), balance_model()) == []


def test_row_order_and_side_of_equality_do_not_matter():
    assert differences(balance_model(), balance_model(split=True, flipped=True)) == []


def test_small_coefficient_change_is_caught():
    diffs = differences(balance_model(), balance_model(cost=COST * (1 + 1e-6)))
    assert any(d.startswith("model/cost") for d in diffs)


def test_shifted_time_series_is_caught_by_the_mixed_series():
    # Demand of hour t ends up in the balance of hour t+1. The values of the
    # variable part and of the constant part are the same sets as before, so
    # only the mixed series M, which pairs them row by row, can see it.
    diffs = differences(balance_model(), balance_model(demand=np.roll(DEMAND, 1)))
    balance = [d for d in diffs if d.startswith("constraint/Balance")]
    assert balance
    assert all("[M]" in d for d in balance)


def test_coefficient_hidden_behind_a_large_constant_is_caught():
    def model(coefficient):
        x = cp.Variable(3, nonneg=True)
        return {"constraint/Limit/Inequality": [coefficient * x <= 1e10]}
    assert differences(model(1.0), model(1.0 + 1e-6))


def test_reshape_order_is_caught():
    # The risk in the cvxpy upgrade: the default order of cp.reshape changed.
    limits = np.arange(12.0)

    def model(order):
        x = cp.Variable((3, 4), nonneg=True)
        return {"constraint/Limit/Inequality": [cp.reshape(x, (12,), order=order) <= limits]}
    assert differences(model("F"), model("F")) == []
    assert differences(model("F"), model("C"))


def test_changed_variables_are_reported_before_residuals():
    def model(n):
        x = cp.Variable(n, nonneg=True)
        return {"constraint/Limit/Inequality": [x <= 1]}
    diffs = differences(model(3), model(4))
    assert len(diffs) == 1 and diffs[0].startswith("variables differ")


def test_boolean_and_integer_variables_get_values_in_their_domain():
    b = cp.Variable(5, boolean=True)
    n = cp.Variable(5, integer=True)
    fp = fingerprint_items({"model/units": b + n})
    assert fp["variables"] == ["5 boolean", "5 integer"]


def test_small_change_in_a_small_row_is_caught():
    # The case that the first version of the method missed: a 1e-6 change in
    # an efficiency of 495 inside a block whose largest value is ~7e4. The
    # sums barely move; the sum of log|value| moves by 1e-6 per row changed.
    # The 24 rows sit between smaller and larger ones, away from min, max and
    # the quantiles, as in the real model.
    def model(efficiency):
        large = cp.Variable(1000, nonneg=True)
        small = cp.Variable(5000, nonneg=True)
        y = cp.Variable(24, nonneg=True)
        return {"constraint/TechEfficency/Equality": [7e4 * large == 0,
                                                      efficiency * y == 0,
                                                      1.0 * small == 0]}
    assert differences(model(495.0), model(495.0)) == []
    assert differences(model(495.0), model(495.0 * (1 + 1e-6)))


def test_equivalent_rows_next_to_large_constants_are_not_reported():
    # x - 1e10 and (x - 5e9) - 5e9 are the same function. Computing the
    # variable part as f(x) - f(0) would lose six digits to cancellation and
    # could report a difference; evaluating at 2**40 x does not.
    def model(split):
        x = cp.Variable(50, nonneg=True)
        lhs = (x - 5e9) - 5e9 if split else x - 1e10
        return {"constraint/Limit/Inequality": [lhs <= 0]}
    assert differences(model(False), model(True)) == []
