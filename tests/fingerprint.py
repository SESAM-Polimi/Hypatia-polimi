"""
Solver-free fingerprint of a built Hypatia model.

Why
---
The example regression tests (test_examples.py) compare optimal objectives.
They cannot see a change in a constraint that is not binding at the optimum,
nor in code that no example builds. Refactorings such as the cvxpy upgrade
touch exactly that code, so here we check the *built* model instead of the
solved one.

How
---
Every constraint and every expression Hypatia builds is an affine function of
the decision variables, f(x) = A x + b. For each of them we compute

* the constant part,  C = f(0) = b
* the variable part,  L = (f(s x) - f(0)) / s = A x, at a random point x

Two code versions that give the same values at the same random point build the
same A and b, except with negligible probability.

Two numerical details:

* s = 2**40. In a row like "x <= 1e10", f(x) - f(0) would lose about six
  digits of A x to cancellation against b. At s x the variable part dominates,
  and multiplying and dividing by a power of two is exact in floating point.
* Each series is summarised by order-invariant statistics, so rows generated
  in a different order are not reported. Plain sums are dominated by the
  largest rows; the sum of log|value| is not, so it catches a small relative
  change in a small row (say an efficiency off by 1e-6 next to a 1e10 limit).

A third series, M = L/scale_L + C/scale_C, ties each variable part to its own
constant: it changes if, say, the demand of hour t ends up in the balance of
hour t+1, even though the sets of L and C values stay the same.

For equality constraints a == b and b == a are the same constraint, so their
series are taken in absolute value.

Variables are matched between two runs by creation order (cvxpy ids grow as
variables are created). The fingerprint stores the shape and type of every
variable, so if that list changes the comparison says so first: residuals
would then be evaluated at different points and mean nothing.

Usage
-----
In tests: ``fingerprint(model)`` and ``compare(reference, current)``.

To (re)write the stored references after a change that is *meant* to alter the
built model (say why in the PR):

    python -m tests.fingerprint update             all cases
    python -m tests.fingerprint update Atlantis    one case

For a model that is not in the repository (e.g. a JRP scenario):

    from tests.fingerprint import fingerprint, save, load, compare
    save(fingerprint(model), "before.json")                   # old code or environment
    print(compare(load("before.json"), fingerprint(model)))   # new one
"""
import json
import sys

import cvxpy as cp
import numpy as np

SEED = 20261019
SCALE = 2.0 ** 40
QUANTILES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

# Values are normalised by their largest absolute value, so they lie in
# [-1, 1]. Rounding differences between numpy/cvxpy versions are around 1e-16
# per value, so a sum of n values can move by about n * 1e-16. We allow
# ATOL_PER_ROW per value: a margin of 1e4 over rounding, and still enough to
# see one value change by 1e-7 in a block of 1e5 rows.
# The sums get only this absolute tolerance. A tolerance relative to the sum
# itself would be wrong: log_sum is often around -1e4, and 1e-9 of that
# would hide a change of 1e-6 in a few dozen rows.
RTOL = 1e-9
ATOL_PER_ROW = 1e-12
# Values below this fraction of the largest one are treated as zero by the
# log statistic (rounding noise around zero must not count).
LOG_FLOOR = {"L": 1e-12, "C": 1e-12, "M": 1e-6}

# Attributes that are not part of the built model.
_SKIP_ATTRIBUTES = {"model_data", "vars", "constr", "constraints_by_class"}


# ----------------------------------------------------------------- building --

def build(model):
    """Build constraints, variables and objectives of a Model, without solving."""
    from hypatia.backend.Build import BuildModel
    return BuildModel(model._Model__model_data)


def collect_items(built):
    """Everything to fingerprint, as {name: expression or list of constraints}.

    Constraints are grouped by the class that made them and by kind (Equality
    or Inequality). Expressions are all the cvxpy objects reachable from the
    attributes of the model variables (production, costs, emissions, ...) and
    of the built model (objectives).
    """
    items = {}
    for cls_name, rules in built.constraints_by_class.items():
        for constraint in rules:
            key = f"constraint/{cls_name}/{type(constraint).__name__}"
            items.setdefault(key, []).append(constraint)

    def walk(obj, path):
        if isinstance(obj, cp.Expression):
            items[path] = obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, f"{path}/{key}")
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                walk(value, f"{path}/{i}")

    for name, value in vars(built.vars).items():
        if name not in _SKIP_ATTRIBUTES:
            walk(value, f"variables/{name}")
    for name, value in vars(built).items():
        if name not in _SKIP_ATTRIBUTES:
            walk(value, f"model/{name}")
    return items


# --------------------------------------------------------------- evaluation --

def _variables_of(item):
    if isinstance(item, list):
        return [v for constraint in item for v in constraint.variables()]
    return item.variables()


def _describe(variable):
    kind = [name for name in ("boolean", "integer", "nonneg", "nonpos")
            if variable.attributes.get(name)]
    return f"{'x'.join(map(str, variable.shape)) or 'scalar'} {'/'.join(kind) or 'free'}"


def _random_value(variable, rng):
    """Random point: 0/1 for booleans, small integers for integers, values in
    [0.5, 1.5] otherwise (away from zero, so no row vanishes by accident)."""
    shape = variable.shape
    if variable.attributes.get("boolean"):
        return rng.integers(0, 2, size=shape).astype(float)
    if variable.attributes.get("integer"):
        return rng.integers(0, 6, size=shape).astype(float)
    sign = -1.0 if variable.attributes.get("nonpos") else 1.0
    return sign * rng.uniform(0.5, 1.5, size=shape)


def _set_values(variables, values):
    # save_value skips cvxpy's domain check, which would reject a boolean
    # equal to 2**40. We only evaluate expressions, we never solve.
    for v, value in zip(variables, values):
        v.save_value(np.asarray(value, dtype=float).reshape(v.shape))


def _values(item):
    """Flat array of the values of an expression, or of the residuals
    (lhs - rhs) of a list of constraints."""
    if isinstance(item, list):
        parts = [np.ravel(np.asarray(c.expr.value, dtype=float)) for c in item]
        return np.concatenate(parts) if parts else np.zeros(0)
    return np.ravel(np.asarray(item.value, dtype=float))


# -------------------------------------------------------------- statistics --

def summarise(values, log_floor=1e-12):
    """Order-invariant summary of a series of numbers."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    out = {"n": int(values.size), "nonfinite": int(values.size - finite.size)}
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    out["scale"] = scale
    if scale > 0:
        u = finite / scale
        out.update(
            sum=float(np.sum(u)),
            abs_sum=float(np.sum(np.abs(u))),
            sq_sum=float(np.sum(u ** 2)),
            cube_sum=float(np.sum(u ** 3)),
            log_sum=float(np.sum(np.log(np.maximum(np.abs(u), log_floor)))),
            min=float(np.min(u)),
            max=float(np.max(u)),
            quantiles=[float(q) for q in np.quantile(u, QUANTILES)],
        )
    return out


def _largest(values):
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _series(linear, constant, symmetric):
    mixed = np.zeros_like(linear)
    for part in (linear, constant):
        scale = _largest(part)
        if scale > 0:
            mixed = mixed + part / scale
    series = {"L": linear, "C": constant, "M": mixed}
    if symmetric:
        series = {name: np.abs(values) for name, values in series.items()}
    return {name: summarise(values, LOG_FLOOR[name]) for name, values in series.items()}


# ------------------------------------------------------------- fingerprint --

def fingerprint(model, seed=SEED):
    """Build the model and return its fingerprint (a JSON-serialisable dict)."""
    return fingerprint_items(collect_items(build(model)), seed)


def fingerprint_items(items, seed=SEED):
    """Fingerprint of {name: expression or list of constraints}.

    Kept separate from fingerprint() so it can be tested on small hand-made
    expressions, without building a Hypatia model.
    """
    variables = {}
    for item in items.values():
        for v in _variables_of(item):
            variables[v.id] = v
    ordered = [variables[i] for i in sorted(variables)]

    _set_values(ordered, [np.zeros(v.shape) for v in ordered])
    constant = {name: _values(item) for name, item in items.items()}

    rng = np.random.default_rng(seed)
    _set_values(ordered, [SCALE * _random_value(v, rng) for v in ordered])
    linear = {name: (_values(item) - constant[name]) / SCALE for name, item in items.items()}

    result = {
        "created_with": {"cvxpy": cp.__version__, "numpy": np.__version__},
        "variables": [_describe(v) for v in ordered],
        "items": {},
    }
    for name in sorted(items):
        symmetric = name.startswith("constraint/") and name.endswith("/Equality")
        result["items"][name] = _series(linear[name], constant[name], symmetric)
    return result


# --------------------------------------------------------------- comparison --

def _close(a, b, atol, rtol=RTOL):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _compare_summary(where, ref, new):
    if ref["n"] != new["n"]:
        return [f"{where}: {ref['n']} values in the reference, {new['n']} now"]
    if ref["nonfinite"] != new["nonfinite"]:
        return [f"{where}: {ref['nonfinite']} non-finite values in the reference, {new['nonfinite']} now"]
    if not _close(ref["scale"], new["scale"], 0.0):
        return [f"{where}: largest |value| {ref['scale']:.17g} in the reference, {new['scale']:.17g} now"]
    if ref["scale"] == 0:
        return []
    atol_sum = ATOL_PER_ROW * max(ref["n"], 1)
    diffs = []
    for stat in ("sum", "abs_sum", "sq_sum", "cube_sum", "log_sum"):
        if not _close(ref[stat], new[stat], atol_sum, rtol=0.0):
            diffs.append(f"{where}: {stat} {ref[stat]:.17g} in the reference, {new[stat]:.17g} now")
    for stat in ("min", "max"):
        if not _close(ref[stat], new[stat], ATOL_PER_ROW):
            diffs.append(f"{where}: {stat} {ref[stat]:.17g} in the reference, {new[stat]:.17g} now")
    for q, a, b in zip(QUANTILES, ref["quantiles"], new["quantiles"]):
        if not _close(a, b, ATOL_PER_ROW):
            diffs.append(f"{where}: quantile {q} {a:.17g} in the reference, {b:.17g} now")
    return diffs


def compare(ref, new):
    """List of differences between two fingerprints (empty if equivalent)."""
    if ref["variables"] != new["variables"]:
        n_ref, n_new = len(ref["variables"]), len(new["variables"])
        first = next((i for i, (a, b) in enumerate(zip(ref["variables"], new["variables"])) if a != b),
                     min(n_ref, n_new))
        return [f"variables differ ({n_ref} in the reference, {n_new} now); first difference at "
                f"position {first}. Residuals are not compared, since they would be evaluated at "
                f"different points."]
    diffs = []
    for name in sorted(set(ref["items"]) | set(new["items"])):
        if name not in new["items"]:
            diffs.append(f"{name}: in the reference, missing now")
        elif name not in ref["items"]:
            diffs.append(f"{name}: new, not in the reference")
        else:
            for part in ("L", "C", "M"):
                diffs += _compare_summary(f"{name} [{part}]", ref["items"][name][part],
                                          new["items"][name][part])
    return diffs


def save(fp, path):
    with open(path, "w") as f:
        json.dump(fp, f, indent=1, sort_keys=True)
        f.write("\n")


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------- CLI --

def _update(names):
    from tests.fingerprint_cases import CASES, load_case, reference_path
    for name in names or list(CASES):
        fp = fingerprint(load_case(name))
        save(fp, reference_path(name))
        print(f"{name}: {len(fp['variables'])} variables, {len(fp['items'])} items")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "update":
        sys.exit(__doc__)
    _update(sys.argv[2:])
