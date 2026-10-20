"""
Models whose built form is checked by test_build_fingerprint.py.

Together they make every constraint class produce rows at least once:
the examples as they are, two examples with optional constraints switched on,
and the feature model (tests/data/make_feature_model.py) for what no example
uses: MILP capacity, ramp limits, minimum power output, capacity groups.
"""
import os

from hypatia import available_examples, load_example

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, "data", "fingerprints")

CHP = "Operation_1Region_CHP-Variable-range"


def _example(name, optional_constraints=None):
    return lambda: load_example(name, optional_constraints=optional_constraints)


def _feature_model():
    from tests.data.make_feature_model import load
    return load()


CASES = {name: _example(name) for name in available_examples()}
CASES[CHP + "+options"] = _example(CHP, ["chp_operating_range", "electrolysis_consumption",
                                         "boolean_for_storage"])
CASES["Minigrid+boolean_for_storage"] = _example("Minigrid", ["boolean_for_storage"])
CASES["Feature_model"] = _feature_model


def load_case(name):
    return CASES[name]()


def reference_path(name):
    return os.path.join(REFERENCE_DIR, f"{name}.json")
