from types import SimpleNamespace

import pandas as pd

from hypatia.backend.ModelVariables import ModelVariables
from hypatia.utility.constants import ModelMode


def _settings(mode, optional_constraints):
    return SimpleNamespace(
        mode=mode,
        optional_constraints=optional_constraints,
        multi_node=False,
        regions=["reg1"],
        years=["Y0"],
        time_steps=["T0", "T1"],
        technologies={
            "reg1": {
                "Supply": ["modular", "continuous"],
                "Demand": ["load"],
            }
        },
    )


def _parameter(values):
    columns = pd.MultiIndex.from_tuples(
        [("Supply", "modular"), ("Supply", "continuous")]
    )
    return pd.DataFrame([values], columns=columns)


def _variables(mode, optional_constraints, regional_parameters=None):
    model_data = SimpleNamespace(
        settings=_settings(mode, optional_constraints),
        regional_parameters=regional_parameters,
        global_parameters=None,
    )
    return SimpleNamespace(model_data=model_data)


def test_operation_does_not_read_planning_only_modular_capacity():
    variables = _variables(
        ModelMode.Operation,
        ("min_tech_power_output",),
        regional_parameters=None,
    )

    units_online, technologies = ModelVariables.create_units_online(variables)

    assert units_online == {}
    assert technologies == {}


def test_disabled_constraint_does_not_create_modular_variables():
    variables = _variables(
        ModelMode.Planning,
        (),
        regional_parameters=None,
    )

    units_online, technologies = ModelVariables.create_units_online(variables)

    assert units_online == {}
    assert technologies == {}


def test_planning_creates_integer_variable_only_for_eligible_technology():
    regional_parameters = {
        "reg1": {
            "new_capacity_step": _parameter([10.0, 0.0]),
            "min_tech_power_output": _parameter([0.2, 0.2]),
        }
    }
    variables = _variables(
        ModelMode.Planning,
        ("min_tech_power_output",),
        regional_parameters=regional_parameters,
    )

    units_online, technologies = ModelVariables.create_units_online(variables)

    assert technologies == {"reg1": {"Supply": ["modular"]}}
    variable = units_online["reg1"]["Supply"]
    assert variable.shape == (2, 1)
    assert variable.attributes["integer"] is True


def test_zero_minimum_output_is_not_eligible():
    regional_parameters = {
        "reg1": {
            "new_capacity_step": _parameter([10.0, 0.0]),
            "min_tech_power_output": _parameter([0.0, 0.2]),
        }
    }
    variables = _variables(
        ModelMode.Planning,
        ("min_tech_power_output",),
        regional_parameters=regional_parameters,
    )

    units_online, technologies = ModelVariables.create_units_online(variables)

    assert units_online == {"reg1": {}}
    assert technologies == {"reg1": {}}
