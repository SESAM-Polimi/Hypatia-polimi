from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd

from hypatia.backend.constraints.MinTechPowerOutput import MinTechPowerOutput
from hypatia.utility.constants import EnsureFeasibility, ModelMode


def _technology_parameter(value):
    return pd.DataFrame(
        [[value]],
        columns=pd.MultiIndex.from_tuples([("Supply", "plant")]),
    )


def test_operation_rules_do_not_require_modular_capacity_parameter():
    settings = SimpleNamespace(
        mode=ModelMode.Operation,
        ensure_feasibility=EnsureFeasibility.No,
        optional_constraints=("min_tech_power_output",),
        multi_node=False,
        regions=["reg1"],
        years=["Y0"],
        time_steps=["T0"],
        timeslice_fraction=1,
        technologies={"reg1": {"Supply": ["plant"]}},
    )
    regional_parameters = {
        "reg1": {
            "min_tech_power_output": _technology_parameter(0.2),
            "res_capacity_factor": pd.DataFrame(
                [[1.0]],
                index=pd.MultiIndex.from_tuples([("Y0", "T0")]),
                columns=pd.MultiIndex.from_tuples([("Supply", "plant")]),
            ),
            "annualprod_per_unitcapacity": _technology_parameter(8760.0),
        }
    }
    model_data = SimpleNamespace(
        settings=settings,
        regional_parameters=regional_parameters,
        global_parameters=None,
    )
    variables = SimpleNamespace(
        totalcapacity={
            "reg1": {"Supply": cp.Constant(np.array([[1.0]]))},
        },
        technology_prod={
            "reg1": {"Supply": cp.Variable((1, 1), nonneg=True)},
        },
        boolean_for_minpower={
            "reg1": {"Supply": cp.Variable((1, 1), boolean=True)},
        },
        units_online={},
        units_online_technologies={},
    )

    rules = MinTechPowerOutput(model_data, variables).get()

    assert len(rules) == 2


def test_planning_modular_technology_uses_four_unit_level_constraints():
    settings = SimpleNamespace(
        mode=ModelMode.Planning,
        ensure_feasibility=EnsureFeasibility.No,
        optional_constraints=("min_tech_power_output",),
        multi_node=False,
        regions=["reg1"],
        years=["Y0"],
        time_steps=["T0"],
        timeslice_fraction=1,
        technologies={"reg1": {"Supply": ["plant"]}},
    )
    regional_parameters = {
        "reg1": {
            "new_capacity_step": _technology_parameter(10.0),
            "min_tech_power_output": _technology_parameter(0.2),
            "res_capacity_factor": pd.DataFrame(
                [[1.0]],
                index=pd.MultiIndex.from_tuples([("Y0", "T0")]),
                columns=pd.MultiIndex.from_tuples([("Supply", "plant")]),
            ),
            "annualprod_per_unitcapacity": _technology_parameter(8760.0),
        }
    }
    model_data = SimpleNamespace(
        settings=settings,
        regional_parameters=regional_parameters,
        global_parameters=None,
    )
    variables = SimpleNamespace(
        totalcapacity={
            "reg1": {"Supply": cp.Constant(np.array([[10.0]]))},
        },
        technology_prod={
            "reg1": {"Supply": cp.Variable((1, 1), nonneg=True)},
        },
        boolean_for_minpower={
            "reg1": {"Supply": cp.Variable((1, 1), boolean=True)},
        },
        units_online={
            "reg1": {"Supply": cp.Variable((1, 1), integer=True)},
        },
        units_online_technologies={"reg1": {"Supply": ["plant"]}},
    )

    rules = MinTechPowerOutput(model_data, variables).get()

    assert len(rules) == 4
