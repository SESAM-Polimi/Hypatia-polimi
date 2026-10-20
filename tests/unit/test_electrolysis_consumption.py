"""
ElectrolysisConsumption: electrolysers may only use as much electricity as the
region produces from renewables in that year.

Two bugs fixed in October 2026, one test each:
- a model without electrolysers crashed (UnboundLocalError);
- in a multi-region model, a region without electrolysers reused the
  electrolysis consumption of the previous region, limited by its own
  renewable production: a wrong constraint, without any error.
"""
from types import SimpleNamespace

import cvxpy as cp
import pandas as pd
import pytest

from hypatia import load_example
from hypatia.backend.Build import BuildModel
from hypatia.backend.constraints.ElectrolysisConsumption import ElectrolysisConsumption

YEARS = ["Y0", "Y1"]
TIME_STEPS = ["1", "2", "3"]
ROWS = len(YEARS) * len(TIME_STEPS)


def two_regions_electrolysis_in_reg1_only():
    technologies = {
        "reg1": {"Supply": ["PV"], "Conversion": ["Electrolysis"]},
        "reg2": {"Supply": ["PV"]},
    }
    settings = SimpleNamespace(
        regions=["reg1", "reg2"], years=YEARS, time_steps=TIME_STEPS, technologies=technologies,
        regional_settings={
            reg: {"Carrier_output": pd.DataFrame({"Technology": ["PV"], "Carrier_out": ["Electricity"]})}
            for reg in technologies
        },
    )
    renewable = {
        "reg1": pd.DataFrame([[1, 0]], columns=pd.MultiIndex.from_tuples([("Supply", "PV"),
                                                                           ("Conversion", "Electrolysis")])),
        "reg2": pd.DataFrame([[1]], columns=pd.MultiIndex.from_tuples([("Supply", "PV")])),
    }
    model_data = SimpleNamespace(
        settings=settings,
        regional_parameters={reg: {"renewable_tech": renewable[reg]} for reg in technologies},
    )
    production = {reg: {cat: cp.Variable((ROWS, len(techs)), name=f"prod_{reg}_{cat}")
                        for cat, techs in technologies[reg].items()} for reg in technologies}
    use = {reg: {cat: cp.Variable((ROWS, len(techs)), name=f"use_{reg}_{cat}")
                 for cat, techs in technologies[reg].items() if cat != "Supply"} for reg in technologies}
    variables = SimpleNamespace(
        technology_prod=production, technology_use=use,
        totalprodbycarrier={reg: {"Electricity": None} for reg in technologies},
    )
    return model_data, variables


def test_only_regions_with_electrolysers_are_constrained():
    model_data, variables = two_regions_electrolysis_in_reg1_only()
    rules = ElectrolysisConsumption(model_data, variables).rules()

    assert len(rules) == 1
    names = {v.name() for v in rules[0].variables()}
    assert names == {"use_reg1_Conversion", "prod_reg1_Supply"}


@pytest.mark.parametrize("example", ["Atlantis", "Operation_2Regions"])
def test_models_without_electrolysers_build(example):
    model = load_example(example, optional_constraints=["electrolysis_consumption"])
    built = BuildModel(model._Model__model_data)
    assert built.constraints_by_class["ElectrolysisConsumption"] == []
