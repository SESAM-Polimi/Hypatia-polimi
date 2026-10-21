"""
Cost of unmet demand (ensure_feasibility="Yes"): the sheet
Unmet_demand_penalty, one value per demand carrier, default 1e10.

Until October 2026 the penalty was fixed at 1e10 in the code. Parameter files
written before then have no such sheet: they still read, with the default.
"""
import contextlib
import io

import numpy as np
import openpyxl
import pandas as pd

from hypatia import Model, download_example, load_example
from hypatia.backend.Build import BuildModel
from hypatia.backend.constraints.BalanceUnMetDemand import DEFAULT_UNMET_DEMAND_PENALTY

EXAMPLE = "Operation_1Region_UnMetDemand"  # one region, demand carrier Final_electricity


def test_files_without_the_sheet_use_the_default():
    data = load_example(EXAMPLE).get_model_data()
    penalty = data.regional_parameters["reg1"]["unmet_demand_penalty"]
    assert list(penalty.columns) == ["Final_electricity"]
    assert penalty.iloc[0, 0] == DEFAULT_UNMET_DEMAND_PENALTY


def test_value_in_the_sheet_is_read(tmp_path):
    download_example(EXAMPLE, str(tmp_path / "model"))
    model = Model(path=str(tmp_path / "model" / "sets"), mode="Operation",
                  optimization="Single", ensure_feasibility="Yes")
    model.create_data_excels(str(tmp_path / "templates"))
    # Templates hold plain values (no formulas), so openpyxl may rewrite them.
    path = tmp_path / "templates" / "parameters_reg1.xlsx"
    workbook = openpyxl.load_workbook(path)
    workbook["Unmet_demand_penalty"]["B2"] = 5e5
    workbook.save(path)
    model.read_input_data(str(tmp_path / "templates"))
    assert model.get_model_data().regional_parameters["reg1"]["unmet_demand_penalty"].iloc[0, 0] == 5e5


def test_penalty_multiplies_the_unmet_demand_in_the_cost():
    data = load_example(EXAMPLE).get_model_data()
    data.regional_parameters["reg1"]["unmet_demand_penalty"].iloc[0, 0] = 1e6
    with contextlib.redirect_stdout(io.StringIO()):
        built = BuildModel(data)
    unmet = built.vars.unmetdemandbycarrier["reg1"]["Final_electricity"]
    (variable,) = unmet.variables()
    variable.value = np.ones(variable.shape)  # one unit unmet in each of the 8760 steps
    cost = built.vars.cost_unmet_demand["reg1"]["Final_electricity"].value
    assert np.ravel(cost)[0] == 8760 * 1e6


def test_models_without_unmet_demand_do_not_get_the_sheet(tmp_path):
    download_example(EXAMPLE, str(tmp_path / "model"))
    model = Model(path=str(tmp_path / "model" / "sets"), mode="Operation",
                  optimization="Single", ensure_feasibility="No")
    model.create_data_excels(str(tmp_path / "templates"))
    sheets = pd.ExcelFile(tmp_path / "templates" / "parameters_reg1.xlsx").sheet_names
    assert "Unmet_demand_penalty" not in sheets
