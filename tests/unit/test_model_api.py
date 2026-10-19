import pandas as pd

from hypatia import load_example

'''
Tests for public Model methods that are not covered by the example regression
tests (they do not affect the optimisation result).
'''


def test_print_model():
    model = load_example("Operation_1Region")
    text = str(model)
    assert "Operation_1Region" in text
    assert "optional constraints= none" in text


def test_create_config_file(tmp_path):
    # tmp_path: a fresh temporary folder that pytest creates for this test
    model = load_example("Neverland")
    out = tmp_path / "config.xlsx"
    model.create_config_file(str(out))

    sheets = pd.read_excel(out, sheet_name=None)
    assert {"Techs", "Fuels", "Regions", "Emissions"} <= set(sheets)
    assert len(sheets["Regions"]) == 2  # Neverland has two regions
