"""
Checks on the inputs and on how a model is run: each test shows the mistake
and the message Hypatia gives for it (October 2026; before, most of these
failed late with an unrelated error, or not at all).
"""
import contextlib
import io
import os
import warnings

import pandas as pd
import pytest

from hypatia import Model, load_example
from hypatia.backend.ModelSettings import ModelSettings
from hypatia.core.main import _prepare_results_folder
from hypatia.error_log.Exceptions import ResultOverWrite, WrongNumberOfYears, WrongTechCategory
from hypatia.examples import load_examples
from hypatia.utility.constants import EnsureFeasibility, ModelMode, OptimizationMode


def sets(fractions=(0.5, 0.5), demand=True):
    """Set tables of a one-region model: supply -> conversion -> demand."""
    techs = [("gas", "Supply"), ("plant", "Conversion")] + ([("load", "Demand")] if demand else [])
    names = [name for name, _ in techs]
    categories = [category for _, category in techs]
    carriers = {"Carrier": ["fuel", "elec"], "Carr_name": ["fuel", "elec"],
                "Carr_type": ["Resource", "Demand"]}
    global_settings = {
        "Regions": pd.DataFrame({"Region": ["reg1"], "Region_name": ["reg1"]}),
        "Years": pd.DataFrame({"Year": ["Y0"], "Year_name": [2030]}),
        "Timesteps": pd.DataFrame({"Timeslice": [1, 2], "Timeslice_name": ["a", "b"],
                                   "Timeslice_fraction": list(fractions)}),
        "Carriers_glob": pd.DataFrame({**carriers, "Carr_unit": ["MWh", "MWh"]}),
        "Technologies_glob": pd.DataFrame({"Technology": names, "Tech_name": names,
                                           "Tech_category": categories,
                                           "Tech_cap_unit": ["MW"] * len(names),
                                           "Tech_act_unit": ["MWh"] * len(names)}),
        "Emissions": pd.DataFrame({"Emission": ["CO2"], "Emission_name": ["CO2"],
                                   "Emission_unit": ["t"]}),
    }
    inputs = [("plant", "fuel")] + ([("load", "elec")] if demand else [])
    regional = {"reg1": {
        "Technologies": pd.DataFrame({"Technology": names, "Tech_name": names,
                                      "Tech_category": categories}),
        "Carriers": pd.DataFrame(carriers),
        "Carrier_input": pd.DataFrame(inputs, columns=["Technology", "Carrier_in"]),
        "Carrier_output": pd.DataFrame([("gas", "fuel"), ("plant", "elec")],
                                       columns=["Technology", "Carrier_out"]),
    }}
    return global_settings, regional


def settings(**kwargs):
    global_settings, regional = sets(**kwargs)
    return ModelSettings(ModelMode.Operation, OptimizationMode.Single, EnsureFeasibility.No,
                         global_settings, regional)


def example_sets(name):
    return os.path.join(load_examples.path, name, "sets")


# --- sets ---------------------------------------------------------------------------


def test_operation_model_with_several_years_is_rejected():
    with pytest.raises(WrongNumberOfYears, match="exactly one year.*has 11"):
        Model(path=example_sets("Atlantis"), mode="Operation", optimization="Single",
              ensure_feasibility="No")


def test_region_without_demand_technology_is_rejected():
    with pytest.raises(WrongTechCategory, match="'reg1' has no technology of category 'Demand'"):
        settings(demand=False)


def test_time_step_fractions_not_adding_up_to_one_warn():
    with pytest.warns(UserWarning, match="add up to 0.9"):
        settings(fractions=(0.5, 0.4))


def test_time_step_fractions_that_are_not_numbers_warn_instead_of_crashing():
    with pytest.warns(UserWarning, match="not numbers"):
        settings(fractions=("0.5", "half"))


def test_time_step_fractions_adding_up_to_one_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settings(fractions=(0.5, 0.5))


# --- parameter files ------------------------------------------------------------------

CHP = "Operation_1Region_CHP-Variable-range"


def test_filled_sheet_of_disabled_option_warns():
    """The case of the RES-4 rerun: a filled CHP sheet, option not enabled."""
    with pytest.warns(UserWarning, match="'CHP_variable_techs_selection'.*'chp_operating_range'"):
        load_example(CHP, optional_constraints=[])


def test_sheets_left_at_their_default_do_not_warn():
    """The CHP example also holds ramp sheets at their default (1, no limit):
    ignoring them changes nothing, so no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_example(CHP)  # chp_operating_range enabled, production_ramp not
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


# --- results folder ----------------------------------------------------------------


def test_missing_parent_folders_are_created(tmp_path):
    target = tmp_path / "results" / "scenario" / "run1"
    _prepare_results_folder(str(target), force_rewrite=False)
    assert target.is_dir()


def test_existing_empty_folder_is_used(tmp_path):
    _prepare_results_folder(str(tmp_path), force_rewrite=False)


def test_non_empty_folder_needs_force_rewrite(tmp_path):
    (tmp_path / "tech_cost.csv").write_text("x")
    with pytest.raises(ResultOverWrite, match="force_rewrite=True"):
        _prepare_results_folder(str(tmp_path), force_rewrite=False)


def test_force_rewrite_removes_previous_results(tmp_path):
    (tmp_path / "old_quantity.csv").write_text("x")
    (tmp_path / "total_capacity" / "reg1").mkdir(parents=True)
    (tmp_path / "total_capacity" / "reg1" / "Supply.csv").write_text("x")
    _prepare_results_folder(str(tmp_path), force_rewrite=True)
    assert list(tmp_path.iterdir()) == []


def test_folder_with_other_files_is_never_emptied(tmp_path):
    """A mistyped path (say, the project folder) must not be deleted."""
    (tmp_path / "results.csv").write_text("x")
    (tmp_path / "notes.docx").write_text("x")
    with pytest.raises(ResultOverWrite, match="notes.docx"):
        _prepare_results_folder(str(tmp_path), force_rewrite=True)
    assert (tmp_path / "notes.docx").exists() and (tmp_path / "results.csv").exists()


# --- run_MO arguments, checked before solving ----------------------------------------


def test_run_mo_needs_multi_optimization(tmp_path):
    model = load_example("Atlantis")  # optimization="Single"
    with pytest.raises(ValueError, match="optimization='Multi'.*'Single'"):
        model.run_MO(solver="highs", number_solutions=3, path=str(tmp_path))


@pytest.mark.parametrize("number", [1, 0, 2.5, True])
def test_run_mo_needs_at_least_two_solutions(number, tmp_path):
    model = load_example("Atlantis", optimization="Multi")
    with pytest.raises(ValueError, match="at least 2"):
        model.run_MO(solver="highs", number_solutions=number, path=str(tmp_path))
