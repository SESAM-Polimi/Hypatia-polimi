"""
Write the "feature model" used by the build fingerprint tests.

The shipped examples never build some parts of Hypatia: modular (MILP)
capacity, ramp limits, minimum power output and capacity groups. This model is
Neverland (planning, two regions, one line, 24 time steps, 10 years) with those
features switched on. It is meant to be *built*, not solved: the values below
are chosen to make every constraint produce rows, not to be realistic.

Run from the repository root:

    python -m tests.data.make_feature_model

Files are written as plain values (no formulas), so re-reading them does not
depend on cached formula results. Changing anything here changes the built
model: regenerate the fingerprint reference too (python -m tests.fingerprint
update Feature_model) and explain why in the PR.
"""
import os
import shutil

import pandas as pd

from hypatia import Model

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(REPO, "hypatia", "examples", "Neverland")
TARGET = os.path.join(REPO, "tests", "data", "feature_model")

OPTIONAL_CONSTRAINTS = ["production_ramp", "min_tech_power_output"]
YEARS = [f"Y{i}" for i in range(10)]

# Row of the value in the technology sheets (tech category, technology,
# parameter name, value), and the values to write there.
VALUE_ROW = 3
TECH_VALUES = {
    "Modular_cap_unit": {"Coal_PP": 0.5, "Gas_PP": 0.25},           # MILP
    "Max_rate_ramp-up": {"Coal_PP": 0.3, "Gas_PP": 0.6},
    "Max_rate_ramp-down": {"Coal_PP": 0.4},
    "Min_power_output": {"Coal_PP": 0.3, "Gas_PP": 0.2},
}
GROUP_SHEETS = {
    "Capacity_group_membership": pd.DataFrame(
        [("RES", "Hydro_PP"), ("RES", "Solar_PP"), ("FOSSIL", "Coal_PP"), ("FOSSIL", "Gas_PP")],
        columns=["Group", "Technology"]),
    # A limit table must give every model year of each group it names.
    "Max_totalcap_group": pd.DataFrame(
        [("RES", y, 40.0 + 2 * i) for i, y in enumerate(YEARS)],
        columns=["Group", "Year", "Max_totalcap"]),
    "Max_newcap_group": pd.DataFrame(
        [("FOSSIL", y, 3.0 + i) for i, y in enumerate(YEARS)],
        columns=["Group", "Year", "Max_newcap"]),
}


def raw_sheets(path):
    """Every sheet as a grid of cell values, headers included."""
    return pd.read_excel(path, sheet_name=None, header=None)


def set_tech_values(grid, values):
    technologies = list(grid.iloc[1])
    for tech, value in values.items():
        grid.iloc[VALUE_ROW, technologies.index(tech)] = value
    return grid


def write_raw(sheets, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, grid in sheets.items():
            grid.to_excel(writer, sheet_name=name, header=False, index=False)


def main(target=TARGET, features=True):
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(os.path.join(SOURCE, "sets"), os.path.join(target, "sets"),
                    ignore=shutil.ignore_patterns("__init__.py", "__pycache__"))

    # Templates of the sheets the optional constraints need.
    templates = os.path.join(target, "_templates")
    Model(path=os.path.join(target, "sets"), mode="Planning", optimization="Single",
          ensure_feasibility="No", optional_constraints=OPTIONAL_CONSTRAINTS
          ).create_data_excels(templates)

    os.makedirs(os.path.join(target, "parameters"))
    for name in sorted(os.listdir(os.path.join(SOURCE, "parameters"))):
        sheets = raw_sheets(os.path.join(SOURCE, "parameters", name))
        if features and name == "parameters_global.xlsx":
            template = raw_sheets(os.path.join(templates, name))
            for sheet, values in TECH_VALUES.items():
                grid = sheets[sheet] if sheet in sheets else template[sheet]
                sheets[sheet] = set_tech_values(grid.copy(), values)
            for sheet, table in GROUP_SHEETS.items():
                sheets[sheet] = pd.concat([pd.DataFrame([table.columns]), pd.DataFrame(table.values)],
                                          ignore_index=True)
        write_raw(sheets, os.path.join(target, "parameters", name))
    shutil.rmtree(templates)


def load(target=TARGET, optional_constraints=OPTIONAL_CONSTRAINTS):
    model = Model(path=os.path.join(target, "sets"), mode="Planning", optimization="Single",
                  ensure_feasibility="No", optional_constraints=optional_constraints)
    model.read_input_data(os.path.join(target, "parameters"))
    return model


if __name__ == "__main__":
    main()
    print(f"written {TARGET}")
