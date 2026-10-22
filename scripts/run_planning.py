"""
Run a Hypatia planning model
============================

Reads the input files of a planning model, solves it, saves the results as
CSV files and draws the main plots.

How to use it
-------------
1. Change the values in the SETTINGS section below. Nothing else needs
   changing to run a model.
2. Run the whole script (Spyder: F5), or one cell at a time (Spyder:
   Ctrl+Enter); cells start with "# %%".

The first time, if MODEL_FOLDER does not exist, the example "Planning_1Region"
is copied into it, so the script runs as it is. To run your own model, point
MODEL_FOLDER to a folder with the sub-folders "sets" and "parameters".

All files are written inside MODEL_FOLDER: "results", "plots" and, on request,
"parameter_templates".
"""
from pathlib import Path

from hypatia import Model, Plotter, download_example

# %% SETTINGS ----------------------------------------------------------------

# Folder of the model, holding "sets" and "parameters". A relative path is
# taken from the folder of this script.
MODEL_FOLDER = "Planning_1Region"

# Example copied into MODEL_FOLDER when that folder does not exist yet.
# Set it to None for your own model.
EXAMPLE = "Planning_1Region"

# What to minimise:
#   "Single" - the total discounted cost
#   "MinEm"  - the total emissions (then the cost, among the least-emission solutions)
#   "Multi"  - a series of solutions from least cost to least emissions
OPTIMIZATION = "Single"
NUMBER_SOLUTIONS = 3           # used only with "Multi"; at least 2

# "Yes" lets the model leave demand unmet, at a very high cost, so that it
# always has a solution and the results show where supply is missing.
ENSURE_FEASIBILITY = "No"

# Optional constraints, e.g. ["chp_operating_range"]. They must be set
# before creating the templates, and the parameter files must contain their
# sheets. Names: chp_operating_range, production_ramp, min_tech_power_output,
# electrolysis_consumption, boolean_for_storage.
OPTIONAL_CONSTRAINTS = []

# "highs" is open source and installed with Hypatia; "gurobi" needs a licence.
SOLVER = "highs"

# True writes empty parameter files (with default values) to
# MODEL_FOLDER/parameter_templates, as the starting point of a new model.
CREATE_TEMPLATES = False

# Plots need MODEL_FOLDER/config.xlsx, which sets names, colours and groups of
# technologies and carriers. If it is missing, the script creates it for you
# to fill in. TECH_GROUP and FUEL_GROUP are groups of that file.
MAKE_PLOTS = True
TECH_GROUP = "Power Generation"
FUEL_GROUP = "Electricity"

# %% 1. Model files -----------------------------------------------------------

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # some editors run cells without __file__
    HERE = Path.cwd()
model_folder = (HERE / MODEL_FOLDER).resolve()

if not model_folder.exists():
    if EXAMPLE is None:
        raise FileNotFoundError(f"The model folder {model_folder} does not exist.")
    download_example(EXAMPLE, str(model_folder))
    print(f"Copied the example {EXAMPLE} to {model_folder}")

results_folder = model_folder / "results"
plots_folder = model_folder / "plots"

# %% 2. Create the model from the sets -----------------------------------------

model = Model(
    path=str(model_folder / "sets"),
    mode="Planning",
    optimization=OPTIMIZATION,
    ensure_feasibility=ENSURE_FEASIBILITY,
    optional_constraints=OPTIONAL_CONSTRAINTS,
    name=model_folder.name,
)
print(model)

# %% 3. Parameter templates (optional) -----------------------------------------

if CREATE_TEMPLATES:
    model.create_data_excels(str(model_folder / "parameter_templates"), force_rewrite=True)
    print(f"Templates written to {model_folder / 'parameter_templates'}")

# %% 4. Read the parameters ------------------------------------------------------

model.read_input_data(str(model_folder / "parameters"))

# %% 5. Solve --------------------------------------------------------------------

# force_rewrite=True lets this cell run again after a change.
if OPTIMIZATION == "Single":
    model.run(solver=SOLVER, verbosity=True, force_rewrite=True)
elif OPTIMIZATION == "MinEm":
    model.run_MinEm(solver=SOLVER, verbosity=True, force_rewrite=True)
else:
    # Each solution is saved in results/trade_off/1, 2, ... (1 = least cost).
    model.run_MO(
        solver=SOLVER,
        number_solutions=NUMBER_SOLUTIONS,
        path=str(results_folder / "trade_off"),
        verbosity=True,
    )

if model.results is None:
    raise RuntimeError(
        "The solver found no optimal solution. Check the data, or set "
        "ENSURE_FEASIBILITY = 'Yes' to see where demand cannot be met."
    )

# %% 6. Save the results -----------------------------------------------------------

# One CSV file per quantity, in long format (easy to filter in Excel or pandas).
if OPTIMIZATION != "Multi":  # run_MO has already saved its solutions
    model.to_csv(str(results_folder / OPTIMIZATION), postprocessing_module="aggregated",
                 force_rewrite=True)
    print(f"Results saved to {results_folder / OPTIMIZATION}")

# %% 7. Plots ------------------------------------------------------------------------

config = model_folder / "config.xlsx"
years = list(model.settings.years)
hourly = len(model.settings.time_steps) == 8760

if not MAKE_PLOTS:
    print("Plots skipped (MAKE_PLOTS = False).")
elif OPTIMIZATION == "Multi":
    print("Plots skipped: with 'Multi' the model holds only the last solution. "
          "Plot each folder of results/trade_off instead.")
elif not config.exists():
    model.create_config_file(str(config))
    print(f"Created {config}. Fill in the groups and colours of technologies and "
          f"carriers, then run this cell again.")
else:
    plots_folder.mkdir(exist_ok=True)
    plots = Plotter(model, config=str(config), hourly_resolution=hourly)

    plots.plot_total_capacity(
        path=str(plots_folder / "total_capacity.html"),
        tech_group=TECH_GROUP,
        kind="bar",            # "bar" or "area"
        decom_cap=True,        # also show the decommissioned capacity
    )
    plots.plot_new_capacity(
        path=str(plots_folder / "new_capacity.html"),
        tech_group=TECH_GROUP,
        kind="bar",
        cummulative=False,     # True: capacity added since the first year
    )
    plots.plot_prod_by_tech(
        path=str(plots_folder / "production.html"),
        tech_group=TECH_GROUP,
        kind="bar",
    )
    plots.plot_use_by_technology(
        path=str(plots_folder / "use.html"),
        fuel_group=FUEL_GROUP,
        kind="bar",
    )
    # Shares of production and consumption of the fuel group, first and last year.
    for year in (years[0], years[-1]):
        plots.plot_fuel_prod_cons(
            path=str(plots_folder / f"production_consumption_{year}.html"),
            years=[year],
            fuel_group=FUEL_GROUP,
            trade=model.settings.multi_node,
        )
    plots.plot_emissions(
        path=str(plots_folder / "emissions.html"),
        tech_group=TECH_GROUP,
        emission_type=["CO2 emissions"],   # emission_name in config.xlsx
        kind="bar",
    )
    plots.plot_regional_costs(
        path=str(plots_folder / "costs.html"),
        stacked_by="techs",    # "techs" or cost "items"
    )
    if hourly:
        plots.plot_hourly_prod_by_tech(
            path=str(plots_folder / "hourly_production_first_year.html"),
            tech_group=TECH_GROUP,
            fuel_group=FUEL_GROUP,
            year=[years[0]],
            start="01-01 00:00:00",   # "MM-DD hh:mm:ss"
            end="01-07 23:00:00",
            kind="area",
        )
    print(f"Plots saved to {plots_folder}")
