"""Planning tutorial: build Neverland from its files, solve it, read the results,
then compute the trade-off between cost and emissions.

Shown in doc/source/tutorials/planning.rst and run by tests/test_docs.py.
"""
import os

import pandas as pd

from hypatia import Model, download_example

# [copy]
if not os.path.exists("neverland"):
    download_example("Neverland", "neverland")
# [/copy]

# [model]
model = Model(
    path="neverland/sets",
    mode="Planning",
    optimization="Single",
    ensure_feasibility="No",
    name="Neverland",
)
model.create_data_excels("neverland/templates", force_rewrite=True)
model.read_input_data("neverland/parameters")
# [/model]

# [run]
model.run(solver="highs", verbosity=False)
assert model.results is not None, "no optimal solution"
os.makedirs("neverland/results", exist_ok=True)  # to_csv creates only the last folder
model.to_csv("neverland/results/least_cost", postprocessing_module="aggregated",
             force_rewrite=True)
# [/run]

# [capacity]
capacity = pd.read_csv("neverland/results/least_cost/total_capacity.csv")
plants = capacity[capacity["Technology"].isin(["Coal_PP", "Gas_PP", "Hydro_PP", "Solar_PP"])]
print(plants.pivot_table(index="Datetime", columns=["Region", "Technology"], values="Value").round(1))
# [/capacity]

# [costs]
costs = pd.read_csv("neverland/results/least_cost/tech_cost.csv")
print(costs.groupby("Cost")["Value"].sum().sort_values(ascending=False))
# [/costs]

# [trade_off]
trade_off = Model(
    path="neverland/sets",
    mode="Planning",
    optimization="Multi",
    ensure_feasibility="No",
    name="Neverland trade-off",
)
trade_off.read_input_data("neverland/parameters")
trade_off.run_MO(solver="highs", number_solutions=3, path="neverland/results/trade_off",
                 verbosity=False)

# A run that finds no optimal solution writes no folder, so read the ones that exist.
folders = [f"neverland/results/trade_off/{i}" for i in range(1, 4)]
points = pd.concat(
    [pd.read_csv(f"{folder}/pareto_point.csv", index_col=0) for folder in folders
     if os.path.exists(f"{folder}/pareto_point.csv")],
    ignore_index=True,
)
print(points)
# [/trade_off]

# [mix]
mix = {}
for i, folder in enumerate(folders, start=1):
    if os.path.exists(f"{folder}/total_capacity.csv"):
        capacity = pd.read_csv(f"{folder}/total_capacity.csv")
        last_year = capacity[capacity["Datetime"] == capacity["Datetime"].max()]
        plants = last_year[last_year["Technology"].isin(["Coal_PP", "Gas_PP", "Hydro_PP", "Solar_PP"])]
        mix[f"solution {i}"] = plants.groupby("Technology")["Value"].sum()
print(pd.DataFrame(mix).round(1))
# [/mix]
