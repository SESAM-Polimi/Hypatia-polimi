"""Operation tutorial: hourly dispatch of two regions for one year, then the
same system with a higher demand and unmet demand allowed.

Shown in doc/source/tutorials/operation.rst and run by tests/test_docs.py.
"""
import os

import pandas as pd

from hypatia import load_example

# [run]
model = load_example("Operation_2Regions")
print(model)
model.run(solver="highs", verbosity=False)
assert model.results is not None, "no optimal solution"
os.makedirs("operation", exist_ok=True)
model.to_csv("operation/reference", postprocessing_module="aggregated", force_rewrite=True)
# [/run]

# [annual]
production = pd.read_csv("operation/reference/tech_production_annual.csv")
plants = production[production["Carrier"] == "Electricity"]
gwh = plants.pivot_table(index="Technology", columns="Region", values="Value", fill_value=0) / 1e6
print(gwh.round())  # kWh -> GWh
# [/annual]

# [week]
hourly = pd.read_csv("operation/reference/tech_production.csv")
reg1 = hourly[(hourly["Region"] == "reg1") & (hourly["Carrier"] == "Electricity")]
dispatch = reg1.pivot_table(index="Timesteps", columns="Technology", values="Value") / 1e3
july_week = dispatch.loc[4345:4512]  # hours of 1-7 July
print(july_week.describe().loc[["mean", "min", "max"]].round())  # kWh per hour -> MW
# [/week]

# [lines]
flows = pd.read_csv("operation/reference/line_import_annual.csv")
flows["GWh"] = flows["Value"] / 1e6
print(flows[flows["GWh"] > 0][["To reg", "From reg", "Carrier", "GWh"]].round())
# [/lines]

# [unmet]
stressed = load_example("Operation_2Regions_UnMetDemand")
stressed.run(solver="highs", verbosity=False)
stressed.to_csv("operation/high_demand", postprocessing_module="aggregated", force_rewrite=True)

unmet = pd.read_csv("operation/high_demand/unmet_demand.csv")
short = unmet[unmet["Value"] > 1e-6]
summary = short.groupby("Region")["Value"].agg(hours="count", GWh="sum", peak_MW="max")
summary["GWh"] /= 1e6
summary["peak_MW"] /= 1e3
print(summary.round())
# [/unmet]
