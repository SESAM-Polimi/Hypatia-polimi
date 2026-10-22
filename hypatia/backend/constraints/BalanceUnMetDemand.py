from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.constants import (
    EnsureFeasibility
)
import cvxpy as cp
import pandas as pd

# Cost of one unit of unmet demand when the sheet Unmet_demand_penalty is not
# in the parameter files (files written before October 2026 do not have it).
DEFAULT_UNMET_DEMAND_PENALTY = 1e10


def demand_carriers(settings, reg):
    """Carriers consumed by the demand technologies of a region: the ones that
    get an unmet-demand variable."""
    inputs = settings.regional_settings[reg]["Carrier_input"]
    demand_techs = settings.technologies[reg].get("Demand", [])
    carriers = inputs.loc[inputs["Technology"].isin(demand_techs), "Carrier_in"]
    return list(dict.fromkeys(carriers))  # unique, in order

"""
Ensures the energy balance of each carrier within each region allowing unmet demand
"""
class BalanceUnMetDemand(Constraint):
    ENSURE_FEASIBILITY = [EnsureFeasibility.Yes]
    
    def _check(self):
        assert hasattr(self.variables, 'totalprodbycarrier'), "totalprodbycarrier must be defined"
        assert hasattr(self.variables, 'totalimportbycarrier'), "totalimportbycarrier must be defined"
        assert hasattr(self.variables, 'totalusebycarrier'), "totalusebycarrier must be defined"
        assert hasattr(self.variables, 'totalexportbycarrier'), "totalexportbycarrier must be defined"
        assert hasattr(self.variables, 'totaldemandbycarrier'), "totaldemandbycarrier must be defined"
        assert hasattr(self.variables, 'unmetdemandbycarrier'), "unmetdemandbycarrier must be defined"

    def rules(self):
        rules = []
        for reg in self.model_data.settings.regions:
            for carr in self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]:
                rules.append(
                    self.variables.totalprodbycarrier[reg][carr]
                    + self.variables.totalimportbycarrier[reg][carr]
                    + cp.reshape(
                        self.variables.unmetdemandbycarrier[reg][carr],
                        self.variables.totalprodbycarrier[reg][carr].shape, order="F"
                    )
                    - cp.reshape(
                        self.variables.totalusebycarrier[reg][carr],
                        self.variables.totalprodbycarrier[reg][carr].shape, order="F"
                    )
                    - self.variables.totalexportbycarrier[reg][carr]
                    - self.variables.totaldemandbycarrier[reg][carr]
                    == 0
                )
        return rules

    def _required_regional_parameters(settings):
        # "optional": files without this sheet still read, with the default.
        return {
            reg: {
                "unmet_demand_penalty": {
                    "sheet_name": "Unmet_demand_penalty",
                    "value": DEFAULT_UNMET_DEMAND_PENALTY,
                    "index": pd.Index(["Penalty per unit of unmet demand"], name="Parameter"),
                    "columns": pd.Index(demand_carriers(settings, reg), name="Carrier"),
                    "optional": True,
                }
            }
            for reg in settings.regions
        }
