from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.constants import (
    EnsureFeasibility
)
import cvxpy as cp

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
                        self.variables.totalprodbycarrier[reg][carr].shape,
                    )
                    - cp.reshape(
                        self.variables.totalusebycarrier[reg][carr],
                        self.variables.totalprodbycarrier[reg][carr].shape,
                    )
                    - self.variables.totalexportbycarrier[reg][carr]
                    - self.variables.totaldemandbycarrier[reg][carr]
                    == 0
                )
        return rules
