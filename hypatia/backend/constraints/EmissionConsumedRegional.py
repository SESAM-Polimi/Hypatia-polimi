from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.utility import get_emission_types
import pandas as pd
import numpy as np
import cvxpy as cp

class EmissionConsumedRegional(Constraint):
    def _check(self):
        assert hasattr(self.variables, "captured_emission_by_region"), "captured_emission_by_region must be defined"
    
    def rules(self):
        rules = []
        
        for emission_type in get_emission_types(self.model_data.settings.global_settings):

            for reg in self.model_data.settings.regions:
                
                captured_emission = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps), 1)
                )
                for key, value in self.variables.captured_emission_by_region[reg][emission_type].items():
                    captured_emission += cp.sum(value, axis=1)
                Stored_cap = self.model_data.regional_parameters[reg]["emission_consumed"][
                    "{} Stored Cap".format(emission_type)
                ].values
                Stored_cap.shape = captured_emission.shape
                rules.append(Stored_cap - captured_emission >= 0)
                
         
        return rules                
    
    @staticmethod
    def _required_regional_parameters(settings):
        required_parameters = {}
        for reg in settings.regions:
            required_parameters[reg] = {
                "emission_consumed": {
                    "sheet_name": "Emission_Storage_annual",
                    "value": 1e10,
                    "index": pd.Index(settings.years, name="Years"),
                    "columns": pd.Index(
                        [emission_type + " Stored Cap" for emission_type in get_emission_types(
                            settings.global_settings
                        )]
                    ),
                },
            }

        return required_parameters