from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.utility import get_regions_with_storage
import cvxpy as cp

"""
Defines the maximum and minumum alllowed storage state of charge in each
timestep of the year based on the total nominal capacity and the minimum
state of charge factor
"""
class StorageMaxMinChange(Constraint):
    def _check(self):
        assert hasattr(self.variables, 'totalcapacity'), "totalcapacity must be defined"

    def rules(self):
        timeslice_fraction = self.model_data.settings.timeslice_fraction
        if not isinstance(timeslice_fraction, int):
            timeslice_fraction.shape = (len(self.model_data.settings.time_steps), 1)
 
        rules = []
        for reg in get_regions_with_storage(self.model_data.settings):
            for indx, year in enumerate(self.model_data.settings.years):

                annual_storage_capacity = cp.multiply(
                    self.variables.totalcapacity[reg]["Storage"][indx : indx + 1, :],
                    timeslice_fraction) * 8760 # shape = (ts, techs)
                
                rules.append(
                    annual_storage_capacity
                    - self.variables.storage_SOC[reg][
                        indx* len(self.model_data.settings.time_steps)
                        : (indx + 1) * len(self.model_data.settings.time_steps),
                        :] # shape = (ts, techs)
                    >= 0
                )                
                
                rules.append(
                    self.variables.storage_SOC[reg][
                        indx * len(self.model_data.settings.time_steps) 
                        : (indx + 1) * len(self.model_data.settings.time_steps),
                        :] # shape = (ts, techs)
                    - cp.multiply(
                        annual_storage_capacity,
                        self.model_data.regional_parameters[reg]["storage_min_SOC"].values[indx : indx + 1, :]
                    ) # shape = (ts, techs)
                    >= 0
                )
                      
        return rules
                