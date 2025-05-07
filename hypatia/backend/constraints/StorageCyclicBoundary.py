from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.utility import (
    get_regions_with_storage,
    get_parameters_from_global_or_regional_file
)
import pandas as pd
import cvxpy as cp

class StorageCyclicBoundary(Constraint):
    def _check(self):
        assert hasattr(self.variables, 'totalcapacity'), "totalcapacity must be defined"
        assert self.variables.technology_use != None, "technology_use must not be None"
        assert self.variables.technology_prod != None, "technology_prod must not be None"

    def rules(self):
        timeslice_fraction = self.model_data.settings.timeslice_fraction
        if not isinstance(timeslice_fraction, int):
            timeslice_fraction.shape = (len(self.model_data.settings.time_steps), 1)

        rules = []
        for reg in get_regions_with_storage(self.model_data.settings):
            
            # Read Input Parameters: choose between global/regional files
            ts_per_cycle_df = get_parameters_from_global_or_regional_file(
                self.model_data.settings,
                self.model_data,
                "golabl_storage_cyclic_boundary",
                "storage_cyclic_boundary"
            )

            ts_per_cycle = ts_per_cycle_df.iloc[0,0]
            
            # Get the number of cycles included in one-year timesteps
            float_cycles_per_year = len(self.model_data.settings.time_steps) / ts_per_cycle
            # Check if the number of timesteps per cycle is coherent
            if float_cycles_per_year.is_integer():
                cycles_per_year = int(float_cycles_per_year)
            else:
                raise ValueError("cycles_per_year is not an integer. Check the storage cycle duration in the input file.")
            
            for indx, year in enumerate(self.model_data.settings.years):
                
                annual_storage_capacity = cp.multiply(
                    self.variables.totalcapacity[reg]["Storage"][indx : indx + 1, :],
                    timeslice_fraction) * 8760 # shape = (1, storage_techs)
                
                for cycle in range(0, cycles_per_year):
                    minimum_annual_storage_capacity = cp.multiply(
                        annual_storage_capacity,
                        self.model_data.regional_parameters[reg]["storage_min_SOC"].values[indx : indx + 1, :]
                    ) # shape = (288, storage_techs)

                    rules.append(
                        self.variables.storage_SOC[reg][
                            (indx*len(self.model_data.settings.time_steps)) + (cycle +1)*ts_per_cycle -1,
                            :] # shape = (1, storage_techs)
                        - minimum_annual_storage_capacity[
                            (cycle +1)*ts_per_cycle -1, 
                            :] # shape = (1, storage_techs)
                        == 0
                        )
        
        return rules 

    
    def _required_regional_parameters(settings):
        # print("\nreading regional Cyclic parameters\n")
        required_parameters = {}
        for reg in get_regions_with_storage(settings):
    
            required_parameters[reg] = {
                "storage_cyclic_boundary": {
                    "sheet_name": "Storage_cyclic_boundary",
                    "value": len(settings.time_steps),
                    "index": pd.Index(["Storage Cycle Duration"]),
                    "columns": pd.Index(["Timesteps"]),
                },
            }
        
        if settings.multi_node:
            required_parameters = {}
            
        return required_parameters
    
    def _required_global_parameters(settings):
        # print("\nreading global Cyclic parameters\n")
        required_global_parameters = {}
        if "Storage" in settings.technologies_glob.keys():
            required_global_parameters = {
                "golabl_storage_cyclic_boundary": {
                    "sheet_name": "Storage_cyclic_boundary",
                    "value": len(settings.time_steps),
                    "index": pd.Index(["Storage Cycle Duration"]),
                    "columns": pd.Index(["Timesteps"]),
                },
            }
            
        return required_global_parameters
