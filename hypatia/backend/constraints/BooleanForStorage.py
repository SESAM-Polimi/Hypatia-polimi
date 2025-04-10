from hypatia.backend.constraints.Constraint import Constraint

import cvxpy as cp
"""
Defines the upper limit by each timestep for both production and use of 
Storage techs. When it's activated for production, it's deactivated for use
and the other way around.
"""

class BooleanForStorage(Constraint):
    def _check(self):
        assert self.variables.technology_prod != None, "technology_prod cannot be None"
        assert self.variables.technology_use != None, "technology_use cannot be None"

    def rules(self):
        
        time_steps = len(self.model_data.settings.time_steps)
        M = 100000000000000000000 # 1e20 Extremely high number (to not limit prduction or use)

        rules = []
        
        for reg in self.model_data.settings.regions:
            
            reg_storage_techs = list(self.model_data.settings.technologies[reg]["Storage"])
            # print(reg_storage_techs)
                
            for key in self.variables.technology_prod[reg].keys():
                
                for tech_indx, tech in enumerate(reg_storage_techs):
           
                    carr_out = self.model_data.settings.regional_settings[reg]["Carrier_output"].loc[
                        self.model_data.settings.regional_settings[reg]["Carrier_output"]["Technology"] == tech
                        ]["Carrier_out"].values[0]
                    
                    carr_in = self.model_data.settings.regional_settings[reg]["Carrier_input"].loc[
                        self.model_data.settings.regional_settings[reg]["Carrier_input"]["Technology"] == tech
                        ]["Carrier_in"].values[0]
                    
                    if key == "Storage" and carr_in == carr_out:
                        
                        # print(f"{tech} output: {carr_out}")
                        # print(f"{tech} input: {carr_in}")
                        
                        for year_indx, year in enumerate(self.model_data.settings.years):
                            
                            # Boolean for production
                            rules.append(
                                self.variables.technology_prod[reg]["Storage"][
                                    (year_indx *time_steps) : ((year_indx +1) *time_steps -1),
                                    tech_indx : tech_indx +1
                                    ]
                                -
                                cp.multiply(
                                    self.variables.boolean_for_storage[reg]["Storage"][
                                        (year_indx *time_steps) : ((year_indx +1) *time_steps -1),
                                        tech_indx : tech_indx +1
                                        ],
                                    M
                                    )
                                <= 0
                                )
                            
                            # Boolean for use
                            rules.append(
                                self.variables.technology_use[reg]["Storage"][
                                    (year_indx *time_steps) : ((year_indx +1) *time_steps -1),
                                    tech_indx : tech_indx +1
                                    ]
                                -
                                cp.multiply(
                                    (1 - self.variables.boolean_for_storage[reg]["Storage"][
                                        (year_indx *time_steps) : ((year_indx +1) *time_steps -1),
                                        tech_indx : tech_indx +1
                                        ]
                                        ),
                                    M
                                    )
                                <= 0
                                )
                            
        return rules









