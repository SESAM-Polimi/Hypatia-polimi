from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.constants import (
    ModelMode,
    TopologyType
)
from hypatia.utility.utility import _calc_production_overall
import pandas as pd
import cvxpy as cp
import numpy as np

"""
Defines lower limit for the annual energy and electric production from renewable energy technologies globally
"""
class RenewableProductionGlobal(Constraint):
    TOPOLOGY_TYPES = [TopologyType.MultiNode]
    
    def _check(self):
        assert self.variables.technology_prod != None, "technology_prod cannot be None"  
        assert self.variables.totalprodbycarrier != None, "totalprodbycarrier cannot be None" 

    def __electicity_production_calc(self):
        
        totalprodelec_annual = {}
        for reg in self.model_data.settings.regions:
            
            for carr,value in self.variables.totalprodbycarrier[reg].items():
                
                if self.model_data.settings.mode == ModelMode.Planning:
                
                    if carr != 'Electricity':
                        continue
        
                    prodelec = []
        
                    for year in range(0, len(self.model_data.settings.years)):
        
                        prodelec_annual_rest = cp.sum(
                            value[(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                            axis=0,
                            keepdims=True
                        )
        
                        prodelec.append(prodelec_annual_rest)
        
                    totalprodelec_annual_regional = cp.vstack(prodelec)
                    
                else:
                   
                    totalprodelec_annual_regional = cp.sum(
                        value[(0) * len(self.model_data.settings.time_steps) : (1) * len(self.model_data.settings.time_steps)],
                        axis=0,
                        keepdims=True
                    )

            totalprodelec_annual[reg] = totalprodelec_annual_regional
            
            globaltotalprodelec_annual = sum(totalprodelec_annual.values())
    
        return globaltotalprodelec_annual
    
    def __energy_production_calc(self):
        
        totalprod_annual = {}
        
        for reg in self.model_data.settings.regions:
    
            totalprod = []

            for year in range(0, len(self.model_data.settings.years)):

                prod_annual_sector = cp.sum(
                    self.model_data.regional_parameters[reg]["demand"][(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                    axis=0,
                    keepdims=True
                )
                totalprod_annual_sector = cp.sum(
                    prod_annual_sector,
                    axis=1,
                    keepdims=True
                )                 

                totalprod.append(totalprod_annual_sector)

            totalprod_annual_regional = cp.vstack(totalprod)

            totalprod_annual[reg] = totalprod_annual_regional
            
            globaltotalprod_annual = sum(totalprod_annual.values())
    
        return globaltotalprod_annual

    def __renewable_electicity_production_calc(self):
        
        renewable_elec_prod = {}
        
        for reg in self.model_data.settings.regions:        
            
            techprodelec_annual_regional = {}
            # transmission_efficiency = self.model_data.regional_parameters[reg]["tech_efficiency"]["Transmission"]["Elec_transmission_distribution"].values
                    
            for key in self.model_data.settings.technologies[reg].keys():
                
                if key != "Storage" and key != "Demand" and key != "Transmission": 

                    for indx, tech in enumerate(self.model_data.settings.technologies[reg][key]):
                        
                        if self.model_data.global_parameters["global_renewable_tech"][tech].values == 1:
                            
                            for carr,value in self.variables.totalprodbycarrier[reg].items():
                                
                                if carr != 'Electricity':
                                    continue
                        
                                if (
                                    carr
                                    in self.model_data.settings.regional_settings[reg]["Carrier_output"]
                                    .loc[
                                        self.model_data.settings.regional_settings[reg]["Carrier_output"]["Technology"]
                                        == tech
                                    ]["Carrier_out"]
                                    .values
                                ):
                                    
                                    if key == "Conversion_plus":

                                        techprodelec_annual_conv = []
                                        convprodelec = cp.multiply(self.variables.technology_prod[reg][key][:, indx],self.model_data.regional_parameters[reg]["carrier_ratio_out"][(tech, carr)].values)
                                        
                                        
                                        for year in range(0, len(self.model_data.settings.years)):
                                            
                                            techprodelec_annual_rest = cp.sum(
                                                convprodelec[(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                                                axis=0,
                                                keepdims=True,
                                            )
                                            techprodelec_annual_conv.append(techprodelec_annual_rest) 
                                            
                                        techprodelec_conv = cp.vstack(techprodelec_annual_conv)
                                        # transmission_efficiency.shape = techprodelec_conv.shape
                                            
                                        techprodelec_annual_regional[tech] = techprodelec_conv

                                    else: 
                                        
                                        techprodelec_annual_other = []
                                                                        
                                        for year in range(0, len(self.model_data.settings.years)):
                                            
                                            othertechprodelec_annual_rest = cp.sum(
                                                self.variables.technology_prod[reg][key][:, indx][(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                                                axis=0,
                                                keepdims=True,
                                            )
                                            techprodelec_annual_other.append(othertechprodelec_annual_rest) 
                                            
                                        techprodelec_annual = cp.vstack(techprodelec_annual_other)
                                            
                                        techprodelec_annual_regional[tech] = techprodelec_annual

            renewable_elec_prod[reg] = techprodelec_annual_regional
        return renewable_elec_prod
    
    def __renewable_production_calc(self):
        
        renewable_prod = {}
        
        for reg in self.model_data.settings.regions:        
            
            techprod_annual_regional = {}
                    
            for key in self.model_data.settings.technologies[reg].keys():
                
                if key != "Storage" and key != "Demand" and key != "Transmission": 

                    for indx, tech in enumerate(self.model_data.settings.technologies[reg][key]):
                        
                        if self.model_data.global_parameters["global_renewable_tech"][tech].values == 1:
                                        
                            techprod_annual = []
                                                            
                            for year in range(0, len(self.model_data.settings.years)):
                                
                                othertechprod_annual_rest = cp.sum(
                                    self.variables.technology_prod[reg][key][:, indx][(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                                    axis=0,
                                    keepdims=True,
                                )
                                techprod_annual.append(othertechprod_annual_rest) 
                                
                            techprod_annual_regional[tech] = cp.vstack(techprod_annual)
                                        
            renewable_prod[reg] = techprod_annual_regional
        return renewable_prod
    
    def rules(self):
        rules = [] 
        
        globaltotalprodelec_annual = self.__electicity_production_calc()
        globaltotalprod_annual = self.__energy_production_calc()
        renewable_elec_prod = self.__renewable_electicity_production_calc()
        renewable_prod = self.__renewable_production_calc()
        
        global_renewable_elec_prod_annual = np.zeros((len(self.model_data.settings.years), 1))
        global_renewable_prod_annual = np.zeros((len(self.model_data.settings.years), 1))
        for reg in self.model_data.settings.regions:
            global_renewable_elec_prod_annual += sum(renewable_elec_prod[reg].values())
            global_renewable_prod_annual += sum(renewable_prod[reg].values())
        
        rules.append(
            global_renewable_elec_prod_annual
            - cp.multiply(self.model_data.global_parameters["global_min_renewable_electric_penetration"].values, globaltotalprodelec_annual)
            >= 0
            )  

        rules.append(
            global_renewable_prod_annual
            - cp.multiply(self.model_data.global_parameters["global_min_renewable_production"].values, globaltotalprod_annual)
            >= 0
            )                   
                    
        return rules
    
    def _required_global_parameters(settings):
        
        return {
            "global_renewable_tech": {
                "sheet_name": "Renewable_tech_selection",
                "value": 0,
                "index": pd.Index(["Renewable technology (1) or not (0)"], name='Parameter'),
                "columns": pd.Index(
                settings.global_settings["Technologies_glob"].loc[
                    (
                        settings.global_settings["Technologies_glob"]["Tech_category"]
                        != "Demand"
                    )
                    & (
                        settings.global_settings["Technologies_glob"]["Tech_category"]
                        != "Storage"
                    )
                    & (
                        settings.global_settings["Technologies_glob"]["Tech_category"]
                        != "Transmission"
                    )]["Technology"],
                )   
            },
            "global_min_renewable_production": {
                "sheet_name": "Global_Min_RES_production",
                "value": 0,
                "index": pd.Index(settings.years, name="Years"),
                "columns": pd.Index(["Min Renewable Production Share (%)"], name='Parameter'),
            },
            "global_min_renewable_electric_penetration": {
                "sheet_name": "Global_Min_RES_elec_penetration",
                "value": 0,
                "index": pd.Index(settings.years, name="Years"),
                "columns": pd.Index(["Min Renewable Electric Penetration (%)"], name='Parameter'),
            },
        }

    
        
           