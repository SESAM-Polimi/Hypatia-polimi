from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.constants import (
    ModelMode,
    TopologyType
)
from hypatia.utility.utility import (
    _calc_production_overall, 
    _calc_carr_production_overall,
    annual_activity,
    stack
)
import pandas as pd
import numpy as np
import cvxpy as cp


"""
Defines the upper and lower limit for the aggregated annual production
of the technologies over all the regions
"""
class AnnualProductionGlobal(Constraint):
    TOPOLOGY_TYPES = [TopologyType.MultiNode]

    def _check(self):
        assert hasattr(self.variables, "production_annual"), "production_annual must be defined"

    def rules(self):
        
        rules = []
        
        totalprodbycarrier_annual = {}
        for reg in self.model_data.settings.regions:
            
            carr_production_annual = {}
            
            for carr,value in self.variables.totalprodbycarrier[reg].items():
                
                prodbycarrier_annual = []

                for year in range(0, len(self.model_data.settings.years)):

                    totalprodbycarrier_annual_rest = cp.sum(
                        value[(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                        axis=0,
                        keepdims=True
                    )
                    
                    prodbycarrier_annual.append(totalprodbycarrier_annual_rest)

                carr_production_annual[carr] = cp.vstack(prodbycarrier_annual)

            totalprodbycarrier_annual[reg] = carr_production_annual
        
        production_carr_overall = _calc_carr_production_overall(
            self.model_data.settings.global_settings["Carriers_glob"],
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.regional_settings,
            totalprodbycarrier_annual,
        )            
        
        techprodbycarrier_annual = {}
        
        for reg in self.model_data.settings.regions:

            techprod = {}
   
            for key, value in self.model_data.settings.technologies[reg].items():
                
                if key != "Storage" and key != "Demand":

                    if key == "Conversion_plus":
                        
                        for indx, tech in enumerate(self.model_data.settings.technologies[reg][key]):
                        
                            convprobycarr = {}
                            
                            for carr in self.model_data.settings.regional_settings[reg]["Carrier_output"].loc[self.model_data.settings.regional_settings[reg]["Carrier_output"]["Technology"] == tech]["Carrier_out"]:
                                
                                techprodbycarrier_annual_conv = []
                                convprobycarrier = cp.multiply(self.variables.technology_prod[reg][key][:, indx],self.model_data.regional_parameters[reg]["carrier_ratio_out"][(tech, carr)].values)
                                
                                for year in range(0, len(self.model_data.settings.years)):
                                    
                                    techprodbycarrier_annual_rest = cp.sum(
                                        convprobycarrier[(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                                        axis=0,
                                        keepdims=True,
                                    )
                                    techprodbycarrier_annual_conv.append(techprodbycarrier_annual_rest) 
                                convprobycarr[carr] = cp.vstack(techprodbycarrier_annual_conv)
                                    
                            techprod[tech] = convprobycarr
 
                    else:
                        
                        for indx, tech in enumerate(self.model_data.settings.technologies[reg][key]):
                            # print(tech)
                            probycarr = {}
                            
                            for carr in self.model_data.settings.regional_settings[reg]["Carrier_output"].loc[self.model_data.settings.regional_settings[reg]["Carrier_output"]["Technology"] == tech]["Carrier_out"]:
                                
                                techprodbycarr_annual = []
                                probycarrier = self.variables.technology_prod[reg][key][:, indx]
                                
                                for year in range(0, len(self.model_data.settings.years)):
                                    
                                    techprodbycarrier_annual_rest = cp.sum(
                                        probycarrier[(year) * len(self.model_data.settings.time_steps) : (year+1) * len(self.model_data.settings.time_steps)],
                                        axis=0,
                                        keepdims=True,
                                    )
                                    techprodbycarr_annual.append(techprodbycarrier_annual_rest) 
                                probycarr[carr] = cp.vstack(techprodbycarr_annual)
     
                            techprod[tech] = probycarr

            techprodbycarrier_annual[reg] = techprod

        techproductionbycarr_overall = {}
        for tech in list(self.model_data.settings.global_settings["Technologies_glob"].loc[
            (self.model_data.settings.global_settings["Technologies_glob"]["Tech_category"] != "Demand")
            & (self.model_data.settings.global_settings["Technologies_glob"]["Tech_category"] != "Storage")]["Technology"]
        ):
            carrproduction_overall = {}
            for carr in list(self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]):
                carrproduction_overall[carr] = np.zeros((len(self.model_data.settings.years), 1))
                for reg in self.model_data.settings.regions:
                    for key in self.model_data.settings.regional_settings[reg]["Carrier_output"].loc[self.model_data.settings.regional_settings[reg]["Carrier_output"]["Technology"]== tech]["Carrier_out"].values:  
                        if carr in key:
                            carrproduction_overall[carr] += techprodbycarrier_annual[reg][tech][carr]
            techproductionbycarr_overall[tech] = carrproduction_overall 
        
        for tech in list(self.model_data.settings.global_settings["Technologies_glob"].loc[
            (self.model_data.settings.global_settings["Technologies_glob"]["Tech_category"] != "Demand")
            & (self.model_data.settings.global_settings["Technologies_glob"]["Tech_category"] != "Storage")]["Technology"]
        ):
            for carr in list(self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]):
                for reg in self.model_data.settings.regions:
                    for key in self.model_data.settings.regional_settings[reg]["Carrier_output"].loc[self.model_data.settings.regional_settings[reg]["Carrier_output"]["Technology"]== tech]["Carrier_out"].values:  
                        if carr in key:
        
                            rules.append(
                                        techproductionbycarr_overall[tech][carr]
                                        - self.model_data.global_parameters["global_max_production"].loc[:,(tech, carr, slice(None))]
                                        <= 0
                                        )
                            rules.append(
                                        techproductionbycarr_overall[tech][carr]
                                        - self.model_data.global_parameters["global_min_production"].loc[:,(tech, carr, slice(None))]
                                        >= 0
                                        )
                            rules.append(
                                        techproductionbycarr_overall[tech][carr]
                                        - cp.multiply(self.model_data.global_parameters["global_max_production_share"].loc[:,(tech, carr, slice(None))],production_carr_overall[carr])
                                        <= 0
                                        )
                            rules.append(
                                        techproductionbycarr_overall[tech][carr]
                                        - cp.multiply(self.model_data.global_parameters["global_min_production_share"].loc[:,(tech, carr, slice(None))],production_carr_overall[carr])
                                        >= 0
                                        )    
       
        return rules

    def _required_global_parameters(settings):
        
        take_carrierout_ = []
        take_technologyout_ = []
                                
        for reg in settings.regions: 
            for key in settings.technologies[reg].keys():
                for tech in settings.technologies[reg][key]:
                    if tech not in take_technologyout_:
            
                        take_carrierout = [(
                            settings.regional_settings[reg]["Carrier_output"]
                            .loc[settings.regional_settings[reg]["Carrier_output"]["Technology"] == tech][
                                "Carrier_out"
                            ]
                            .values
                        )]
            
                        for index, value in enumerate(take_carrierout):
                            for carr in take_carrierout[index]:
                                take_carrierout_.append(
                                    carr
                                )
        
                        take_technologyout = [(
                            settings.regional_settings[reg]["Carrier_output"]
                            .loc[settings.regional_settings[reg]["Carrier_output"]["Technology"] == tech][
                                "Technology"
                            ]
                            .values
                        )]
        
            
                        for index, value in enumerate(take_technologyout):
                            for tech in take_technologyout[index]: 
                                take_technologyout_.append(tech) 
                
        return {
            "global_max_production": {
                    "sheet_name": "Max_prod_global",
                    "value": 1e30,
                    "index": pd.Index(settings.years, name="Years"),
                    "columns": pd.MultiIndex.from_arrays(
                        [take_technologyout_, take_carrierout_],
                        names=["Technology", "Carrier"],
                    ),
                },     
            "global_max_production_share": {
                    "sheet_name": "Max_prod_global_share",
                    "value": 1,
                    "index": pd.Index(settings.years, name="Years"),
                    "columns": pd.MultiIndex.from_arrays(
                        [take_technologyout_, take_carrierout_],
                        names=["Technology", "Carrier"],
                    ),
                }, 
            "global_min_production": {
                    "sheet_name": "Min_prod_global",
                    "value": 0,
                    "index": pd.Index(settings.years, name="Years"),
                    "columns": pd.MultiIndex.from_arrays(
                        [take_technologyout_, take_carrierout_],
                        names=["Technology", "Carrier"],
                    ),
                }, 
            "global_min_production_share": {
                    "sheet_name": "Min_prod_global_share",
                    "value": 0,
                    "index": pd.Index(settings.years, name="Years"),
                    "columns": pd.MultiIndex.from_arrays(
                        [take_technologyout_, take_carrierout_],
                        names=["Technology", "Carrier"],
                    ),
                }, 
        }
        
    
    
