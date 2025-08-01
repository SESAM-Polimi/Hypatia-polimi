# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:34:38 2025

@author: Tommaso
"""

from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.utility import (
    get_parameters_from_global_or_regional_file
)

import pandas as pd
import numpy as np
import cvxpy as cp


class CHPOperatingRange(Constraint):
    
    # Define class attributes to be used in all methods
    CHP_list = [1,2,3,4,5]  # YOU CAN CHANGE THE NUMBER OF CHP TECHNOLOGY ENTRIES
    CHP_header_list = ["CHP Tech Name", 
                       "Power-branch Tech Name", 
                       "Heat-branch Tech Name",
                       "Max Heat Power Coefficient"] # YOU CAN CHANGE THE HEADINGS, BUT DO NOT CHANGE THE NUMBER OF COLUMNS MUST BE FILLED (4), AND THEIR ORDER
    
    def _check(self):
        assert hasattr(self.variables, 'technology_prod'), "technology_prod must be defined"
        
    
    def rules(self):
        
        CHP_list = self.CHP_list
        CHP_header_list = self.CHP_header_list
        
        rules = []
        
        # Read Input Parameters: choose between global/regional files
        CHP_address = get_parameters_from_global_or_regional_file(
            self.model_data.settings,
            self.model_data,
            "glob_variable_CHP_tech_selection",
            "variable_CHP_tech_selection"
        )
        # if self.model_data.settings.multi_node:
        #     CHP_address = self.model_data.global_parameters["glob_variable_CHP_tech_selection"]
        # else:
        #     for reg in self.model_data.settings.regions:
        #         CHP_address = self.model_data.regional_parameters[reg]["variable_CHP_tech_selection"]
                
        # print(CHP_address)
        # for row in CHP_list:
        #     for col in CHP_header_list[:3]:
        #         print("Row:", row, "-Col:", col, "-Value:", CHP_address.loc[row, col], "-Type:", type(CHP_address.loc[row, col]))

        for reg in self.model_data.settings.regions:
            reg_conv_techs = list(self.model_data.settings.technologies[reg]["Conversion"])
            
            # Create and manage a matrix of indexes reporting the tech index corresponding to the CHP tech name
            CHP_matrix_indx = np.full((len(CHP_list), 3), None, dtype=object)  # Initialize matrix of tech indexes with Nones
            
            for row_indx, row in enumerate(CHP_list):
                for col_indx, col in enumerate(CHP_header_list[:3]):
            
                    for tech_indx, tech in enumerate(reg_conv_techs):
                        
                        if tech == CHP_address.loc[row, col]: 
                            CHP_matrix_indx[row_indx, col_indx] = tech_indx
                            
                if CHP_matrix_indx[row_indx, 0] != None and CHP_matrix_indx[row_indx, 1] != None and CHP_matrix_indx[row_indx, 2] != None:
                    
                    # print(f"Added CHP constraint for row {row}")
                    
                    # tech_prod_Heat <= gamma*tech_prod_CHP
                    rules.append(
                        self.variables.technology_prod[reg]["Conversion"][
                                      :,
                                      CHP_matrix_indx[row_indx, 2]
                                      ]
                        - cp.multiply(
                            self.variables.technology_prod[reg]["Conversion"][
                                          :,
                                          CHP_matrix_indx[row_indx, 0]
                                          ],
                            CHP_address.iloc[row_indx, 3]
                            )
                         <= 0
                    ) 
                    
                    # tech_prod_Heat <= tech_prod_Power
                    rules.append(
                        self.variables.technology_prod[reg]["Conversion"][
                                      :,
                                      CHP_matrix_indx[row_indx, 2]
                                      ]
                        - self.variables.technology_prod[reg]["Conversion"][
                                          :,
                                          CHP_matrix_indx[row_indx, 1]
                                          ]
                         <= 0
                    )
                    
            # print("CHP_matrix_indx:\n", CHP_matrix_indx)

        return rules
    

    def _required_regional_parameters(settings):
        print("\nreading regional CHP parameters\n")
        required_parameters = {}
        for reg in settings.regions:
    
            required_parameters[reg] = {
                "variable_CHP_tech_selection": {
                    "sheet_name": "CHP_variable_techs_selection",
                    "value": 0,
                    "index": pd.Index(CHPOperatingRange.CHP_list, name = "CHP group"),
                    "columns": pd.Index(CHPOperatingRange.CHP_header_list),
                },
            }
        
        if settings.multi_node:
            required_parameters = {}
            
        return required_parameters
    
    def _required_global_parameters(settings):
        print("\nreading global CHP parameters\n")
        return {
            "variable_CHP_tech_selection": {
                "sheet_name": "CHP_variable_techs_selection",
                "value": 0,
                "index": pd.Index(CHPOperatingRange.CHP_list, name = "CHP group"),
                "columns": pd.Index(CHPOperatingRange.CHP_header_list),
            },
        }
    