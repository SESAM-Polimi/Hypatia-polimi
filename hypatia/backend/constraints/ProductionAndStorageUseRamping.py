# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:18:11 2024

@author: Tommaso
"""

from hypatia.backend.constraints.Constraint import Constraint

from hypatia.utility.utility import (
    theoretical_available_prod,
    create_technology_columns
)

import cvxpy as cp
import pandas as pd


"""
Defines the ramp constraint suited ONLY for 1 hour time resolution
"""

class ProductionAndStorageUseRamping(Constraint):
    def _check(self):
        assert hasattr(self.variables, 'totalcapacity'), "totalcapacity must be defined"
        assert hasattr(self.variables, 'technology_prod'), "technology_prod must be defined"
        assert hasattr(self.variables, 'technology_use'), "technology_use must be defined"
        assert len(self.model_data.settings.time_steps) != 1, "time_steps must be more than one per year to apply the ramping constraint"


    def rules(self):
        time_steps = len(self.model_data.settings.time_steps)
        # Be sure that the timeslice_fraction is a column vector
        timeslice_fraction = self.model_data.settings.timeslice_fraction
        if not isinstance(timeslice_fraction, int):
            timeslice_fraction.shape = (time_steps, 1) 
        rules = []
        
        # Read Input Paramters: ramp-up and ramp-down rates
        ramp_up_rates = self.get_parameters_from_global_or_regional_file(
            "glob_max_rate_ramp-up",
            "max_rate_ramp-up"
        )
        ramp_down_rates = self.get_parameters_from_global_or_regional_file(
            "glob_max_rate_ramp-down",
            "max_rate_ramp-down"
        )

        for reg in self.model_data.settings.regions:
            for key in self.variables.technology_prod[reg].keys():
                if key not in ["Transmission", "Demand"]:
                    reg_techs = list(self.model_data.settings.technologies[reg][key])
                    
                    for tech_indx, tech in enumerate(reg_techs):
                        for year_indx, year in enumerate(self.model_data.settings.years):

                            # Build the LHS constraint: DELTA PROD of the year
                            first_row_prod = cp.reshape(
                                self.variables.technology_prod[reg][key][
                                (year_indx * time_steps): (year_indx * time_steps + 1),
                                tech_indx: tech_indx + 1],
                                (1, self.variables.technology_prod[reg][key][0, tech_indx: tech_indx + 1].shape[0])
                            )
                            delta_prod = (self.variables.technology_prod[reg][key][
                                          (year_indx * time_steps + 1): (year_indx + 1) * time_steps,
                                          tech_indx: tech_indx + 1]
                                          - self.variables.technology_prod[reg][key][
                                            (year_indx * time_steps): ((year_indx + 1) * time_steps - 1),
                                            tech_indx: tech_indx + 1]
                                          )
                            annual_energy_prod_difference = cp.vstack([first_row_prod, delta_prod])  # shape = (ts,1)

                            # Build the LHS constraint for storage techs: DELTA USE of the year
                            if key == "Storage":
                                first_row_use = cp.reshape(
                                    self.variables.technology_use[reg][key][
                                    (year_indx * time_steps): (year_indx * time_steps + 1),
                                    tech_indx: tech_indx + 1],
                                    (1, self.variables.technology_use[reg][key][0, tech_indx: tech_indx + 1].shape[0])
                                )
                                delta_use = (self.variables.technology_use[reg][key][
                                             (year_indx * time_steps + 1): (year_indx + 1) * time_steps,
                                             tech_indx: tech_indx + 1]
                                             - self.variables.technology_use[reg][key][
                                               (year_indx * time_steps): ((year_indx + 1) * time_steps - 1),
                                               tech_indx: tech_indx + 1]
                                             )
                                annual_energy_use_difference = cp.vstack([first_row_use, delta_use])  # shape = (ts,1)

                            # Get the THEORETICAL ENERGY deliverable according only to plant capacity
                            annual_available_prod_by_timestep = theoretical_available_prod(
                                self.variables.totalcapacity[reg][key][
                                year_indx: year_indx + 1,
                                tech_indx: tech_indx + 1],  # shape = (1,1)
                                timeslice_fraction,  # shape = (ts,1)
                                self.model_data.regional_parameters[reg]["annualprod_per_unitcapacity"]
                                .loc[:, key].loc[:, tech].values[0],  # shape = (1,1)
                            ) # shape = (ts, 1)
                            
                            # Constraining the RAMP-UP
                            if ramp_up_rates.loc[:, key].loc[:, tech].values[0] < 1:
                                # Build the RHS constraint: RAMP AVAILABLE ENERGY computation
                                max_available_ramp_up = self.maximum_available_ramp(annual_available_prod_by_timestep,
                                                                    ramp_up_rates.loc[:, key].loc[:, tech].values[0]
                                                                    ) # shape = (ts,1)
                                # Apply the constraint by EXCLUDING the first timestep of the first year
                                if year_indx == 0:
                                    rules.append(
                                        max_available_ramp_up[(year_indx * time_steps + 1):
                                                              (year_indx + 1) * time_steps,
                                        :] - delta_prod >= 0      # (Y0, ts=1) 
                                    )
                                    if key == "Storage":
                                        rules.append(
                                            max_available_ramp_up[(year_indx * time_steps + 1):
                                                                  (year_indx + 1) * time_steps,
                                            :] - delta_use >= 0     # (Y0, ts=1)
                                        )
                                        
                                # Apply the FULL CONSTRAINT
                                else:
                                    rules.append(
                                        max_available_ramp_up - annual_energy_prod_difference >= 0
                                    )                                    
                                    if key == "Storage":
                                        rules.append(
                                            max_available_ramp_up - annual_energy_use_difference >= 0
                                        )

                            # Constraining the RAMP-DOWN
                            if ramp_down_rates.loc[:, key].loc[:, tech].values[0] < 1:
                                # Build the RHS constraint: RAMP AVAILABLE ENERGY computation
                                max_available_ramp_down = self.maximum_available_ramp(annual_available_prod_by_timestep,
                                                                    ramp_down_rates.loc[:, key].loc[:, tech].values[0]
                                                                    ) # shape = (ts,1)
                                # Apply the constraint by EXCLUDING the first timestep of the first year
                                if year_indx == 0:
                                    rules.append(
                                        cp.multiply(max_available_ramp_down[(year_indx * time_steps + 1):
                                                                            (year_indx + 1) * time_steps,
                                                    :], -1) - delta_prod <= 0       # (Y0, ts=1) DO NOT APPEND THE CONSTRAINT to avoid unfiseable/unbounded solutions
                                    )
                                    if key == "Storage":
                                        rules.append(
                                            cp.multiply(max_available_ramp_down[(year_indx * time_steps + 1):
                                                                                (year_indx + 1) * time_steps,
                                                        :], -1) - delta_use <= 0        # (Y0, ts=1) DO NOT APPEND THE CONSTRAINT to avoid unfiseable/unbounded solutions
                                        )

                                # Apply the FULL CONSTRAINT
                                else:
                                    rules.append(
                                        cp.multiply(max_available_ramp_down, -1) - annual_energy_prod_difference <= 0
                                    )                                  
                                    if key == "Storage":
                                        rules.append(
                                            cp.multiply(max_available_ramp_down, -1) - annual_energy_use_difference <= 0
                                        )

        return rules
    
    """
    Methods to properly choose whether to apply the triangular or trapezoidal equation to compute the RHS constraint
    """
    def maximum_available_ramp(self, annual_av_prod_by_timestep, ramp_rate):
        # threshold ramp rate = 1.66667 %/min
        # If ramp rate < threshold, it follows a triangular path, if rate > 1.66667 %/min a trapezoidal path
        if ramp_rate <= 1/60:
            max_available_ramp = self.triangular_available_ramp(annual_av_prod_by_timestep, ramp_rate)
        else:
            max_available_ramp = self.trapezoidal_available_ramp(annual_av_prod_by_timestep, ramp_rate)
        return max_available_ramp
    
    def triangular_available_ramp(self, annual_av_prod_by_timestep, ramp_rate):
        hourly_ramp_rate = 60 * ramp_rate
        rectangular_ramp = cp.multiply(annual_av_prod_by_timestep, hourly_ramp_rate)
        triangular_ramp = cp.multiply(rectangular_ramp, 0.5)
        return triangular_ramp

    def trapezoidal_available_ramp(self, annual_av_prod_by_timestep, ramp_rate):
        hourly_ramp_rate = 60 * ramp_rate
        trapezoidal_factor = 1 - 1/(2 * hourly_ramp_rate)
        trapezoidal_ramp = cp.multiply(annual_av_prod_by_timestep, trapezoidal_factor)
        return trapezoidal_ramp
    
    """
    Methods to update the input parameter templates with the new worksheets
    """
    def _required_regional_parameters(settings):
        required_parameters = {}
        for reg in settings.regions:
            indexer = create_technology_columns(
                settings.technologies[reg],
                ignored_tech_categories=["Demand", "Transmission"],
            )
            required_parameters[reg] = {
                "max_rate_ramp-up": {
                    "sheet_name": "Max_rate_ramp-up",
                    "value": 1,
                    "index": pd.Index(
                        ["Max_power_fraction_to_ramp_up [-/min]"], name="Performance Parameter"
                    ),
                    "columns": indexer,
                },
                "max_rate_ramp-down": {
                    "sheet_name": "Max_rate_ramp-down",
                    "value": 1,
                    "index": pd.Index(
                        ["Max_power_fraction_to_ramp_down [-/min]"], name="Performance Parameter"
                    ),
                    "columns": indexer,
                },
            }
        if settings.multi_node:
            required_parameters = {}

        return required_parameters


    def _required_global_parameters(settings):
        indexer_global = create_technology_columns(
            settings.technologies_glob,
            ignored_tech_categories = ["Demand", "Transmission"],
        )

        return {
            "glob_max_rate_ramp-up": {
                "sheet_name": "Max_rate_ramp-up",
                "value": 1,
                "index": pd.Index(
                    ["Max_power_fraction_to_ramp_up [-/min]"], name="Performance Parameter"
                ),
                "columns": indexer_global,
            },
            "glob_max_rate_ramp-down": {
                "sheet_name": "Max_rate_ramp-down",
                "value": 1,
                "index": pd.Index(
                    ["Max_power_fraction_to_ramp_down [-/min]"], name="Performance Parameter"
                ),
                "columns": indexer_global,
            },
        }

    """
    Method to get parameters from regional or global input file
    """
    def get_parameters_from_global_or_regional_file(self, gloal_sheet_name, regional_sheet_name):
        input_DataFrame = {}
        if self.model_data.settings.multi_node:
            input_DataFrame = self.model_data.global_parameters[gloal_sheet_name]
        else:
            for reg in self.model_data.settings.regions:
                input_DataFrame = self.model_data.regional_parameters[reg][regional_sheet_name]

        return input_DataFrame
