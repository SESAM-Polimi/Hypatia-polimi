from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.utility import available_resource_prod
from hypatia.backend.StrData import create_technology_columns
import cvxpy as cp
import pandas as pd

"""
Guarantees that generation by technologies never goes down a minimum
specified threshold, unless the plant keeps turned-off
"""
class MinTechPowerOutput(Constraint):
    def _check(self):
        assert hasattr(self.variables, 'totalcapacity'), "totalcapacity must be defined"
        assert hasattr(self.variables, 'technology_prod'), "technology_prod must be defined"


    def rules(self):
        # reshape timeslice fraction
        timeslice_fraction = self.model_data.settings.timeslice_fraction
        if not isinstance(timeslice_fraction, int):
            timeslice_fraction.shape = (len(self.model_data.settings.time_steps), 1)

        min_power_output_df = self.get_parameters_from_global_or_regional_file(
            "glob_min_tech_power_output",
            "min_tech_power_output"
        )

        time_steps = len(self.model_data.settings.time_steps)
        M = 100000000000000000000 # 1e20 is an extremely high arbitrary number (to not limit prduction or use)

        rules = []
        for reg in self.model_data.settings.regions:
            for key in self.variables.technology_prod[reg].keys():
                if key not in ["Transmission", "Demand", "Storage"]:
                    for indx, year in enumerate(self.model_data.settings.years):
                        # Available production according to resource capacity factors and plant capacity
                        available_prod = available_resource_prod(
                            self.variables.totalcapacity[reg][key][indx : indx + 1, :],
                            self.model_data.regional_parameters[reg]["res_capacity_factor"]
                            .loc[(year, slice(None)), (key, slice(None))]
                            .values,
                            timeslice_fraction,
                            self.model_data.regional_parameters[reg]["annualprod_per_unitcapacity"]
                            .loc[:, (key, slice(None))]
                            .values,
                        )
                        # Minimum available production according to the minmum power output allowed
                        min_available_prod = cp.multiply(
                            available_prod, 
                            min_power_output_df.loc[:, (key, slice(None))].values
                            )
                        
                        # If boolean = 0, this constraint brings production to zero (plant turned-off)
                        rules.append(
                            cp.multiply(
                                self.variables.boolean_for_minpower[reg][key][
                                indx * time_steps : (indx + 1) * time_steps,
                                : 
                                ],
                                M
                                )
                            - self.variables.technology_prod[reg][key][
                                indx * time_steps : (indx + 1) * time_steps,
                                :
                                ]
                            >= 0
                        )
                        # if boolean = 1, this constraint ensures that generation does not go down the minimum allowed
                        rules.append(
                            min_available_prod 
                            - cp.multiply(
                                (
                                    1 - 
                                    self.variables.boolean_for_minpower[reg][key][
                                        indx * time_steps : (indx + 1) * time_steps,
                                        :
                                        ]
                                ),
                                M
                                )
                            - self.variables.technology_prod[reg][key][
                                indx * time_steps : (indx + 1) * time_steps,
                                :
                                ]
                            <= 0
                        )

        return rules


    """
    Methods to update the input parameter templates with the new worksheets
    """
    def _required_regional_parameters(settings):
        required_parameters = {}
        for reg in settings.regions:
            indexer = create_technology_columns(
                settings.technologies[reg],
                ignored_tech_categories=["Demand", "Transmission", "Storage"],
            )
            required_parameters[reg] = {
                "min_tech_power_output": {
                    "sheet_name": "Min_power_output",
                    "value": 0,
                    "index": pd.Index(
                        ["Min relative tech power output [-]"], name="Tech Parameter"
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
            ignored_tech_categories = ["Demand", "Transmission", "Storage"],
        )

        return {
            "glob_min_tech_power_output": {
                "sheet_name": "Min_power_output",
                "value": 0,
                "index": pd.Index(
                    ["Min relative tech power output [-]"], name="Tech Parameter"
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