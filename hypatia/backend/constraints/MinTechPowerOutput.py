from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.utility import available_resource_prod
from hypatia.backend.StrData import create_technology_columns
import cvxpy as cp
import pandas as pd
import numpy as np

"""
Guarantees that generation by technologies never goes down a minimum
specified threshold, unless the plant keeps turned-off
"""
class MinTechPowerOutput(Constraint):
    def _check(self):
        assert hasattr(self.variables, 'totalcapacity'), "totalcapacity must be defined"
        assert hasattr(self.variables, 'technology_prod'), "technology_prod must be defined"


    def rules(self):
        time_steps = len(self.model_data.settings.time_steps)
        timeslice_fraction = self.model_data.settings.timeslice_fraction
        if isinstance(timeslice_fraction, int):
            timeslice_fraction_column = timeslice_fraction
            timeslice_fraction_vector = np.full(time_steps, timeslice_fraction)
        else:
            timeslice_fraction_column = np.asarray(timeslice_fraction).reshape(
                time_steps, 1
            )
            timeslice_fraction_vector = np.asarray(timeslice_fraction).reshape(
                time_steps
            )

        min_power_output_df = self.get_parameters_from_global_or_regional_file(
            "glob_min_tech_power_output",
            "min_tech_power_output"
        )

        has_modular_commitment = any(
            technologies
            for region in self.variables.units_online_technologies.values()
            for technologies in region.values()
        )
        capacity_step_df = None
        if has_modular_commitment:
            capacity_step_df = self.get_parameters_from_global_or_regional_file(
                "global_new_capacity_step",
                "new_capacity_step"
            )

        # Kept as a fallback for non-modular technologies and Operation mode.
        M = 1000000000000000

        rules = []
        for reg in self.model_data.settings.regions:
            for key in self.variables.technology_prod[reg].keys():
                if key in ["Transmission", "Demand", "Storage"]:
                    continue

                technologies = list(self.model_data.settings.technologies[reg][key])
                modular_technologies = self.variables.units_online_technologies.get(
                    reg, {}
                ).get(key, [])
                modular_column = {
                    technology: column
                    for column, technology in enumerate(modular_technologies)
                }
                if modular_technologies:
                    rules.append(self.variables.units_online[reg][key] >= 0)

                for year_index, year in enumerate(self.model_data.settings.years):
                    row_slice = slice(
                        year_index * time_steps,
                        (year_index + 1) * time_steps,
                    )
                    available_prod = available_resource_prod(
                        self.variables.totalcapacity[reg][key][
                            year_index : year_index + 1, :
                        ],
                        self.model_data.regional_parameters[reg][
                            "res_capacity_factor"
                        ].loc[(year, slice(None)), (key, slice(None))].values,
                        timeslice_fraction_column,
                        self.model_data.regional_parameters[reg][
                            "annualprod_per_unitcapacity"
                        ].loc[:, (key, slice(None))].values,
                    )

                    for technology_index, technology in enumerate(technologies):
                        min_power_output = min_power_output_df.loc[
                            :, (key, technology)
                        ].values[0]
                        if min_power_output <= 0:
                            continue

                        production = self.variables.technology_prod[reg][key][
                            row_slice, technology_index
                        ]

                        if technology in modular_column:
                            capacity_step = capacity_step_df.loc[
                                :, (key, technology)
                            ].values[0]
                            units_online = self.variables.units_online[reg][key][
                                row_slice, modular_column[technology]
                            ]
                            total_capacity = self.variables.totalcapacity[reg][key][
                                year_index, technology_index
                            ]
                            resource_capacity_factor = self.model_data.regional_parameters[
                                reg
                            ]["res_capacity_factor"].loc[
                                (year, slice(None)), (key, technology)
                            ].values.reshape(time_steps)
                            annual_production_per_capacity = (
                                self.model_data.regional_parameters[reg][
                                    "annualprod_per_unitcapacity"
                                ].loc[:, (key, technology)].values[0]
                            )
                            unit_available_production = (
                                capacity_step
                                * resource_capacity_factor
                                * timeslice_fraction_vector
                                * annual_production_per_capacity
                            )

                            rules.append(
                                cp.multiply(capacity_step, units_online)
                                <= total_capacity
                            )
                            rules.append(
                                production
                                <= cp.multiply(
                                    unit_available_production, units_online
                                )
                            )
                            rules.append(
                                production
                                >= cp.multiply(
                                    min_power_output * unit_available_production,
                                    units_online,
                                )
                            )
                        else:
                            # Legacy aggregate commitment, also used in Operation.
                            technology_on = self.variables.boolean_for_minpower[reg][key][
                                row_slice, technology_index
                            ]
                            rules.append(
                                cp.multiply(technology_on, M) - production >= 0
                            )
                            rules.append(
                                cp.multiply(
                                    min_power_output,
                                    available_prod[:, technology_index],
                                )
                                - cp.multiply(1 - technology_on, M)
                                - production
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