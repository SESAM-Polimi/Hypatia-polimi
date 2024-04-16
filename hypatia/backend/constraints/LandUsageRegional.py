# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:52:38 2023

@author: GP
"""

from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.constants import (
    ModelMode,
    TopologyType
)
from hypatia.utility.utility import create_technology_columns
import pandas as pd

"""
Defines the upper limit on the land usage
of each technology within each region
"""
class LandUsageRegional(Constraint):
    MODES = [ModelMode.Planning]

    def _check(self):
        assert hasattr(self.variables, 'totalcapacity'), "totalcapacity must be defined"
        assert hasattr(self.variables, 'land_usage'), "land_usage must be defined"

    def rules(self):
        rules = []
        for reg in self.model_data.settings.regions:
            for key, value in self.variables.land_usage[reg].items():
                rules.append(
                    value <= self.model_data.regional_parameters[reg]["max_land_usage"].loc[:, key]
                )
        return rules

    def _required_regional_parameters(settings):
        required_parameters = {}
        for reg in settings.regions:
            indexer = create_technology_columns(
                settings.technologies[reg],
                ignored_tech_categories=["Demand"],
            )

            required_parameters[reg] = {
                "max_land_usage": {
                    "sheet_name": "Max_land_usage",
                    "value": 1e20,
                    "index": pd.Index(settings.years, name="Years"),
                    "columns": indexer,
                },
            }
        return required_parameters