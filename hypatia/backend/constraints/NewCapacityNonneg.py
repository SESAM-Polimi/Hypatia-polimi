# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:11:31 2024

@author: Tommaso
"""

from hypatia.backend.constraints.Constraint import Constraint
from hypatia.utility.constants import ModelMode


"""
Defines the non-negative constraint for new capacity installed in Planning Mode
"""

class NewCapacityNonneg(Constraint):
    MODES = [ModelMode.Planning]
    
    def _check(self):
        assert self.variables.real_new_capacity != None

    def rules(self):
        rules = []
        for reg in self.model_data.settings.regions:
            for tech_newcap in self.variables.new_capacity[reg]:
                rules.append(self.variables.new_capacity[reg][tech_newcap] >= 0)
        return rules
