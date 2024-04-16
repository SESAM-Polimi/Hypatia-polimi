from hypatia.backend.ModelData import ModelData
from hypatia.utility.constants import ModelMode, EnsureFeasibility
import numpy as np
import cvxpy as cp
import itertools as it
from hypatia.utility.utility import *
from collections import defaultdict

class ModelVariables():
    def __init__(self, model_data: ModelData):
        self.model_data = model_data
        
        # decision variables
        self.technology_prod = self.create_technology_prod()
        self.technology_use = self.create_technology_use()
        self.line_import = self.create_line_import_export()
        self.line_export = self.create_line_import_export()
        self.new_capacity = self.create_new_capacity()
        self.line_new_capacity = self.create_line_new_capacity()
        self.unmetdemandbycarrier = self.create_unmet_demand_by_carrier()

        # intermediate variables
        self._balance_()
        if self.model_data.settings.mode == ModelMode.Planning:
            self._calc_variable_planning()

            if self.model_data.settings.multi_node:
                self._calc_variable_planning_line()

        elif self.model_data.settings.mode == ModelMode.Operation:
            self._calc_variable_operation()
            if self.model_data.settings.multi_node:
                self._calc_variable_operation_line()
        
        self._calc_variable_storage_SOC()
        self._calc_emission_variables()
        
        if self.model_data.settings.mode == ModelMode.Planning:
            self._calc_regional_cost_planning()

            if not self.model_data.settings.multi_node:
                self._calc_tot_cost_singlenode()
            else:
                self._calc_lines_cost_planning()
                self._calc_tot_cost_multinode()

        elif self.model_data.settings.mode == ModelMode.Operation:
            self._calc_regional_cost_operation()

            if not self.model_data.settings.multi_node:
                self._calc_tot_cost_singlenode()
            else:
                self._calc_lines_cost_operation()
                self._calc_tot_cost_multinode()
                
        self._calc_regional_emission()
        self._calc_tot_emission()

        # Reshape the demand
        self.demand = {
            reg: self.model_data.regional_parameters[reg]["demand"] for reg in self.model_data.settings.regions
        }
        
        self.carrier_ratio_in = {}
        for reg in self.model_data.settings.regions:
            if "Conversion_plus" not in self.model_data.settings.technologies[reg].keys(): 
                continue
            self.carrier_ratio_in[reg] = self.model_data.regional_parameters[reg]["carrier_ratio_in"]
            
        self.carrier_ratio_out = {}
        for reg in self.model_data.settings.regions:
            if "Conversion_plus" not in self.model_data.settings.technologies[reg].keys(): 
                continue
            self.carrier_ratio_out[reg] = self.model_data.regional_parameters[reg]["carrier_ratio_out"]
        

    """
    Primary variables
    """
    # explain this variable
    def create_technology_prod(self):
        technology_prod = {}
        for reg in self.model_data.settings.regions:
            regional_prod = {}
            for technology_category in self.model_data.settings.technologies[reg].keys():
                if technology_category != "Demand":
                    regional_prod[technology_category] = cp.Variable(
                        shape=(
                            len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),
                            len(self.model_data.settings.technologies[reg][technology_category]),
                        ),
                        nonneg=True,
                    )
            technology_prod[reg] = regional_prod
        return technology_prod

    # explain this variable
    def create_technology_use(self):
        technology_use = {}
        for reg in self.model_data.settings.regions:
            regional_use = {}
            for technology_category in self.model_data.settings.technologies[reg].keys():
                if technology_category != "Demand" and technology_category != "Supply":
                    regional_use[technology_category] = cp.Variable(
                        shape=(
                            len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),
                            len(self.model_data.settings.technologies[reg][technology_category]),
                        ),
                        nonneg=True,
                    )
            technology_use[reg] = regional_use
        return technology_use

    # explain this variable
    def create_line_import_export(self):
        if not self.model_data.settings.multi_node:
            return None

        import_export = {}
        for reg in self.model_data.settings.regions:
            regional_import_export = {}
            for other_reg in self.model_data.settings.regions:
                if reg != other_reg:
                    regional_import_export[other_reg] = cp.Variable(
                        shape=(
                            len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),
                            len(self.model_data.settings.global_settings["Carriers_glob"].index),
                        ),
                        nonneg=True,
                    )
            import_export[reg] = regional_import_export
        return import_export

    # explain this variable
    def create_new_capacity(self):
        if self.model_data.settings.mode != ModelMode.Planning:
            return None

        new_capacity = {}
        for reg in self.model_data.settings.regions:
            regional_new_capacity = {}
            for tech_type in self.model_data.settings.technologies[reg].keys():
                    if tech_type != "Demand":
                        regional_new_capacity[tech_type] = cp.Variable(
                            shape=(
                                len(self.model_data.settings.years),
                                len(self.model_data.settings.technologies[reg][tech_type]),
                            ),
                            nonneg=True,
                        )
                       
            new_capacity[reg] = regional_new_capacity
        return new_capacity

    # explain this variable
    def create_line_new_capacity(self):
        if self.model_data.settings.mode != ModelMode.Planning:
            return None

        if not self.model_data.settings.multi_node:
            return None

        line_newcapacity = {}
        for line in self.model_data.settings.lines_list:
            line_newcapacity[line] = cp.Variable(
                shape=(
                    len(self.model_data.settings.years),
                    len(self.model_data.settings.global_settings["Carriers_glob"].index),
                ),
                nonneg=True,
            )
        return line_newcapacity
    
    def create_unmet_demand_by_carrier(self):

        unmetdemandbycarrier = {}
        for reg in self.model_data.settings.regions:

            unmetdemandbycarrier_regional = {}

            for carr in self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]:
                
                unmetdemandbycarrier_regional[carr] = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),)
                )
                
                for key in self.model_data.settings.technologies[reg].keys():
                    
                    if not key == "Demand":
                        continue

                    for indx, tech in enumerate(self.model_data.settings.technologies[reg][key]):

                        if (
                            carr
                            in self.model_data.settings.regional_settings[reg]["Carrier_input"]
                            .loc[
                                self.model_data.settings.regional_settings[reg]["Carrier_input"]["Technology"]
                                == tech
                            ]["Carrier_in"]
                            .values
                        ):

                            unmetdemandbycarrier_regional[carr] += cp.Variable(
                                shape=(
                                    len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),
                                ),
                                nonneg=True,
                            )                                 

            unmetdemandbycarrier[reg] = unmetdemandbycarrier_regional
            
        return unmetdemandbycarrier

    """
    Secondary variables
    """
    def _calc_variable_planning(self):

        """
        Calculates all the cost components of the objective function and the
        intermediate variables in the planning mode, for each region
        """

        self.real_new_capacity = {}
        self.cost_inv = {}
        self.cost_inv_tax = {}
        self.cost_inv_sub = {}
        self.cost_inv_fvalue = {}
        self.salvage_inv = {}
        self.accumulated_newcapacity = {}
        self.totalcapacity = {}
        self.cost_fix = {}
        self.cost_fix_tax = {}
        self.cost_fix_sub = {}
        self.decommissioned_capacity = {}
        self.cost_decom = {}
        self.cost_variable = {}
        self.production_annual = {}
        self.consumption_annual = {}
        self.land_usage = {}
        self.residual_capacity = {}
        self.unmet_demand_annual = {}
        self.cost_unmet_demand = {}

        for reg in self.model_data.settings.regions:
            
            real_new_capacity_regional = {}
            cost_inv_regional = {}
            cost_inv_tax_regional = {}
            cost_inv_sub_regional = {}
            cost_fvalue_regional = {}
            salvage_inv_regional = {}
            accumulated_newcapacity_regional = {}
            totalcapacity_regional = {}
            cost_fix_regional = {}
            cost_fix_tax_regional = {}
            cost_fix_Sub_regional = {}
            decomcapacity_regional = {}
            cost_decom_regional = {}
            cost_variable_regional = {}
            production_annual_regional = {}
            consumption_annual_regional = {}
            land_usage_regional = {}
            residual_capacity_regional = {}
            unmet_demand_annual_regional = {}
            cost_unmet_demand_regional = {}

            for key in self.new_capacity[reg].keys():

                real_new_capacity_regional[key] = shift_new_cap(
                    self.new_capacity[reg][key], 
                    self.model_data.settings.technologies[reg][key], 
                    self.model_data.regional_parameters[reg]["time_of_construction"].loc[:, key],
                    self.model_data.settings.years)

                (
                    cost_inv_regional[key],
                    cost_inv_tax_regional[key],
                    cost_inv_sub_regional[key],
                ) = invcosts(
                    self.model_data.regional_parameters[reg]["tech_inv"][key],
                    self.new_capacity[reg][key],
                    self.model_data.regional_parameters[reg]["inv_taxsub"]["Tax"][key],
                    self.model_data.regional_parameters[reg]["inv_taxsub"]["Sub"][key],
                )

                salvage_inv_regional[key] = cp.multiply(
                    salvage_factor(
                        self.model_data.settings.years,
                        self.model_data.settings.technologies[reg][key],
                        self.model_data.regional_parameters[reg]["tech_lifetime"].loc[:, key],
                        self.model_data.regional_parameters[reg]["interest_rate"].loc[:, key],
                        self.model_data.regional_parameters[reg]["discount_rate"],
                        self.model_data.regional_parameters[reg]["economic_lifetime"].loc[:, key],
                    ),
                    cost_inv_regional[key],
                )

                accumulated_newcapacity_regional[key] = newcap_accumulated(
                    real_new_capacity_regional[key],
                    self.model_data.settings.technologies[reg][key],
                    self.model_data.settings.years,
                    self.model_data.regional_parameters[reg]["tech_lifetime"].loc[:, key],
                )

                totalcapacity_regional[key] = (
                    accumulated_newcapacity_regional[key]
                    + self.model_data.regional_parameters[reg]["tech_residual_cap"].loc[:, key]
                )

                (
                    cost_fix_regional[key],
                    cost_fix_tax_regional[key],
                    cost_fix_Sub_regional[key],
                ) = fixcosts(
                    self.model_data.regional_parameters[reg]["tech_fixed_cost"][key],
                    totalcapacity_regional[key],
                    self.model_data.regional_parameters[reg]["fix_taxsub"]["Tax"][key],
                    self.model_data.regional_parameters[reg]["fix_taxsub"]["Sub"][key],
                )

                decomcapacity_regional[key] = decomcap(
                    real_new_capacity_regional[key],
                    self.model_data.settings.technologies[reg][key],
                    self.model_data.settings.years,
                    self.model_data.regional_parameters[reg]["tech_lifetime"].loc[:, key],
                )

                cost_decom_regional[key] = cp.multiply(
                    self.model_data.regional_parameters[reg]["tech_decom_cost"].loc[:, key].values,
                    decomcapacity_regional[key],
                )

                production_annual_regional[key] = annual_activity(
                    self.technology_prod[reg][key],
                    self.model_data.settings.years,
                    self.model_data.settings.time_steps,
                )
                
                if key != "Demand" and key != "Supply": 
                
                    consumption_annual_regional[key] = annual_activity(
                        self.technology_use[reg][key],
                        self.model_data.settings.years,
                        self.model_data.settings.time_steps,
                    )            

                cost_variable_regional[key] = cp.multiply(
                    production_annual_regional[key],
                    self.model_data.regional_parameters[reg]["tech_var_cost"].loc[:, key],
                )

                cost_fvalue_regional[key] = invcosts_annuity(
                    cost_inv_regional[key],
                    self.model_data.regional_parameters[reg]["interest_rate"].loc[:, key],
                    self.model_data.regional_parameters[reg]["economic_lifetime"].loc[:, key],
                    self.model_data.settings.technologies[reg][key],
                    self.model_data.settings.years,
                    self.model_data.regional_parameters[reg]["discount_rate"],
                )  
                
                land_usage_regional[key] = cp.multiply(
                    totalcapacity_regional[key],self.model_data.regional_parameters[reg]["specific_land_usage"].loc[:, key]
                ) 
                
                residual_capacity_regional[key] = self.model_data.regional_parameters[reg]["tech_residual_cap"].loc[:, key]
                
               
            for carr in self.unmetdemandbycarrier[reg].keys():
            
                unmet_demand_annual_regional[carr] = unmet_demand_function(
                    self.unmetdemandbycarrier[reg][carr],
                    self.model_data.settings.years,
                    self.model_data.settings.time_steps                
                    )
            
                cost_unmet_demand_regional[carr] = unmet_demand_annual_regional[carr] * 1e10
                
            self.real_new_capacity[reg] = real_new_capacity_regional
            self.cost_inv[reg] = cost_inv_regional
            self.cost_inv_tax[reg] = cost_inv_tax_regional
            self.cost_inv_sub[reg] = cost_inv_sub_regional
            self.salvage_inv[reg] = salvage_inv_regional
            self.totalcapacity[reg] = totalcapacity_regional
            self.cost_fix[reg] = cost_fix_regional
            self.cost_fix_tax[reg] = cost_fix_tax_regional
            self.cost_fix_sub[reg] = cost_fix_Sub_regional
            self.decommissioned_capacity[reg] = decomcapacity_regional
            self.cost_decom[reg] = cost_decom_regional
            self.cost_variable[reg] = cost_variable_regional
            self.cost_inv_fvalue[reg] = cost_fvalue_regional
            self.production_annual[reg] = production_annual_regional
            self.consumption_annual[reg] = consumption_annual_regional
            self.land_usage[reg] = land_usage_regional
            self.residual_capacity[reg] = residual_capacity_regional 
            self.unmet_demand_annual[reg] = unmet_demand_annual_regional 
            self.cost_unmet_demand[reg] = cost_unmet_demand_regional
            

    def _calc_variable_planning_line(self):

        """
        Calculates all the cost and intermediate variables related to the inter-
        regional links in the planning mode
        """

        self.real_new_line_capacity = {}
        self.cost_inv_line = {}
        self.line_accumulated_newcapacity = {}
        self.line_totalcapacity = {}
        self.cost_fix_line = {}
        self.line_decommissioned_capacity = {}
        self.cost_decom_line = {}

        for key in self.line_new_capacity.keys():
            
            self.real_new_line_capacity[key] = shift_new_line_cap(
                self.line_new_capacity[key], 
                self.model_data.settings.global_settings["Carriers_glob"]["Carrier"], 
                self.model_data.trade_parameters["line_time_of_construction"].loc[:, key],
                self.model_data.settings.years)

            self.cost_inv_line[key] = cp.multiply(
                self.model_data.trade_parameters["line_inv"].loc[:, key].values,
                self.line_new_capacity[key],
            )

            self.line_accumulated_newcapacity[key] = line_newcap_accumulated(
                self.real_new_line_capacity[key],
                self.model_data.settings.global_settings["Carriers_glob"]["Carrier"],
                self.model_data.settings.years,
                self.model_data.trade_parameters["line_lifetime"].loc[:, key],
            )

            self.line_totalcapacity[key] = (
                self.line_accumulated_newcapacity[key]
                + self.model_data.trade_parameters["line_residual_cap"].loc[:, key].values
            )

            self.cost_fix_line[key] = cp.multiply(
                self.model_data.trade_parameters["line_fixed_cost"].loc[:, key].values,
                self.line_totalcapacity[key],
            )

            self.line_decommissioned_capacity[key] = line_decomcap(
                self.real_new_line_capacity[key],
                self.model_data.settings.global_settings["Carriers_glob"]["Carrier"],
                self.model_data.settings.years,
                self.model_data.trade_parameters["line_lifetime"].loc[:, key],
            )

            self.cost_decom_line[key] = cp.multiply(
                self.model_data.trade_parameters["line_decom_cost"].loc[:, key].values,
                self.line_decommissioned_capacity[key],
            )
            
        
        self.cost_variable_line = line_varcost(
            self.model_data.trade_parameters["line_var_cost"],
            self.line_import,
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.time_steps,
            self.model_data.settings.lines_list,
        )
        
        self.line_import_annual = line_annual_activity(
            self.line_import,
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.time_steps,
            )
        
        self.line_export_annual = line_annual_activity(
            self.line_export,
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.time_steps,
            )

    def _calc_variable_operation(self):

        """
        Calculates all the cost components of the objective function and the
        intermediate variables in the operation mode, for each region
        """

        self.totalcapacity = {}
        self.cost_fix = {}
        self.cost_fix_tax = {}
        self.cost_fix_sub = {}
        self.cost_variable = {}
        self.production_annual = {}
        self.consumption_annual = {}
        self.residual_capacity = {}
        self.unmet_demand_annual = {} 
        self.cost_unmet_demand = {}
        
        for reg in self.model_data.settings.regions:

            totalcapacity_regional = {}
            cost_fix_regional = {}
            cost_fix_tax_regional = {}
            cost_fix_Sub_regional = {}
            cost_variable_regional = {}
            production_annual_regional = {}
            consumption_annual_regional = {}
            residual_capacity_regional = {}
            unmet_demand_annual_regional = {}
            cost_unmet_demand_regional = {}            

            for key in self.model_data.settings.technologies[reg].keys():

                if key != "Demand":

                    totalcapacity_regional[key] = (
                        self.model_data.regional_parameters[reg]["tech_residual_cap"].loc[:, key].values
                    )

                    (
                        cost_fix_regional[key],
                        cost_fix_tax_regional[key],
                        cost_fix_Sub_regional[key],
                    ) = fixcosts(
                        self.model_data.regional_parameters[reg]["tech_fixed_cost"][key],
                        totalcapacity_regional[key],
                        self.model_data.regional_parameters[reg]["fix_taxsub"]["Tax"][key],
                        self.model_data.regional_parameters[reg]["fix_taxsub"]["Sub"][key],
                    )

                    production_annual_regional[key] = annual_activity(
                        self.technology_prod[reg][key],
                        self.model_data.settings.years,
                        self.model_data.settings.time_steps,
                    )
                    
                    if key != "Demand" and key != "Supply": 
                    
                        consumption_annual_regional[key] = annual_activity(
                            self.technology_use[reg][key],
                            self.model_data.settings.years,
                            self.model_data.settings.time_steps,
                        ) 
                    
                    cost_variable_regional[key] = cp.multiply(
                        production_annual_regional[key],
                        self.model_data.regional_parameters[reg]["tech_var_cost"].loc[:, key],
                    )
                    
                    residual_capacity_regional[key] = self.model_data.regional_parameters[reg]["tech_residual_cap"].loc[:, key]
                    
            for carr in self.unmetdemandbycarrier[reg].keys():
                
                unmet_demand_annual_regional[carr] = unmet_demand_function(
                    self.unmetdemandbycarrier[reg][carr],
                    self.model_data.settings.years,
                    self.model_data.settings.time_steps                
                    )
            
                cost_unmet_demand_regional[carr] = unmet_demand_annual_regional[carr] * 1e10

            self.totalcapacity[reg] = totalcapacity_regional
            self.cost_fix[reg] = cost_fix_regional
            self.cost_fix_tax[reg] = cost_fix_tax_regional
            self.cost_fix_sub[reg] = cost_fix_Sub_regional
            self.cost_variable[reg] = cost_variable_regional
            self.production_annual[reg] = production_annual_regional
            self.consumption_annual[reg] = consumption_annual_regional
            self.residual_capacity[reg] = residual_capacity_regional
            self.unmet_demand_annual[reg] = unmet_demand_annual_regional 
            self.cost_unmet_demand[reg] = cost_unmet_demand_regional
        

    def _calc_variable_operation_line(self):

        """
        Calculates all the cost and intermediate variables related to the inter-
        regional links in the operation mode
        """

        self.line_totalcapacity = {}
        self.cost_fix_line = {}
        for key in self.model_data.settings.lines_list:

            self.line_totalcapacity[key] = (
                self.model_data.trade_parameters["line_residual_cap"].loc[:, key].values
            )
            self.cost_fix_line[key] = cp.multiply(
                self.model_data.trade_parameters["line_fixed_cost"].loc[:, key].values,
                self.line_totalcapacity[key],
            )

        self.cost_variable_line = line_varcost(
            self.model_data.trade_parameters["line_var_cost"],
            self.line_import,
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.time_steps,
            self.model_data.settings.lines_list,
        )
        
        self.line_import_annual = line_annual_activity(
            self.line_import,
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.time_steps,
            )
        
        self.line_export_annual = line_annual_activity(
            self.line_export,
            self.model_data.settings.regions,
            self.model_data.settings.years,
            self.model_data.settings.time_steps,
            )

    def _calc_variable_storage_SOC(self):

        """
        Calculates the annual state of charge of the on grid storage technologies,
        in the models with hourly temporal resolution
        """

        self.storage_SOC = {}

        for reg in get_regions_with_storage(self.model_data.settings):

            self.storage_SOC[reg] = storage_state_of_charge(
                self.model_data.regional_parameters[reg]["storage_initial_SOC"],
                self.technology_use[reg]["Storage"],
                self.technology_prod[reg]["Storage"],
                self.model_data.settings.years,
                self.model_data.settings.time_steps,
                self.model_data.regional_parameters[reg]["storage_charge_efficiency"],
                self.model_data.regional_parameters[reg]["storage_discharge_efficiency"],
                self.totalcapacity[reg]["Storage"]
            )

    def _balance_(self):

        """
        Creates the dictionaries for the annual total production by each technology,
        total consumption by each technology, total import,total exports and total final demand
        of each energy carrier within each region
        """

        self.totalusebycarrier = {}
        self.totalprodbycarrier = {}
        self.totalimportbycarrier = {}
        self.totalexportbycarrier = {}
        self.totaldemandbycarrier = {}

        for reg in self.model_data.settings.regions:

            totalusebycarrier_regional = {}
            totalprodbycarrier_regional = {}
            totalimportbycarrier_regional = {}
            totalexportbycarrier_regional = {}
            totaldemandbycarrier_regional = {}

            for carr in self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]:

                totalusebycarrier_regional[carr] = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),)
                )
                totalprodbycarrier_regional[carr] = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),)
                )
                totalimportbycarrier_regional[carr] = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),)
                )
                totalexportbycarrier_regional[carr] = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),)
                )
                totaldemandbycarrier_regional[carr] = np.zeros(
                    (len(self.model_data.settings.years) * len(self.model_data.settings.time_steps),)
                )                             

                for key in self.model_data.settings.technologies[reg].keys():

                    for indx, tech in enumerate(self.model_data.settings.technologies[reg][key]):

                        if (
                            carr
                            in self.model_data.settings.regional_settings[reg]["Carrier_input"]
                            .loc[
                                self.model_data.settings.regional_settings[reg]["Carrier_input"]["Technology"]
                                == tech
                            ]["Carrier_in"]
                            .values
                        ):

                            if key == "Conversion_plus":

                                totalusebycarrier_regional[carr] += cp.multiply(
                                    self.technology_use[reg][key][
                                        :, indx
                                    ],
                                    self.model_data.regional_parameters[reg]["carrier_ratio_in"][
                                        (tech, carr)
                                    ].values,
                                )

                            elif key == "Demand":

                                totaldemandbycarrier_regional[carr] += self.model_data.regional_parameters[
                                    reg
                                ]["demand"][tech].values                                

                            elif key != "Supply":

                                totalusebycarrier_regional[carr] += self.technology_use[reg][key][:, indx]

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

                                totalprodbycarrier_regional[carr] += cp.multiply(
                                    self.technology_prod[reg][key][
                                        :, indx
                                    ],
                                    self.model_data.regional_parameters[reg]["carrier_ratio_out"][
                                        (tech, carr)
                                    ].values,
                                )
                            else:

                                totalprodbycarrier_regional[carr] += self.technology_prod[reg][key][:, indx]

                if len(self.model_data.settings.regions) > 1:

                    for key in self.line_import[reg].keys():

                        if "{}-{}".format(reg, key) in self.model_data.settings.lines_list:

                            line_eff = (
                                pd.concat(
                                    [
                                        self.model_data.trade_parameters["line_eff"][
                                            ("{}-{}".format(reg, key), carr)
                                        ]
                                    ]
                                    * len(self.model_data.settings.time_steps)
                                )
                                .sort_index()
                                .values
                            )
                            
                            line_length = (
                                pd.concat(
                                    [
                                        self.model_data.trade_parameters["line_length"][
                                            ("{}-{}".format(reg, key), carr)
                                        ]
                                    ]
                                    * len(self.model_data.settings.time_steps)
                                )
                                .sort_index()
                                .values
                            )

                        elif "{}-{}".format(key, reg) in self.model_data.settings.lines_list:

                            line_eff = (
                                pd.concat(
                                    [
                                        self.model_data.trade_parameters["line_eff"][
                                            ("{}-{}".format(key, reg), carr)
                                        ]
                                    ]
                                    * len(self.model_data.settings.time_steps)
                                )
                                .sort_index()
                                .values
                            )
                            
                            line_length = (
                                pd.concat(
                                    [
                                        self.model_data.trade_parameters["line_length"][
                                            ("{}-{}".format(key, reg), carr)
                                        ]
                                    ]
                                    * len(self.model_data.settings.time_steps)
                                )
                                .sort_index()
                                .values
                            )
                            
                        length_ratio = cp.multiply(line_length, 0.01)
                        line_efficiency = (cp.exp(cp.multiply(np.log(line_eff), length_ratio)))

                        totalimportbycarrier_regional[carr] += cp.multiply(
                            self.line_import[reg][key][
                                :,
                                list(
                                    self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]
                                ).index(carr),
                            ],
                            line_efficiency,
                        )

                        totalexportbycarrier_regional[carr] += self.line_export[reg][key][
                                :,
                                list(
                                    self.model_data.settings.global_settings["Carriers_glob"]["Carrier"]
                                ).index(carr),
                            ]
            
            self.totalusebycarrier[reg] = totalusebycarrier_regional
            self.totalprodbycarrier[reg] = totalprodbycarrier_regional
            self.totalimportbycarrier[reg] = totalimportbycarrier_regional
            self.totalexportbycarrier[reg] = totalexportbycarrier_regional
            self.totaldemandbycarrier[reg] = totaldemandbycarrier_regional


    def _calc_emission_variables(self):
        self.emission_by_region = {}
        self.captured_emission_by_region = {}
        self.emission_cost_by_region = {}
        self.used_emissions_by_region = {}
        self.emission_by_type = {}
        self.captured_emission_by_type = {}
        self.emission_cost_by_type = {}
        self.used_emissions_by_type = {}

        for reg in self.model_data.settings.regions:
            emissions_regional = defaultdict(dict)
            emission_cost_regional = defaultdict(dict)
            total_captured_emissions_regional = defaultdict(dict)
            used_emissions_regional = defaultdict(dict)
            for key in self.model_data.settings.technologies[reg].keys():
                if key == "Demand" or key == "Transmission" or key == "Storage":
                    continue

                for emission_type in get_emission_types(self.model_data.settings.global_settings):

                    regional_emissions = []
                    total_captured_emissions = []
                    used_emissions = []                    
                    
                    for indx,tech in enumerate(self.model_data.regional_parameters[reg]["specific_emission"][emission_type].loc[:, key]):
                        if self.model_data.regional_parameters[reg]["specific_emission"][emission_type].loc[:, key].iloc[:,indx].values[0] >= 0:
                            
                            regional_emissions.append(cp.reshape(cp.multiply(
                                self.production_annual[reg][key][:,indx],
                                cp.multiply(self.model_data.regional_parameters[reg]["specific_emission"][emission_type].loc[:, key].iloc[:,indx].values,
                                (np.ones((len(self.model_data.settings.years),))
                                 -self.model_data.regional_parameters[reg]["emission_capture_efficiency"][emission_type].loc[:, key].iloc[:,indx].values))), 
                                (len(self.model_data.settings.years),1)
                            ))
                            
                            total_captured_emissions.append(cp.reshape(cp.multiply(
                                self.production_annual[reg][key][:,indx],
                                cp.multiply(self.model_data.regional_parameters[reg]["specific_emission"][emission_type].loc[:, key].iloc[:,indx].values,
                                self.model_data.regional_parameters[reg]["emission_capture_efficiency"][emission_type].loc[:, key].iloc[:,indx].values)), 
                                (len(self.model_data.settings.years),1)
                            ))
                            
                            used_emissions.append(cp.reshape(cp.multiply(
                                self.production_annual[reg][key][:,indx],
                                np.zeros((self.production_annual[reg][key][:,indx].shape[0],))),
                                (len(self.model_data.settings.years),1)
                            ))
                            
                        else:
                            
                            regional_emissions.append(cp.reshape(cp.multiply(
                                self.production_annual[reg][key][:,indx],
                                np.zeros((self.production_annual[reg][key][:,indx].shape[0],))),
                                (len(self.model_data.settings.years),1)
                            ))
                            
                            total_captured_emissions.append(cp.reshape(cp.multiply(
                                self.production_annual[reg][key][:,indx],
                                np.zeros((self.production_annual[reg][key][:,indx].shape[0],))),
                                (len(self.model_data.settings.years),1)
                            ))
                            
                            used_emissions.append(cp.reshape(cp.multiply(
                                self.production_annual[reg][key][:,indx],
                                -self.model_data.regional_parameters[reg]["specific_emission"][emission_type].loc[:, key].iloc[:,indx].values),
                                (len(self.model_data.settings.years),1)
                            ))
                    
                    emissions_regional[emission_type][key] = cp.hstack(regional_emissions) 
                    total_captured_emissions_regional[emission_type][key] = cp.hstack(total_captured_emissions) 
                    used_emissions_regional[emission_type][key] = cp.hstack(used_emissions)
                    emission_cost_regional[emission_type][key] = cp.multiply(
                        emissions_regional[emission_type][key],
                        self.model_data.regional_parameters[reg]["emission_tax"][emission_type].loc[:, key],
                    )

            self.emission_by_region[reg] = emissions_regional
            self.captured_emission_by_region[reg] = total_captured_emissions_regional
            self.emission_cost_by_region[reg] = emission_cost_regional
            self.used_emissions_by_region[reg] = used_emissions_regional

        self.emission_by_type = self.flip_keys(self.emission_by_region)
        self.captured_emission_by_type = self.flip_keys(self.captured_emission_by_region)
        self.emission_cost_by_type = self.flip_keys(self.emission_cost_by_region)
        self.used_emissions_by_type = self.flip_keys(self.used_emissions_by_region)


    def flip_keys(self, d):
        result = defaultdict(dict)
        for key, value in d.items():
                for k, v in value.items():
                    result[k][key] = v
        return result
    
    def _calc_regional_cost_planning(self):
        
        self.totalcost_allregions = np.zeros((len(self.model_data.settings.years), 1))
        self.inv_allregions = 0
        years = -1 * np.arange(len(self.model_data.settings.years))

        for reg in self.model_data.settings.regions:

            totalcost_regional = np.zeros((len(self.model_data.settings.years), 1))

            for ctgry in self.model_data.settings.technologies[reg].keys():

                if ctgry != "Demand":

                    totalcost_regional += cp.sum(
                        self.cost_inv_tax[reg][ctgry]
                        - self.cost_inv_sub[reg][ctgry]
                        + self.cost_fix[reg][ctgry]
                        + self.cost_fix_tax[reg][ctgry]
                        - self.cost_fix_sub[reg][ctgry]
                        + self.cost_variable[reg][ctgry]
                        + self.cost_decom[reg][ctgry]
                        - self.salvage_inv[reg][ctgry],
                        axis=1,
                    )

                    self.inv_allregions += self.cost_inv_fvalue[reg][ctgry]

                    if ctgry != "Transmission" and ctgry != "Storage":
                        for emission_type in get_emission_types(self.model_data.settings.global_settings):
                            totalcost_regional += cp.sum(
                                self.emission_cost_by_region[reg][emission_type][ctgry], axis=1
                            )
                            
            for carr in self.unmetdemandbycarrier[reg].keys():
                            
                totalcost_regional += cp.sum(self.cost_unmet_demand[reg][carr],axis=1)

            discount_factor = (
                1 + self.model_data.regional_parameters[reg]["discount_rate"]["Annual Discount Rate"].values
            )

            totalcost_regional_discounted = cp.multiply(
                totalcost_regional, np.power(discount_factor, years)
            )
            self.totalcost_allregions += totalcost_regional_discounted
            
    def _calc_regional_cost_operation(self):
        
        self.totalcost_allregions = 0
        for reg in self.model_data.settings.regions:

            totalcost_regional = 0

            for ctgry in self.model_data.settings.technologies[reg].keys():

                if ctgry != "Demand":

                    totalcost_regional += cp.sum(
                        self.cost_fix[reg][ctgry]
                        + self.cost_fix_tax[reg][ctgry]
                        - self.cost_fix_sub[reg][ctgry]
                        + self.cost_variable[reg][ctgry]
                    )

                    if ctgry != "Transmission" and ctgry != "Storage":
                        for emission_type in get_emission_types(self.model_data.settings.global_settings):
                            totalcost_regional += cp.sum(
                                self.emission_cost_by_region[reg][emission_type][ctgry], axis=1
                            )
                            
            for carr in self.unmetdemandbycarrier[reg].keys():
                            
                totalcost_regional += cp.sum(self.cost_unmet_demand[reg][carr],axis=1)

            self.totalcost_allregions += totalcost_regional
            
    def _calc_lines_cost_planning(self):
        
        years = -1 * np.arange(len(self.model_data.settings.years))
        self.totalcost_lines = np.zeros((len(self.model_data.settings.years), 1))

        for line in self.model_data.settings.lines_list:

            self.totalcost_lines += cp.sum(
                self.cost_inv_line[line]
                + self.cost_fix_line[line]
                + self.cost_decom_line[line],
                axis=1,
            )

        for reg in self.model_data.settings.regions:

            for key, value in self.cost_variable_line[reg].items():

                self.totalcost_lines += cp.sum(value, axis=1)

        discount_factor_global = (
            1
            + self.model_data.global_parameters["global_discount_rate"][
                "Annual Discount Rate"
            ].values
        )

        self.totalcost_lines_discounted = cp.multiply(
            self.totalcost_lines, np.power(discount_factor_global, years)
        )
        
    def _calc_lines_cost_operation(self):
    
        self.totalcost_lines = 0

        for line in self.model_data.settings.lines_list:

            self.totalcost_lines += cp.sum(self.cost_fix_line[line], axis=1)

        for reg in self.model_data.settings.regions:

            for key, value in self.cost_variable_line[reg].items():

                self.totalcost_lines += cp.sum(value, axis=1)
                
    def _calc_tot_cost_singlenode(self):

        if self.model_data.settings.mode == ModelMode.Planning:
            self.tot_cost_single_node = (
                cp.sum(self.totalcost_allregions) + self.inv_allregions
            )

        elif self.model_data.settings.mode == ModelMode.Operation:

            self.tot_cost_single_node = self.totalcost_allregions

    def _calc_tot_cost_multinode(self):

        if self.model_data.settings.mode == ModelMode.Planning:

            self.tot_cost_multi_node = (
                cp.sum(self.totalcost_lines_discounted + self.totalcost_allregions)
                + self.inv_allregions
            )

        elif self.model_data.settings.mode == ModelMode.Operation:

            self.tot_cost_multi_node = self.totalcost_allregions + self.totalcost_lines
            
    def _calc_regional_emission(self):

        self.totalemission_allregions = np.zeros((len(self.model_data.settings.years), 1))
        
        for reg in self.model_data.settings.regions:
            
            totalemission_regional = np.zeros((len(self.model_data.settings.years), 1))
            
            for emission_type in get_emission_types(self.model_data.settings.global_settings):
                
                totalemission_regional_by_type = np.zeros((len(self.model_data.settings.years), 1))
                
                for ctgry in self.model_data.settings.technologies[reg].keys():

                    if ctgry != "Demand" and ctgry != "Transmission" and ctgry != "Storage":
                        
                        totalemission_regional_by_type += cp.sum(
                            self.emission_by_region[reg][emission_type][ctgry], axis=1
                        )
                        
                totalemission_regional += totalemission_regional_by_type
                
            self.totalemission_allregions += totalemission_regional
            
    def _calc_tot_emission(self):

        if self.model_data.settings.mode == ModelMode.Planning:
            self.tot_emissions = cp.sum(self.totalemission_allregions)

        elif self.model_data.settings.mode == ModelMode.Operation:

            self.tot_emissions = cp.sum(self.totalemission_allregions)
        
            