# -*- coding: utf-8 -*-
"""
This module contains the core of the optimization model, containing
the definiton of problem (variables,constraints,objective function,...)
in CVXPY for planning and operation modes.
"""
import warnings

import cvxpy as cp
import numpy as np
from collections import namedtuple
from hypatia.error_log.Exceptions import NoSolution
from hypatia.backend.ModelData import ModelData
from hypatia.backend.ModelVariables import ModelVariables
from hypatia.utility.constants import ModelMode, OptimizationMode
from hypatia.utility.utility import (
    get_emission_types
)
import logging

logger = logging.getLogger(__name__)


RESULTS = [
    "technology_prod",
    "production_annual",
    "technology_use",
    "consumption_annual",
    "unmet_demand_annual",
    "cost_fix",
    "cost_variable",
    "totalcapacity",
    "cost_fix_tax",
    "cost_fix_sub",
    "cost_unmet_demand",
    "emission_cost_by_type",
    "emission_cost_by_region",
    "emission_by_type",
    "captured_emission_by_type",
    "used_emissions_by_type",
    "demand",
    "storage_SOC",
    "residual_capacity",
    "carrier_ratio_in",
    "carrier_ratio_out",
    "tot_emissions",
    "unmetdemandbycarrier"
]

PLANNING_RESULTS = [
    "cost_decom",
    "decommissioned_capacity",
    "new_capacity",
    "real_new_capacity",
    "units_online",
    "units_online_technologies",
    "cost_inv",
    "salvage_inv",
    "cost_inv_tax",
    "cost_inv_sub",
]

MULTI_NODE_RESULTS = [
    "line_import",
    "line_import_annual",
    "line_export",
    "line_export_annual",
]

# Relative slack on an emission bound (second step of run_MinEm, points of
# run_MO). With the exact minimum as bound a solver
# can report the problem infeasible, because the minimum it found satisfies its
# own tolerances but not the bound exactly. max(|.|, 1) keeps the slack positive
# and meaningful for zero or negative minimum emissions.
EMISSION_SLACK = 1e-6


def emission_bound(minimum):
    """The emission bound used in place of the exact minimum."""
    return minimum + EMISSION_SLACK * max(abs(minimum), 1.0)


def check_solution(problem, run):
    """Decide whether the solution of a solved problem can be used.

    * "optimal": yes.
    * "optimal_inaccurate": yes, with a warning: the solver met looser
      tolerances than requested.
    * "user_limit" (e.g. a time limit) with a feasible solution: yes, with a
      warning: the solution is feasible but not proven optimal (for a MIP, see
      the gap in the solver log).
    * anything else (infeasible, unbounded, no solution found): NoSolution.

    ``run`` names the run in the messages, e.g. "the least-cost run".
    """
    status = problem.status
    if status == cp.OPTIMAL:
        return
    has_solution = problem.value is not None and np.isfinite(problem.value) and all(
        v.value is not None for v in problem.variables()
    )
    if status == cp.OPTIMAL_INACCURATE and has_solution:
        warnings.warn(f"{run}: the solver reports an inaccurate optimum (status "
                      f"'{status}'); results are kept.", UserWarning, stacklevel=3)
        return
    if status == cp.USER_LIMIT and has_solution:
        warnings.warn(f"{run}: the solver stopped at a limit (status '{status}', e.g. a "
                      f"time limit) with a feasible solution, which is kept but not proven "
                      f"optimal.", UserWarning, stacklevel=3)
        return
    raise NoSolution(f"{run}: no usable solution (solver status: '{status}'). An "
                     f"infeasible model is easier to diagnose with ensure_feasibility='Yes'.")


class BuildModel:

    """Class that builds the variables and equations of the model

    Attributes
    -----------
    sets:
        The instance of the Readsets class for delivering the structural inputs
        including regions, technologies, years, timesteps, mapping tables

    variables: dict
        a nested dictionary of all the decision variables including the new capacity, production
        by each technology, use (consumption) by each technology, imports and exports

    """

    def __init__(self, model_data: ModelData):

        self.model_data = model_data

        # define all primary and secondary variables
        self.vars = ModelVariables(self.model_data)

        # create all constraints; constraints_by_class keeps them grouped by the
        # class that made them (used to inspect or test the built model)
        self.constr = []
        self.constraints_by_class = {}
        for constraint in self.model_data.settings.constraints:
            rules = constraint(self.model_data, self.vars).get()
            self.constraints_by_class[constraint.__name__] = rules
            self.constr += rules

        timeslice_fraction = self.model_data.settings.timeslice_fraction
        if not isinstance(timeslice_fraction, int):
            timeslice_fraction.shape = (len(self.model_data.settings.time_steps), 1)
        self.timeslice_fraction = timeslice_fraction

        # Total emissions are always computed: they are the objective in "MinEm"
        # and "Multi" runs, and are reported as total_emission in every mode.
        # (The previous test "optimization == Multi or MinEm" was always true.)
        self._set_regional_emission_objective()
        self._set_final_emission_objective()

        # calling the methods based on the defined mode by the user
            
        if self.model_data.settings.mode == ModelMode.Planning:
            self._set_regional_objective_planning()

            if not self.model_data.settings.multi_node:
                self._set_final_objective_singlenode()
            else:
                self._set_lines_objective_planning()
                self._set_final_objective_multinode()

        elif self.model_data.settings.mode == ModelMode.Operation:
            self._set_regional_objective_operation()

            if not self.model_data.settings.multi_node:
                self._set_final_objective_singlenode()
            else:
                self._set_lines_objective_operation()
                self._set_final_objective_multinode()
                
        self.NPC = self.global_objective
        self.total_emission = self.global_emission_objective
                    

    def _results(self):
        """The result expressions of this model, as a named tuple."""
        names = RESULTS.copy()
        if self.model_data.settings.multi_node:
            if self.model_data.settings.mode == ModelMode.Planning:
                names += [
                    "line_totalcapacity",
                    "line_new_capacity",
                    "real_new_line_capacity",
                    "line_decommissioned_capacity",
                    "cost_inv_line",
                    "salvage_inv_line",
                    "cost_fix_line",
                    "cost_decom_line",
                    "cost_variable_line",
                    "tot_cost_multi_node",
                ]
            else:
                names += [
                    "line_totalcapacity",
                    "cost_fix_line",
                    "cost_variable_line",
                    "tot_cost_multi_node",
                ]
        else:
            names += ["tot_cost_single_node"]
        if self.model_data.settings.mode == ModelMode.Planning:
            names += PLANNING_RESULTS
        if self.model_data.settings.multi_node:
            names += MULTI_NODE_RESULTS
        result_collector = namedtuple("result", names)
        return result_collector(**{name: getattr(self.vars, name) for name in names})

    def _minimise(self, objective, constraints, run, verbosity, solver, **kwargs):
        """Solve min objective s.t. constraints; raise NoSolution if unusable."""
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=solver, verbose=verbosity, **kwargs)
        check_solution(problem, run)
        return problem

    def _solve(self, verbosity, solver, **kwargs):
        """Minimise the total cost; return the results."""
        self._minimise(self.global_objective, self.constr, "The model",
                       verbosity, solver, **kwargs)
        return self._results()

    def _solve_MO(self, number_solutions, path, verbosity, solver, **kwargs):
        """Least-cost and least-emission runs; return the emission bounds of the
        number_solutions points to solve, evenly spaced from the least-cost
        emissions down to the minimum, each plus EMISSION_SLACK."""
        if self.model_data.settings.optimization != OptimizationMode.Multi:
            raise ValueError(
                "run_MO needs a model created with optimization='Multi'."
            )

        print("\n ----- NEW RUN - Least Cost ------ \n")
        self._minimise(self.global_objective, self.constr, "run_MO: the least-cost run",
                       verbosity, solver, **kwargs)
        max_emissions = float(np.ravel(self.vars.tot_emissions.value)[0])

        print("\n ----- NEW RUN - Least Emissions ----- \n")
        self._minimise(self.global_emission_objective, self.constr,
                       "run_MO: the least-emission run", verbosity, solver, **kwargs)
        min_emissions = float(np.ravel(self.global_emission_objective.value)[0])

        # Every bound gets the slack: the first one equals the emissions of the
        # least-cost solution and the last one the minimum, and both can be
        # reported infeasible as exact bounds (on Atlantis they coincide).
        step = (max_emissions - min_emissions) / (number_solutions - 1)
        return [emission_bound(max_emissions - i * step) for i in range(number_solutions)]

    def _solve_with_emission(self, emission_value, verbosity, solver, **kwargs):
        """Least cost with total emissions <= emission_value.

        Returns the results, or None (with a warning) when this point has no
        usable solution, so that the other points of run_MO are still saved.
        """
        print("\n ------ NEW RUN Solution ------ \n")
        try:
            self._minimise(self.global_objective,
                           self.constr + [self.vars.tot_emissions <= emission_value],
                           f"run_MO: the point with emissions <= {emission_value:g}",
                           verbosity, solver, **kwargs)
        except NoSolution as error:
            warnings.warn(str(error), UserWarning, stacklevel=2)
            return None
        return self._results()

    def _solve_MinEm(self, verbosity, solver, **kwargs):
        """Minimise the emissions, then the cost among the solutions whose
        emissions are within EMISSION_SLACK of the minimum."""
        print("\n ------------------- NEW RUN ------------------- \n")

        if self.model_data.settings.optimization == OptimizationMode.Single:
            raise ValueError(
                "run_MinEm needs a model created with optimization='MinEm' or 'Multi'."
            )

        self._minimise(self.global_emission_objective, self.constr,
                       "run_MinEm: the least-emission run", verbosity, solver, **kwargs)
        min_emissions = float(np.ravel(self.global_emission_objective.value)[0])
        print(f"Minimum cumulative emissions value: {min_emissions}")

        self._minimise(self.global_objective,
                       self.constr + [self.vars.tot_emissions <= emission_bound(min_emissions)],
                       "run_MinEm: the least-cost run at minimum emissions",
                       verbosity, solver, **kwargs)
        return self._results()

    def _set_regional_objective_planning(self):

        """
        Calculates the regional objective function in the planning mode
        """

        self.totalcost_allregions = np.zeros((len(self.model_data.settings.years), 1))
        self.inv_allregions = 0
        years = -1 * np.arange(len(self.model_data.settings.years))

        for reg in self.model_data.settings.regions:

            totalcost_regional = np.zeros((len(self.model_data.settings.years), 1))

            for ctgry in self.model_data.settings.technologies[reg].keys():

                if ctgry != "Demand":

                    totalcost_regional = totalcost_regional + cp.sum(
                        self.vars.cost_inv_tax[reg][ctgry]
                        - self.vars.cost_inv_sub[reg][ctgry]
                        + self.vars.cost_fix[reg][ctgry]
                        + self.vars.cost_fix_tax[reg][ctgry]
                        - self.vars.cost_fix_sub[reg][ctgry]
                        + self.vars.cost_variable[reg][ctgry]
                        + self.vars.cost_decom[reg][ctgry]
                        - self.vars.salvage_inv[reg][ctgry],
                        axis=1,
                    )

                    self.inv_allregions = self.inv_allregions + self.vars.cost_inv_fvalue[reg][ctgry]

                    if ctgry != "Transmission" and ctgry != "Storage":
                        for emission_type in get_emission_types(self.model_data.settings.global_settings):
                            totalcost_regional = totalcost_regional + cp.sum(
                                self.vars.emission_cost_by_region[reg][emission_type][ctgry], axis=1
                            )
            for carr in self.vars.unmetdemandbycarrier[reg].keys():
                            
                totalcost_regional = totalcost_regional + cp.sum(self.vars.cost_unmet_demand[reg][carr],axis=1)                

            discount_factor = (
                1 + self.model_data.regional_parameters[reg]["discount_rate"]["Annual Discount Rate"].values
            )

            totalcost_regional_discounted = cp.multiply(
                totalcost_regional, np.power(discount_factor, years)
            )
            self.totalcost_allregions = self.totalcost_allregions + totalcost_regional_discounted
            
    def _set_regional_emission_objective(self):
        """
        Calculates the regional emission objective function in the planning mode
        """
        self.totalemission_allregions = np.zeros((len(self.model_data.settings.years), 1))
        
        for reg in self.model_data.settings.regions:
            
            totalemission_regional = np.zeros((len(self.model_data.settings.years), 1))
            
            for emission_type in get_emission_types(self.model_data.settings.global_settings):
                
                totalemission_regional_by_type = np.zeros((len(self.model_data.settings.years), 1))
                
                for ctgry in self.model_data.settings.technologies[reg].keys():
                    
                    # if ctgry != "Demand" and ctgry != "Transmission":
                    if ctgry != "Demand" and ctgry != "Transmission" and ctgry != "Storage":
                        
                        totalemission_regional_by_type = totalemission_regional_by_type + cp.sum(
                            self.vars.emission_by_region[reg][emission_type][ctgry], axis=1
                        )
                        
                totalemission_regional = totalemission_regional + totalemission_regional_by_type
                
            self.totalemission_allregions = self.totalemission_allregions + totalemission_regional
            
    def _set_regional_objective_operation(self):

        """
        Calculates the regional objective function in the operation mode
        """
        self.totalcost_allregions = 0
        for reg in self.model_data.settings.regions:

            totalcost_regional = 0

            for ctgry in self.model_data.settings.technologies[reg].keys():

                if ctgry != "Demand":

                    totalcost_regional = totalcost_regional + cp.sum(
                        self.vars.cost_fix[reg][ctgry]
                        + self.vars.cost_fix_tax[reg][ctgry]
                        - self.vars.cost_fix_sub[reg][ctgry]
                        + self.vars.cost_variable[reg][ctgry]
                    )

                    if ctgry != "Transmission" and ctgry != "Storage":
                        for emission_type in get_emission_types(self.model_data.settings.global_settings):
                            totalcost_regional = totalcost_regional + cp.sum(
                                self.vars.emission_cost_by_region[reg][emission_type][ctgry], axis=1
                            )
                            
            for carr in self.vars.unmetdemandbycarrier[reg].keys():
                            
                totalcost_regional = totalcost_regional + cp.sum(self.vars.cost_unmet_demand[reg][carr],axis=1) 

            self.totalcost_allregions = self.totalcost_allregions + totalcost_regional

    def _set_lines_objective_planning(self):

        """
        Calculates the objective function of the inter-regional links in the
        planning mode
        """

        years = -1 * np.arange(len(self.model_data.settings.years))
        self.totalcost_lines = np.zeros((len(self.model_data.settings.years), 1))

        # Investments enter as annuities (inv_lines, already discounted) and
        # the salvage value is subtracted, as for technologies.
        self.inv_lines = 0
        for line in self.model_data.settings.lines_list:

            self.totalcost_lines = self.totalcost_lines + cp.sum(
                self.vars.cost_fix_line[line]
                + self.vars.cost_decom_line[line]
                - self.vars.salvage_inv_line[line],
                axis=1,
            )
            self.inv_lines = self.inv_lines + self.vars.cost_inv_line_fvalue[line]

        for reg in self.model_data.settings.regions:

            for key, value in self.vars.cost_variable_line[reg].items():

                self.totalcost_lines = self.totalcost_lines + cp.sum(value, axis=1)

        discount_factor_global = (
            1
            + self.model_data.global_parameters["global_discount_rate"][
                "Annual Discount Rate"
            ].values
        )

        self.totalcost_lines_discounted = cp.multiply(
            self.totalcost_lines, np.power(discount_factor_global, years)
        )

    def _set_lines_objective_operation(self):

        """
        Calculates the objective function of the inter-regional links in the
        operation mode
        """

        self.totalcost_lines = 0

        for line in self.model_data.settings.lines_list:

            self.totalcost_lines = self.totalcost_lines + cp.sum(self.vars.cost_fix_line[line], axis=1)

        for reg in self.model_data.settings.regions:

            for key, value in self.vars.cost_variable_line[reg].items():

                self.totalcost_lines = self.totalcost_lines + cp.sum(value, axis=1)

    def _set_final_objective_singlenode(self):

        """
        Calculates the overall objective function in a single-node model
        """
        if self.model_data.settings.mode == ModelMode.Planning:
            self.global_objective = (
                cp.sum(self.totalcost_allregions) + self.inv_allregions
            )

        elif self.model_data.settings.mode == ModelMode.Operation:

            self.global_objective = self.totalcost_allregions

    def _set_final_objective_multinode(self):

        """
        Calculates the overall objective function as the summation of all the
        regional and inter-regional links objective functions in a multi-node
        model
        """

        if self.model_data.settings.mode == ModelMode.Planning:

            self.global_objective = (
                cp.sum(self.totalcost_lines_discounted + self.totalcost_allregions)
                + self.inv_allregions
                + self.inv_lines
            )

        elif self.model_data.settings.mode == ModelMode.Operation:

            self.global_objective = self.totalcost_allregions + self.totalcost_lines
            
    def _set_final_emission_objective(self):

        """
        Calculates the overall emission objective function 
        """
        if self.model_data.settings.mode == ModelMode.Planning:
            self.global_emission_objective = cp.sum(self.totalemission_allregions)

        elif self.model_data.settings.mode == ModelMode.Operation:

            self.global_emission_objective = cp.sum(self.totalemission_allregions)
