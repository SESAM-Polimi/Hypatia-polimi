# -*- coding: utf-8 -*-
"""
Hypatia interface module. Contains the Model class with all
the methods needed to create,solve and save the results of a hypatia
model.
"""
import pandas as pd
from cvxpy import installed_solvers
from hypatia.error_log.Exceptions import (
    WrongInputMode,
    DataNotImported,
    ResultOverWrite,
    SolverNotFound,
)
from hypatia.utility.excel import (
    read_settings,
    write_parameters_files,
    read_parameters,
)
from hypatia.utility.constants import ModelMode
from hypatia.utility.constants import OptimizationMode
from hypatia.utility.constants import EnsureFeasibility 
from hypatia.backend.Build import BuildModel
from copy import deepcopy
from hypatia.postprocessing.PostProcessingList import POSTPROCESSING_MODULES
import os
import shutil
import logging

logger = logging.getLogger(__name__)


def _prepare_results_folder(path, force_rewrite):
    """Create ``path`` (and its parents), or empty it of an earlier run's results.

    Only a folder whose files are all CSV files is emptied: a wrong path such
    as the project folder stops with an error instead of being deleted.
    """
    if not os.path.isdir(path) or not os.listdir(path):
        os.makedirs(path, exist_ok=True)
        return
    if not force_rewrite:
        raise ResultOverWrite(
            f"Folder {path} already exists and is not empty. To replace its "
            f"results, use force_rewrite=True."
        )
    other_files = [
        os.path.relpath(os.path.join(root, name), path)
        for root, _, names in os.walk(path)
        for name in names
        if not name.lower().endswith(".csv")
    ]
    if other_files:
        raise ResultOverWrite(
            f"Folder {path} contains files that are not results of Hypatia "
            f"({other_files[:5]}{' ...' if len(other_files) > 5 else ''}). "
            f"It was not emptied; choose another folder for the results."
        )
    shutil.rmtree(path)
    os.makedirs(path)


class Model:

    """
    A Hypatia Model
    """

    def __init__(self, path, mode, optimization, ensure_feasibility, name="unknown", optional_constraints=None):

        """Initializes a Hypatia model by passing the optimization mode and
        the path of the structural input files

        Parameters
        ----------
        path : str
            path defines the directory where model set files are located.
            model sets are including regions, modelling years, timeslices,
            technologies, carriers and regional mappings.

        mode : str
            Defines the mode of the model. Acceptable values are :

                * 'Planning'
                * 'Operation'
                
        ensure_feasibility: str
            Allows for unmet demand ensuring feasibility of the model. Acceptable values are:
                
                * 'Yes'
                * 'No'
                
        name : str (Optional)
            Defines the name of the model.

        optional_constraints : list of str (Optional)
            Optional constraints to add to the model, by name. By default none
            is active. Accepted names (case-insensitive):

                * 'chp_operating_range'
                * 'production_ramp'
                * 'min_tech_power_output'
                * 'electrolysis_consumption'
                * 'boolean_for_storage'

            Enabling a constraint also adds the parameter sheets it needs to
            the files written by create_data_excels and read by
            read_input_data, so set it before creating the parameter files.
            Example: Model(..., optional_constraints=["chp_operating_range"])
        """

        assert mode in ["Planning", "Operation"], "Invalid Operation"
        assert optimization in ["Multi", "Single", "MinEm"], "Invalid Optimization Mode"
        assert ensure_feasibility in ["Yes", "No"], "Invalid Input"
        model_mode = ModelMode.Planning if mode == "Planning" else ModelMode.Operation
        optimization_mode = OptimizationMode.Multi if optimization == "Multi" else  OptimizationMode.Single if optimization == "Single" else OptimizationMode.MinEm
        ensurefeasibility = EnsureFeasibility.Yes if ensure_feasibility == "Yes" else EnsureFeasibility.No
        self.results = None
        self.backup_results = None
        self.__settings = read_settings(
            path=path,
            mode=model_mode,
            optimization=optimization_mode,
            ensure_feasibility=ensurefeasibility,
            optional_constraints=optional_constraints,
        )
        self.__model_data = None
        self.name = name

    @property
    def settings(self):
        """Model settings read from the sets folder (read-only).

        Includes ``optional_constraints``, the optional constraints active in
        this model.
        """
        return self.__settings

    def create_data_excels(self, path, force_rewrite=False):

        """Writes the parameter excel files with the default values and
        required indexing by passing an arbitary path

        .. note::

            The input data files are including regional parameter files and
            global and connections parameter files in case of multi-node model

        Parameters
        ----------
        path : str
            path defines the directory where model parameter files are going
            to be witten. It can be different from the path where set files
            are located

        force_rewrite : boolean
            to avoid over writing the parameters, this will stop the code if
            the same file already exists. In case, you need to over-write,
            force_rewrite = True will do it
        """
        write_parameters_files(self.__settings, path, force_rewrite=force_rewrite)

    def read_input_data(self, path):

        """Reades the filled input data excel files by passing the path
        where they are located

        Parameters
        -------
        path : str
            path defines the directory where the filled input parameter files
            are located. It can be different from the path where the parameter
            files with default values were written

        """

        self.__model_data = read_parameters(self.__settings, path)

    def run(self, solver, verbosity=True, force_rewrite=False, **kwargs):

        """
        Run the model by passing the solver, verbosity and force_rewrite.

        .. note::

            The passed solver must be in the installed solvers package of the DSL
            (CVXPY).

        Parameters
        ---------
        solver : str
            Solver indicates for kind of solver to be used.

        verbosity : Boolean
            Verbosity overrides the default of hiding solver output

        force_rewrite : boolean
            If the force_rewrite is True, any existing results will
            be overwritten and the previous results will be saved
            in a back-up file.

        kwargs : Optional
            solver specific options. for more information refer to `cvxpy documentation <https://www.cvxpy.org/api_reference/cvxpy.problems.html?highlight=solve#cvxpy.problems.problem.Problem.solve>`_

        """

        # checks if the input parameters are imported to the model
        if self.__model_data == None:

            raise DataNotImported(
                "No data is imported to the model. Use " "'read_input_data' function."
            )

        # checks if the model is already solved when force_rewrite is false
        # and takes a backup of previous results if force_rewrite is true
        if self.results != None:

            if not force_rewrite:
                raise ResultOverWrite(
                    "Model is already solved."
                    "To overwrite the results change "
                    "'force_rewrite'= True"
                )

            self.backup_results = deepcopy(self.results)

            self.results = None

        # checks if the given solver is in the installed solver package
        if solver.upper() not in installed_solvers():

            raise SolverNotFound(
                f"Installed solvers on your system are {installed_solvers()}"
            )

        model = BuildModel(model_data=self.__model_data)
        self.constr_backup = model.constr

        results = model._solve(verbosity=verbosity, solver=solver.upper(), **kwargs)
        self.check = results
        if results is not None:
            self.results = results
            

    def run_MinEm(self, solver, verbosity=True, force_rewrite=False, **kwargs):

        """
        Run the model by passing the solver, verbosity and force_rewrite.

        .. note::

            The passed solver must be in the installed solvers package of the DSL
            (CVXPY).

        Parameters
        ---------
        solver : str
            Solver indicates for kind of solver to be used.

        verbosity : Boolean
            Verbosity overrides the default of hiding solver output

        force_rewrite : boolean
            If the force_rewrite is True, any existing results will
            be overwritten and the previous results will be saved
            in a back-up file.

        kwargs : Optional
            solver specific options. for more information refer to `cvxpy documentation <https://www.cvxpy.org/api_reference/cvxpy.problems.html?highlight=solve#cvxpy.problems.problem.Problem.solve>`_

        """

        # checks if the input parameters are imported to the model
        if self.__model_data == None:

            raise DataNotImported(
                "No data is imported to the model. Use " "'read_input_data' function."
            )

        # checks if the model is already solved when force_rewrite is false
        # and takes a backup of previous results if force_rewrite is true
        if self.results != None:

            if not force_rewrite:
                raise ResultOverWrite(
                    "Model is already solved."
                    "To overwrite the results change "
                    "'force_rewrite'= True"
                )

            self.backup_results = deepcopy(self.results)

            self.results = None

        if self.__settings.optimization == OptimizationMode.Single:
            raise ValueError(
                "run_MinEm needs a model created with optimization='MinEm' or 'Multi' "
                "(this one has 'Single'). Use run() for a least-cost solution."
            )

        # checks if the given solver is in the installed solver package
        if solver.upper() not in installed_solvers():

            raise SolverNotFound(
                f"Installed solvers on your system are {installed_solvers()}"
            )

        model = BuildModel(model_data=self.__model_data)
        self.constr_backup = model.constr

        results = model._solve_MinEm(verbosity=verbosity, solver=solver.upper(), **kwargs)
        self.check = results
        if results is not None:
            self.results = results

    def run_MO(self, solver, number_solutions, path, verbosity=True, force_rewrite=False, **kwargs):
    
        """
        Run the model by passing the solver, verbosity and force_rewrite.
    
        .. note::
    
            The passed solver must be in the installed solvers package of the DSL
            (CVXPY).
    
        Parameters
        ---------
        solver : str
            Solver indicates for kind of solver to be used.
    
        verbosity : Boolean
            Verbosity overrides the default of hiding solver output
    
        force_rewrite : boolean
            If the force_rewrite is True, any existing results will
            be overwritten and the previous results will be saved
            in a back-up file.
    
        kwargs : Optional
            solver specific options. for more information refer to `cvxpy documentation <https://www.cvxpy.org/api_reference/cvxpy.problems.html?highlight=solve#cvxpy.problems.problem.Problem.solve>`_
    
        """
    
        # checks if the input parameters are imported to the model
        if self.__model_data == None:
            raise DataNotImported(
                "No data is imported to the model. Use 'read_input_data' function."
            )

        # checked before any solving, so that a mistake costs no solver time
        if self.__settings.optimization != OptimizationMode.Multi:
            raise ValueError(
                f"run_MO needs a model created with optimization='Multi' "
                f"(this one has '{self.__settings.optimization.value}'). Use run() "
                f"for 'Single' and run_MinEm() for 'MinEm'."
            )
        if isinstance(number_solutions, bool) or not isinstance(number_solutions, int) \
                or number_solutions < 2:
            raise ValueError(
                f"number_solutions must be an integer of at least 2 (the least-cost "
                f"and the least-emission solutions), not {number_solutions!r}."
            )
    
        # checks if the given solver is in the installed solver package
        if solver.upper() not in installed_solvers():
            raise SolverNotFound(
                f"Installed solvers on your system are {installed_solvers()}"
            )
    
        model = BuildModel(model_data=self.__model_data)
        self.constr_backup = model.constr        
    
        # Generate emission list using solve_MO
        emission_list = model._solve_MO(number_solutions, path, verbosity=verbosity, solver=solver.upper(), **kwargs)
        
        # Iterate through each emission value and solve
        for i, emission_value in enumerate(emission_list):
            print(f"Processing solution {i + 1}/{len(emission_list)} with emission value: {emission_value}")
            results = model._solve_with_emission(emission_value, verbosity=verbosity, solver=solver.upper(), **kwargs)
        
            if results is not None:
                self.results = results  # Store the latest valid result
        
                # Create folder for the solution
                solution_folder = os.path.join(path, f"{i + 1}")
                if not os.path.exists(solution_folder):
                    os.makedirs(solution_folder)
        
                # Save solution to CSV
                self.to_csv(path=solution_folder, postprocessing_module="aggregated", force_rewrite=True)
                print(f"Solution {i + 1} saved to {solution_folder}")
            else:
                print(f"No solution for emission value {emission_value}")
        
        print(f"All {number_solutions} solutions have been processed and saved.")

    def to_csv(self, path, postprocessing_module="default", force_rewrite=False):
        """Exports the results of the model to csv files with nested folders

        Parameters
        ----------
        path : str
            Folder where the results are written. Missing parent folders are
            created.
        postprocessing_module : str
            "default" (one folder per quantity, wide tables) or "aggregated"
            (one long table per quantity).
        force_rewrite : boolean
            If the folder exists and is not empty: False stops with an error;
            True deletes the previous results first, provided the folder holds
            only CSV files (in any subfolder). A folder with other files is
            never emptied, to protect a mistyped path.
        """

        if self.results == None:
            raise WrongInputMode("model has not any results")

        if postprocessing_module not in POSTPROCESSING_MODULES:
            raise ValueError(
                f"Unknown post-processing module '{postprocessing_module}'. "
                f"Available: {list(POSTPROCESSING_MODULES)}"
            )
        _prepare_results_folder(path, force_rewrite)

        POSTPROCESSING_MODULES[postprocessing_module](
            self.__model_data,
            self.results
        ).write_processed_results(path)

    def __str__(self):
        to_print = (
            "name = {}\n"
            "mode = {}\n"
            "regions= {}\n"
            "techs= {}\n"
            "horizon= {}\n"
            "resolution= {}\n"
            "optional constraints= {}\n".format(
                self.name,
                self.settings.mode,
                self.settings.regions,
                self.settings.technologies,
                self.settings.years,
                len(self.settings.time_steps),
                list(self.settings.optional_constraints) or "none",
            )
        )

        return to_print

    ##### RESTORED ###########
    def create_config_file(self, path):
     """Creates a config excel file for plots

     Parameters
     ----------
     path : str
         defines the path and the name of the excel file to be created.
     """

     techs_property = {"tech_name": list(self.__settings.global_settings["Technologies_glob"]["Tech_name"]),
             "tech_group": '',
             "tech_color": '',
             "tech_cap_unit": list(self.__settings.global_settings["Technologies_glob"]["Tech_cap_unit"]),
             "tech_production_unit": list(self.__settings.global_settings["Technologies_glob"]["Tech_act_unit"]),}
     
     techs_sheet = pd.DataFrame(techs_property,
         index=self.__settings.global_settings["Technologies_glob"]["Technology"],
     )
     
     imports=[]
     exports=[]
     imports_name=[]
     exports_name=[]
     for step_index, region in enumerate(self.__settings.regions):            
         for step_indexx, regions in enumerate(self.__settings.regions):
             if(regions == region):
                 continue
             imports.append("Import to " + region + " from " + regions)
             exports.append("Export from " + region + " to " + regions)
             imports_name.append("Import from " + regions)
             exports_name.append("Export to " + regions)
         
     import_export = {"line_name": imports_name + exports_name,
         "line_color": '' }
     
     importexport_sheet = pd.DataFrame(import_export,
         index = imports + exports
     )
     

     fuels_property = {"fuel_name": list(self.__settings.global_settings["Carriers_glob"]["Carr_name"]),
             "fuel_group": '',
             "fuel_color": '',
             "fuel_unit": list(self.__settings.global_settings["Carriers_glob"]["Carr_unit"]),}

     fuels_sheet = pd.DataFrame(fuels_property,
         index=self.__settings.global_settings["Carriers_glob"]["Carrier"],
     )

     regions_property = {"region_name": list(self.__settings.global_settings["Regions"]["Region_name"]),
             "region_color": '',}

     regions_sheet = pd.DataFrame(regions_property,
         index=self.__settings.global_settings["Regions"]["Region"],
     )

     emissions_sheet = self.__settings.global_settings['Emissions'].set_index(['Emission'],inplace=False)
     emissions_sheet = pd.DataFrame(
         emissions_sheet.values,
         index = emissions_sheet.index,
         columns = ['emission_name','emission_unit']
     )
     emissions_sheet.index.name = 'Emission'

     if self.__settings.multi_node:
         with pd.ExcelWriter(path) as file:
             for sheet in [
                 "techs_sheet",
                 "importexport_sheet",
                 "fuels_sheet",
                 "regions_sheet",
                 "emissions_sheet",
             ]:
                 eval(sheet).to_excel(file, sheet_name=sheet.split("_")[0].title())
     else:
         with pd.ExcelWriter(path) as file:
             for sheet in [
                 "techs_sheet",
                 "fuels_sheet",
                 "regions_sheet",
                 "emissions_sheet",
             ]:
                 eval(sheet).to_excel(file, sheet_name=sheet.split("_")[0].title())

    ##################
    
    def get_model_data(self):
        return self.__model_data

    def __repr__(self):
        return self.__str__()
