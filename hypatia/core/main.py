# -*- coding: utf-8 -*-
"""
Hypatia interface module. Contains the Model class with all
the methods needed to create,solve and save the results of a hypatia
model.
"""
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
import logging

logger = logging.getLogger(__name__)


class Model:

    """
    A Hypatia Model
    """

    def __init__(self, path, mode, optimization, ensure_feasibility, name="unknown"):

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
        """

        assert mode in ["Planning", "Operation"], "Invalid Operation"
        assert optimization in ["Multi", "Single", "MinEm"], "Invalid Optimization Mode"
        assert ensure_feasibility in ["Yes", "No"], "Invalid Input"
        model_mode = ModelMode.Planning if mode == "Planning" else ModelMode.Operation
        optimization_mode = OptimizationMode.Multi if optimization == "Multi" else  OptimizationMode.Single if optimization == "Single" else OptimizationMode.MinEm
        ensurefeasibility = EnsureFeasibility.Yes if ensure_feasibility == "Yes" else EnsureFeasibility.No
        self.results = None
        self.backup_results = None
        self.__settings = read_settings(path=path, mode=model_mode, optimization=optimization_mode, ensure_feasibility=ensurefeasibility)
        self.__model_data = None
        self.name = name

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
            self.results = None
    
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
            Defines the path to th 'folder' which all the results will be
            created.
        force_rewrite : boolean
            if False, will stop the code in case the file already exists,
            if True, will delete the file if alreadey exists and create a new one
        """

        if self.results == None:
            raise WrongInputMode("model has not any results")

        
        self.__model_data.settings
    
        if os.path.exists(path):
            if not force_rewrite:
                raise ResultOverWrite(
                    f"Folder {path} already exists. To over write"
                    f" the results, use force_rewrite=True."
                )
        else:
            os.mkdir(path)

        if postprocessing_module in POSTPROCESSING_MODULES.keys():
            POSTPROCESSING_MODULES[postprocessing_module](
                self.__model_data,
                self.results
            ).write_processed_results(path)
        else:
            raise Exception("Post processing module do not exist") 

    def __str__(self):
        to_print = (
            "name = {}\n"
            "mode = {}\n"
            "regions= {}\n"
            "techs= {}\n"
            "horizon= {}\n"
            "resolution= {}\n".format(
                self.name,
                self.settings.mode,
                self.settings.regions,
                self.settings.technologies,
                self.settings.years,
                len(self.settings.time_steps),
            )
        )

        return to_print

    def get_model_data(self):
        return self.__model_data

    def __repr__(self):
        return self.__str__()
