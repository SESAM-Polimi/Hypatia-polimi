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
import itertools
import os
import pandas as pd
import numpy as np
from datetime import (
    datetime,
    timedelta
)

import logging
import random
import tempfile
import shutil
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

        results = model._solve_MO(number_solutions, path, verbosity=verbosity, solver=solver.upper(), **kwargs)
        self.check = results
        if results is not None:
            self.results = results
        
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

    # Enhancements made to the original script
    # Kianjn

    def identify_technology_order(self, new_capacity_path):
        """
        Identify the order of technologies based on their appearance in the initial run.

        This function reads a CSV file containing new capacity data for various technologies. 
        It identifies the unique technologies and assigns an order based on their appearance 
        in the file, returning a dictionary that maps each technology to its respective order.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.

        Returns:
        dict: A dictionary mapping each technology name (str) to its corresponding order number (int).
        """

        # Read the CSV file into a DataFrame
        new_capacity_df = pd.read_csv(new_capacity_path)
        
        # Identify the unique technologies present in the 'Technology' column
        technologies_order = new_capacity_df['Technology'].unique()

        # Create a dictionary where each technology is mapped to its order (starting from 1)
        technology_order_mapping = {tech: i + 1 for i, tech in enumerate(technologies_order)}

        # Return the dictionary with technology-order mappings
        return technology_order_mapping

    def find_technologies(self, new_capacity_path, technology_order_mapping):
        """
        Find and summarize technology data from the new capacity dataset.

        This function reads a CSV file containing new capacity data, aggregates the values 
        for each technology, and identifies the technology with the maximum value, as well 
        as a randomly selected technology. It ensures that "Elec_distribution" is not selected 
        as the random technology and neither are other forbidden technologies.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        technology_order_mapping (dict): A dictionary mapping technologies to their respective order.

        Returns:
        tuple: Contains the following elements:
            - max_technology (str): The technology with the highest total value.
            - max_value (float): The total value of the max_technology.
            - max_tech_order (int): The order of the max_technology based on the mapping.
            - random_technology (str): A randomly selected technology that is not forbidden.
            - random_value (float): The total value of the random_technology.
            - random_tech_order (int): The order of the random_technology based on the mapping.
            - technologies_list (list): A list of tuples, each containing the order, technology name, and total value.
        """
        
        # List of forbidden technologies that cannot be chosen as max_technology or random_technology
        forbidden_technologies = [
            "Elec_distribution", "Bio_oil_supply", "Crude_oil_supply", "NG_supply", "SFF_supply",
            "BW_supply", "Hydrogen_supply", "OP_supply", "Electricity_imports", "Oil_refinery",
            "Bio_refinery", "Transport_biofuels_blending", "BW_CHP_P", "NG_CHP_P", "Transport_mix",
            "Industry_mix", "Civil_and_agriculture_mix", "Export_mix", "Elec_transmission_distribution"
        ]

        # Read the CSV file into a DataFrame
        new_capacity_df = pd.read_csv(new_capacity_path)

        # Sum the 'Value' column for each technology across all years
        tech_sum = new_capacity_df.groupby('Technology')['Value'].sum()

        # Create a list of technologies, including their order and total value
        technologies_list = [(technology_order_mapping[tech], tech, tech_sum[tech]) for tech in tech_sum.index]

        # Filter out forbidden technologies for max technology selection
        allowed_technologies = tech_sum.drop(labels=forbidden_technologies, errors='ignore')

        # Identify the technology with the maximum total value from the allowed technologies
        max_technology = allowed_technologies.idxmax()  # This returns the technology name with the highest value
        max_value = allowed_technologies[max_technology]  # Get the value of the max_technology

        # Drop the max technology from the remaining technologies for random selection
        remaining_technologies = allowed_technologies.drop(labels=[max_technology])

        # Filter out forbidden technologies for random technology selection
        remaining_technologies = remaining_technologies.drop(labels=forbidden_technologies, errors='ignore')

        # Select a random technology from the remaining allowed technologies
        random_technology = random.choice(remaining_technologies.index.tolist())
        
        # Get the value of the selected random technology
        random_value = remaining_technologies[random_technology]

        # Find the order of the max and random technologies using the provided mapping
        max_tech_order = technology_order_mapping[max_technology]
        random_tech_order = technology_order_mapping[random_technology]

        # Return the technology details and the full list of technologies
        return max_technology, max_value, max_tech_order, random_technology, random_value, random_tech_order, technologies_list

    def find_last_year_values(self, total_capacity_path, max_technology, random_technology):
        """
        Retrieve the capacity values for the max and random technologies in the last year of the dataset.

        This function reads a CSV file containing total capacity data, identifies the last year available 
        in the dataset, and extracts the capacity values for the specified max and random technologies 
        for that year.

        Parameters:
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        max_technology (str): The technology with the maximum value in the new capacity data.
        random_technology (str): The randomly selected technology from the new capacity data.

        Returns:
        tuple: Contains the following elements:
            - last_year (int/str): The last year in the dataset.
            - max_technology_value (float): The capacity value for the max_technology in the last year.
            - random_technology_value (float): The capacity value for the random_technology in the last year.
        """
        
        # Read the total capacity CSV file into a DataFrame
        total_capacity_df = pd.read_csv(total_capacity_path)

        # Identify the last year available in the 'Datetime' column
        last_year = total_capacity_df['Datetime'].max()

        # Filter the DataFrame to only include data for the last year
        last_year_df = total_capacity_df[total_capacity_df['Datetime'] == last_year]

        # Retrieve the capacity value for the max_technology in the last year
        max_technology_value = last_year_df[last_year_df['Technology'] == max_technology]['Value'].values[0]

        # Retrieve the capacity value for the random_technology in the last year
        random_technology_value = last_year_df[last_year_df['Technology'] == random_technology]['Value'].values[0]

        # Return the last year and the capacity values for both technologies
        return last_year, max_technology_value, random_technology_value

    def calculate_total_cost(self, run_result_path):
        """
        Calculate the total cost for a given run based on the 'tech_cost.csv' file.

        This function reads a CSV file containing the cost data for various technologies and sums the values 
        to calculate the total cost for the run. If the file is not found, it attempts to generate it by 
        calling another method and raising an error if the file still doesn't exist.

        Parameters:
        run_result_path (str): Path to the directory where the run results are stored.

        Returns:
        float: The total cost for the run.
        """
        
        # Define the path to the 'tech_cost.csv' file within the results directory
        tech_cost_file_path = os.path.join(run_result_path, "tech_cost.csv")

        # Check if the 'tech_cost.csv' file exists at the specified path
        if not os.path.exists(tech_cost_file_path):
            print(f"Warning: {tech_cost_file_path} not found. Attempting to generate it.")
            
            # Attempt to generate or save the file by calling the 'to_csv' method
            self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")
            
            # If the file still doesn't exist after attempting to generate it, raise an error
            if not os.path.exists(tech_cost_file_path):
                raise FileNotFoundError(f"{tech_cost_file_path} not found after attempting to generate it. Ensure the results are saved correctly.")
        
        # Read the 'tech_cost.csv' file into a DataFrame
        tech_cost_df = pd.read_csv(tech_cost_file_path)
        
        # Calculate the total cost by summing the values in the 'Value' column of the DataFrame
        total_cost = tech_cost_df['Value'].sum()

        # Return the total cost for the run
        return total_cost

    def calculate_total_emissions(self, run_result_path):
        """
        Calculate the total CO2 emissions for a given run based on the 'emissions.csv' file.

        This function reads a CSV file containing emissions data for different technologies. It sums the 
        values in the 'Value' column to calculate the total CO2 emissions for the run. If the file is missing, 
        it returns 0.0 and prints a warning.

        Parameters:
        run_result_path (str): Path to the directory where the run results are stored.

        Returns:
        float: The total CO2 emissions for the run.
        """
        
        # Define the path to the 'emissions.csv' file in the results directory
        emissions_file_path = os.path.join(run_result_path, "emissions.csv")

        # Check if the 'emissions.csv' file exists at the specified path
        if not os.path.exists(emissions_file_path):
            print(f"Warning: {emissions_file_path} not found. Unable to calculate emissions.")
            return 0.0  # Return 0.0 if the file is not found
        
        # Read the 'emissions.csv' file into a DataFrame
        emissions_df = pd.read_csv(emissions_file_path)
        
        # Calculate the total CO2 emissions by summing the values in the 'Value' column
        total_emissions = emissions_df['Value'].sum()

        # Return the total CO2 emissions for the run
        return total_emissions

    def record_run_results(self, result_path, run_data):
        """
        Record the results of each run into an Excel file.

        This function saves the details of multiple runs (stored in `run_data`) into an Excel file. 
        It uses the provided path to store the file, creates a DataFrame with the run data, and 
        writes it to an Excel file with the appropriate columns, including both cost and emissions 
        for each run.

        Parameters:
        result_path (str): Path to the directory where the results are stored.
        run_data (list): A list of tuples, where each tuple contains details of a single run. Each 
                        tuple should have the following information:
                        - Run Number
                        - Max Technology
                        - Max Tech Order
                        - Max Tech Value (new_capacity)
                        - Max Tech Value (total_capacity)
                        - Random Technology
                        - Random Tech Order
                        - Random Tech Value (new_capacity)
                        - Random Tech Value (total_capacity)
                        - Total Cost
                        - Total Emissions
        """
        
        # Define the path for the results Excel file
        results_file = os.path.join(result_path, "run_results.xlsx")

        # List of columns to store in the Excel file, including both total cost and emissions
        columns = [
            "Run Number", 
            "Max Technology", 
            "Max Tech Order", 
            "Max Tech Value (new_capacity)", 
            "Max Tech Value (total_capacity)", 
            "Random Technology", 
            "Random Tech Order", 
            "Random Tech Value (new_capacity)", 
            "Random Tech Value (total_capacity)",
            "Total Cost",
            "Total Emissions"  # New column for recording CO2 emissions
        ]
        
        # Create a DataFrame from the run data using the specified columns
        results_df = pd.DataFrame(run_data, columns=columns)

        # Save the DataFrame as an Excel file, without the index column
        results_df.to_excel(results_file, index=False)

    def modify_input_parameters_combine(self, new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping, reset_specific=False):
        """
        Modify the input parameters in the 'Min_totalcap' and 'Max_totalcap' sheets based on the results of 
        the current run and adjust capacity limits for max and random technologies.

        This function reads an Excel file containing parameter sheets, modifies the capacity limits for the 
        specified max and random technologies, and then saves the modified sheets back to the file.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the directory containing the 'parameters_reg1.xlsx' file.
        C (float): Adjustment factor to modify the capacity limits.
        max_technology (str): The technology with the maximum value in the new capacity data.
        random_technology (str): The randomly selected technology from the new capacity data.
        technology_order_mapping (dict): A dictionary mapping technologies to their order in the parameter file.
        reset_specific (bool): If True, resets specific capacity values for max and random technologies.

        Returns:
        None
        """
        
        # Define the path to the 'parameters_reg1.xlsx' file
        param_file_path = os.path.join(param_path, "parameters_reg1.xlsx")

        # Find technologies and their corresponding values and orders
        max_technology, x, max_tech_order, random_technology, u, random_tech_order, technologies_list = self.find_technologies(new_capacity_path, technology_order_mapping)
        
        # Find the values for the max and random technologies in the last year of total capacity
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        
        # Load the Excel file containing the parameters
        excel_data = pd.ExcelFile(param_file_path)

        # Read all sheets from the Excel file into a dictionary
        sheets_dict = pd.read_excel(excel_data, sheet_name=None, header=None)

        # Read the 'Min_totalcap' and 'Max_totalcap' sheets into DataFrames
        min_totalcap_df = sheets_dict['Min_totalcap'].copy()
        max_totalcap_df = sheets_dict['Max_totalcap'].copy()

        # Debugging: Print the initial state of the DataFrames
        print("Initial Min_totalcap DataFrame:")
        print(min_totalcap_df.head())
        print("Initial Max_totalcap DataFrame:")
        print(max_totalcap_df.head())

        if reset_specific:
            # If reset_specific is True, reset the capacity limits for max and random technologies only
            max_tech_col_index = max_tech_order  # Column index for max_technology (1-based index)
            random_tech_col_index = random_tech_order  # Column index for random_technology (1-based index)

            # Set the minimum capacity for max_technology and random_technology to zero
            min_totalcap_df.iloc[3:, max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, random_tech_col_index] = 0

            # Set the maximum capacity for both technologies to a large number
            max_totalcap_df.iloc[3:, max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, random_tech_col_index] = 10**10
        else:
            # Get the list of years from column A (starting from the fourth row)
            years = min_totalcap_df.iloc[3:, 0].tolist()

            # Find the column indices for max and random technologies based on their order
            max_tech_col_index = max_tech_order
            random_tech_col_index = random_tech_order

            # Loop through the years and modify the capacity values
            for i, year in enumerate(years):
                if i == len(years) - 1:  # Only modify the last year for random technology
                    new_min_value = T2 + C  # Adjust the minimum capacity for random_technology
                    min_totalcap_df.iat[i + 3, random_tech_col_index] = new_min_value

                # Adjust the maximum capacity values for both max_technology and random_technology
                new_max_value_max_tech = T1 - C
                new_max_value_random_tech = T2 + C
                max_totalcap_df.iat[i + 3, max_tech_col_index] = new_max_value_max_tech
                max_totalcap_df.iat[i + 3, random_tech_col_index] = new_max_value_random_tech

        # Debugging: Print the modified DataFrames
        print("Modified Min_totalcap DataFrame:")
        print(min_totalcap_df.iloc[3:].head())
        print("Modified Max_totalcap DataFrame:")
        print(max_totalcap_df.iloc[3:].head())

        # Update the sheets dictionary with the modified DataFrames
        sheets_dict['Min_totalcap'] = min_totalcap_df
        sheets_dict['Max_totalcap'] = max_totalcap_df

        # Save all modified sheets back to the Excel file using a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            temp_file_path = tmp.name

        # Write the updated data back to the Excel file
        with pd.ExcelWriter(temp_file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets_dict.items():
                # Write the modified sheets back to the Excel file, excluding headers
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        # Replace the original parameters file with the modified temporary file
        shutil.move(temp_file_path, param_file_path)

        # Debugging: Confirm that the file has been saved successfully
        print(f"Parameters modified and saved to {param_file_path}")

    def run_with_adjustments(self, new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver='gurobi'):
        """
        Run the model and adjust parameters if no results are obtained or if the total cost exceeds the allowable limit.

        This function runs a model using an optimization solver, adjusting input parameters if no results are obtained 
        or if the total cost exceeds the specified maximum allowable cost. It repeatedly modifies the capacity limits 
        (adjusted by parameter C) and tries again until a valid result within the allowable cost is found or it changes 
        the randomly selected technology after a set number of attempts.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the parameters Excel file.
        technology_order_mapping (dict): A dictionary mapping technology names to their order.
        maximum_allowable_cost (float): The maximum total cost allowed for a successful run.
        solver (str): Solver to be used for optimization. Default is 'gurobi'.

        Returns:
        tuple: Contains the following elements:
            - C (float): The final adjustment factor used.
            - max_technology (str): The technology with the maximum value.
            - random_technology (str): The randomly selected technology.
            - T1 (float): The total capacity of max_technology in the last year.
            - T2 (float): The total capacity of random_technology in the last year.
            - total_cost (float): The total cost of the final run.
        """
        
        # Define the path to the parameters Excel file
        param_file_path = os.path.join(param_path, "parameters_reg1.xlsx")

        # Identify max and random technologies and calculate initial values of capacities
        max_technology, x, _, random_technology, u, _, _ = self.find_technologies(new_capacity_path, technology_order_mapping)
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        
        # Set the initial value of adjustment factor C as the average of the values for max and random technologies
        initial_C = (x + u) / 2
        C = initial_C
        attempt = 0  # Counter to track the number of attempts

        # Continue adjusting and running the model until results are valid and cost is under the allowable limit
        while self.results is None or (self.results is not None and self.calculate_total_cost(param_path) > maximum_allowable_cost):
            if self.results is not None:
                # Calculate the total cost for the current run
                total_cost = self.calculate_total_cost(param_path)
                
                # Check if the total cost exceeds the allowable limit
                if total_cost > maximum_allowable_cost:
                    print(f"Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters.")
                else:
                    # Return successful run results if cost is within the limit
                    return C, max_technology, random_technology, T1, T2, total_cost

            # Modify the input parameters and re-run the model
            self.modify_input_parameters_combine(new_capacity_path, total_capacity_path, param_file_path, C, max_technology, random_technology, technology_order_mapping)
            self.read_input_data(param_path)  # Reload input data after modifying parameters
            self.run(
                solver=solver,  # Run the optimization model using the specified solver
                verbosity=True,
                force_rewrite=True  # Force re-running of the model
            )

            # Increment attempt counter and reduce C by 5% for each attempt
            attempt += 1
            C = initial_C - (0.05 * initial_C * attempt)

            # After every 10 attempts or if C becomes non-positive, change the random technology
            if attempt % 10 == 0 or C <= 0:
                remaining_technologies = [tech for tech in technology_order_mapping.keys() if tech != max_technology]
                if not remaining_technologies:
                    raise RuntimeError("No remaining allowable technologies to choose from.")
                
                # Randomly select a new technology and recalculate values
                random_technology = random.choice(remaining_technologies)
                _, _, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
                print(f"Changing random technology to {random_technology} after {attempt} attempts.")
                
                # Reset C to its initial value after changing the random technology
                C = initial_C

        # Once a successful run is found, calculate the final total cost and return all relevant data
        total_cost = self.calculate_total_cost(param_path)
        return C, max_technology, random_technology, T1, T2, total_cost

    def run_multiple_solutions(self, number_solutions, param_path, result_path, slack, solver='gurobi'):
        """
        Run the model multiple times, adjusting parameters and recording results for each run.

        This function runs the model a specified number of times, adjusting parameters between runs, 
        and records the results (including total cost and emissions) for each run. The function uses 
        a slack percentage to determine the maximum allowable cost for each run and attempts to 
        find solutions within this limit.

        Parameters:
        number_solutions (int): The total number of successful runs to complete.
        param_path (str): Path to the directory containing the parameters for the model.
        result_path (str): Path to the directory where run results will be saved.
        slack (float): Percentage to add as a buffer to the optimal cost for each run.
        solver (str): Solver to be used for optimization. Default is 'gurobi'.

        Returns:
        None
        """
        
        successful_runs = 0  # Counter for successful runs
        attempt = 0  # Counter for attempts made
        run_data = []  # List to store data from each run
        technology_order_mapping = None  # Will store the mapping of technologies to their order
        optimal_cost = None  # Store the optimal cost from the first successful run
        maximum_allowable_cost = None  # Store the maximum allowable cost for subsequent runs

        # Create a copy of the initial parameters directory for each run
        initial_param_dir = "Thesishypatia/IT/parameters_initial"
        if os.path.exists(initial_param_dir):
            shutil.rmtree(initial_param_dir)  # Remove the directory if it exists
        shutil.copytree(param_path, initial_param_dir)  # Copy the parameter directory for the initial state

        while successful_runs < number_solutions:
            # Reset parameters to initial state before each run
            if os.path.exists(param_path):
                shutil.rmtree(param_path)
            shutil.copytree(initial_param_dir, param_path)

            # Define the result path for the current run
            run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
            os.makedirs(run_result_path, exist_ok=True)

            if successful_runs == 0:
                # For the first run, read input data and run the model
                self.read_input_data(param_path)
                self.run(
                    solver=solver,  # Run the model with the specified solver
                    verbosity=True,
                    force_rewrite=True 
                )

                # Initialize paths for capacity data from the first run
                new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                # Check if the new_capacity file was created, generate it if missing
                if not os.path.exists(new_capacity_path):
                    self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                if not os.path.exists(new_capacity_path):
                    raise FileNotFoundError(f"{new_capacity_path} not found after the initial run.")

                # Identify the order of technologies based on the first run
                technology_order_mapping = self.identify_technology_order(new_capacity_path)

                # Find the initial max and random technologies for the first run
                max_technology, x, max_tech_order, random_technology, u, random_tech_order, technologies_list = self.find_technologies(new_capacity_path, technology_order_mapping)
                initial_C = (x + u) / 2  # Calculate initial adjustment factor C
                C = initial_C

                # Calculate the total cost for the first run and determine the maximum allowable cost
                optimal_cost = self.calculate_total_cost(run_result_path)
                maximum_allowable_cost = optimal_cost * (1 + slack / 100)  # Add slack percentage to the cost limit

                # Calculate the total emissions for the first run**
                total_emissions = self.calculate_total_emissions(run_result_path)

            else:
                # For subsequent runs, load data from the previous successful run
                new_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "new_capacity.csv")
                total_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "total_capacity.csv")

                if not os.path.exists(new_capacity_path):
                    raise FileNotFoundError(f"{new_capacity_path} not found for the run.")

                if attempt == 0:
                    # Adjust parameters and find max and random technologies only once per run
                    C, max_technology, random_technology, T1, T2, total_cost = self.run_with_adjustments(
                        new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                    )
                    initial_C = C  # Set initial C for subsequent adjustments
                else:
                    # Adjust C by reducing it further with each attempt
                    C = initial_C - (0.1 * initial_C * attempt)
                    max_technology, x, max_tech_order, random_technology, u, random_tech_order, technologies_list = self.find_technologies(new_capacity_path, technology_order_mapping)
                self.modify_input_parameters_combine(new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping)
                self.read_input_data(param_path)  # Ensure modified parameters are loaded
                self.run(
                    solver=solver,  # Run the model again with updated parameters
                    verbosity=True,
                    force_rewrite=True 
                )

            if self.results is not None:
                # Save the results after a successful run
                self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                # Update paths for the current successful run
                new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                # Calculate the total cost for the current run
                total_cost = self.calculate_total_cost(run_result_path)

                # **NEW: Calculate the total emissions for the current run**
                total_emissions = self.calculate_total_emissions(run_result_path)

                # If the total cost is within the allowable range, record the results
                if total_cost <= maximum_allowable_cost:
                    # Get the total capacities for the max and random technologies
                    _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(total_capacity_path, max_technology, random_technology)

                    # Recalculate the sum of new capacities for consistency
                    new_capacity_df = pd.read_csv(new_capacity_path)
                    max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                    random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()

                    # Record the results
                    run_data.append((
                        successful_runs + 1,
                        max_technology,
                        technology_order_mapping[max_technology],
                        max_tech_value_new,
                        max_tech_value_total,
                        random_technology,
                        technology_order_mapping[random_technology],
                        random_tech_value_new,
                        random_tech_value_total,
                        total_cost,  # Include total cost in the run data
                        total_emissions  # Include total emissions in the run data
                    ))

                    successful_runs += 1  # Increment successful runs count
                    attempt = 0  # Reset attempt counter for the next run

                    if successful_runs < number_solutions:
                        # Prepare for the next run
                        max_technology, x, max_tech_order, random_technology, u, random_tech_order, technologies_list = self.find_technologies(new_capacity_path, technology_order_mapping)
                        initial_C = (x + u) / 2
                        C = initial_C
                        self.read_input_data(param_path)
                        self.modify_input_parameters_combine(new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping)
                else:
                    # If the total cost exceeds the allowable limit, adjust parameters and retry
                    attempt += 1
                    print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                    C = initial_C - (0.05 * initial_C * attempt)  # Reduce C further
            else:
                # If no results are obtained, retry the run
                attempt += 1
                print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                C = initial_C - (0.05 * initial_C * attempt)  # Reduce C for retries

        # Record all run results in an Excel file
        self.record_run_results(result_path, run_data)

# End of Enhancements
# Kianjn

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
