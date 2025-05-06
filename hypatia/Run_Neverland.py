"""
Neverland MODEL
Planning mode
"""
#%%
# Import of the Model

from hypatia import Model
import os
from hypatia import Plotter

#%% 
# Create the model using as input the sets files
OptimizationMode = "Single"                                             # "Single" or "MinEm" or "Multi" objective optimization. Single minimizes NPC, MinEm minimizes Emissions, Multi Objective minimizes NPC and CO2
Number_solutions = 3                                                    # Number of required solution in case of multi-objective optimization
Ensure_Feasibility = "No"                                               # "Yes" allows unmet demand, "No" otherwise                                               

Neverland = Model(
    path="examples/Neverland/sets",                             # Path to the sets folder
    mode="Planning",                                                    # "Planning" or "Operation" mode
    optimization = OptimizationMode,
    ensure_feasibility = Ensure_Feasibility                                     
)

#%% 
# Create the parameters with default values

# Neverland.create_data_excels(
#     path ='examples/Neverland/parameters_empty',                      # Path to the parameters folder
#     force_rewrite=True                                                  # Overwrite the parameters files (True) or not (False)
# )

#%% 
# Read the parameters

Neverland.read_input_data("examples/Neverland/parameters")         # Path to the parameters folder

#%% 
# Run the model to find the optimal solution

if OptimizationMode == "Multi":    
    Neverland.run_MO(
        solver='gurobi',                                                    # Selection of the solver: 'GUROBI', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS’
        number_solutions = Number_solutions,
        path = "examples/Neverland/Pareto Froniter",                 # Path to the destination folder where all the results of the solutions are saved
        verbosity=True,
        force_rewrite= True                                                 # Overwrite the parameters files (True) or not (False)
    )
elif OptimizationMode == "Single":
    Neverland.run(
        solver='gurobi',                                                    # Selection of the solver: 'GUROBI', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS’
        verbosity=True,
        force_rewrite= True                                                 # Overwrite the parameters files (True) or not (False)
    )
else:
    Neverland.run_MinEm(
        solver='gurobi',                                                    # Selection of the solver: 'GUROBI', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS’
        verbosity=True,
        force_rewrite= True                                                 # Overwrite the parameters files (True) or not (False)
    )
    
#%%
# Create results folder for Single Objective Runs 
    
if not os.path.exists("examples/Neverland/results"):
    os.mkdir("examples/Neverland/results")
    
#%%
# Save the results as csv file in the previous folder

Neverland.to_csv(
    path= "examples/Neverland/results",                         # Path to the destination folder for the results
    force_rewrite=True,                                                 # Overwrite the parameters files (True) or not (False)
    postprocessing_module="aggregated"                                  # "default" and "aggregated" are the two options
)


#%% 
# Create the configuration file for the plots

# Neverland.create_config_file(
#     path = 'examples/Neverland/config.xlsx'                     # Path to the config file
# )

    
#%%
# Create plots folder    
    
if not os.path.exists("examples/Neverland/plots"):
    os.mkdir("examples/Neverland/plots")
    
#%% 
# Read the configuration file

plots = Plotter(
    results = Neverland,                                                   # Name of the Model
    config = 'examples/Neverland/config.xlsx',                  # Path to the config file
    hourly_resolution = True,                                           # if model has an hourly resultion otherwise False
)

#%% 
# Plot the total capacity of each technology in the tech_group in each year and save it in the plots folder 
    
plots.plot_total_capacity(
    path = "examples/Neverland/plots/totalcapacity.html",       # Path to the folder in which the plot will be saved
    tech_group = 'Power Generation',                                    # The group of the techs, reported in the configuration file, to be plotted
    kind= "bar",                                                        # "Bar" or "Area" are the two kind of plots accepted
    decom_cap=True,                                                     # Decommissioning capacity can be included (True) or not (False)
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                     # True to aggregate the results of each region, False to plot them separately
)

#%% 
# Plot the new capacity of each technology in the tech_group in each year and save it in the plots folder 

plots.plot_new_capacity(
    path = "examples/Neverland/plots/newcapacity.html",         # Path to the folder in which the plot will be saved
    tech_group = 'Power Generation',                                    # The group of the techs, reported in the configuration file, to be plotted
    kind="bar",                                                         # "Bar" or "Area" are the two kind of plots accepted
    cummulative=False,                                                  # In each year the cummulative new capacity is plotted (True) or not (False)
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                     # True to aggregate the results of each region, False to plot them separately
)

#%% 
# Plot the annual production of each technology in the tech_group in each year and save it in the plots folder 

plots.plot_prod_by_tech(
    path = "examples/Neverland/plots/prod_by_tech.html",        # Path to the folder in which the plot will be saved
    tech_group = 'Power Transmission',                                    # The group of the techs, reported in the configuration file, to be plotted
    kind="bar",                                                         # "Bar" or "Area" are the two kind of plots accepted
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                     # True to aggregate the results of each region, False to plot them separately
)

#%% 
# Plot the annual consumption of each carrier in the fuel_group in each year and save it in the plots folder 

plots.plot_use_by_technology(
    path = "examples/Neverland/plots/use_by_tech.html",         # Path to the folder in which the plot will be saved
    fuel_group = 'Electricity',                                                 # The group of the carriers, reported in the configuration file, to be plotted
    kind="bar",                                                         # "Bar" or "Area" are the two kind of plots accepted
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                     # True to aggregate the results of each region, False to plot them separately
)

#%%
# Plot as Pie chart the annual consumption and production of each carrier in the fuel_group for a specific year and save it in the plots folder

plots.plot_fuel_prod_cons(
    path = "examples/Neverland/plots/prod_con_share_2020.html",     # Path to the folder in which the plot will be saved
    years = ["Y0"],                                                         # Year considered 
    fuel_group = 'Electricity',                                             # The group of the carriers, reported in the configuration file, to be plotted
    trade=False,                                                             # Only in case of Multi region model trade can be included (True) or not (False)
    regions="all",                                                          # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                         # True to aggregate the results of each region, False to plot them separately
)

plots.plot_fuel_prod_cons(
    path = "examples/Neverland/plots/prod_con_share_2030.html",     # Path to the folder in which the plot will be saved
    years = ["Y9"],                                                        # Year considered 
    fuel_group = 'Electricity',                                             # The group of the carriers, reported in the configuration file, to be plotted
    trade=False,                                                             # Only in case of Multi region model trade can be included (True) or not (False)
    regions="all",                                                          # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                         # True to aggregate the results of each region, False to plot them separately
)

#%% 
# Plot the annual emission of the emission_type for each technology in the tech_group in each year and save it in the plots folder

plots.plot_emissions(
    path = "examples/Neverland/plots/emissions.html",           # Path to the folder in which the plot will be saved
    tech_group = 'Power Generation',                                    # The group of the techs, reported in the configuration file, to be plotted
    emission_type = ["CO2 emissions"],                                  # The type of the emissions, reported in the configuration file, to be plotted
    kind="bar",                                                         # "Bar" or "Area" are the two kind of plots accepted
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=True                                                     # Global emission can be plotted (True) or emission for each region (False)
)

#%%
# Plot the hourly production of the carrier in the fuel_group for each tech in the tech_group, from the start to the end time

plots.plot_hourly_prod_by_tech(
    path = "examples/Neverland/plots/hourlyprod_2020.html",     # Path to the folder in which the plot will be saved
    tech_group = 'Power Generation',                                    # The group of the techs, reported in the configuration file, to be plotted
    fuel_group = 'Electricity',                                         # The group of the carriers, reported in the configuration file, to be plotted
    kind = "bar",                                                       # "Bar" or "Area" are the two kind of plots accepted
    year = ["Y0"],                                                      # Year considered 
    start="2020-01-01 00:00:00",                                        # Starting day and time
    end="2020-01-01 23:00:00",                                          # Ending day and time
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                     # Global hourly production can be plotted (True) or emission for each region (False)
)

plots.plot_hourly_prod_by_tech(
    path = "examples/Neverland/plots/hourlyprod_2030.html",     # Path to the folder in which the plot will be saved
    tech_group = 'Power Generation',                                    # The group of the techs, reported in the configuration file, to be plotted
    fuel_group = 'Electricity',                                         # The group of the carriers, reported in the configuration file, to be plotted
    kind = "bar",                                                       # "Bar" or "Area" are the two kind of plots accepted
    year = ["Y9"],                                                     # Year considered 
    start="2030-01-01 00:00:00",                                        # Starting day and time
    end="2030-01-01 23:00:00",                                          # Ending day and time
    regions="all",                                                      # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                     # Global hourly production can be plotted (True) or emission for each region (False)
)

#%%
# Plot the annual costs in each year and save it in the plots folder

plots.plot_regional_costs(
    path = "examples/Neverland/plots/regionalcost_by_tech.html", # Path to the folder in which the plot will be saved
    stacked_by = 'techs',                                                 # Plot can be stacked by "techs" or by cost "items"
    exclude_tech_groups=[],                                               # Excluded tech groups
    exclude_cost_items=[],                                                # Excluded cost items
    regions="all",                                                        # The regions considered. "all" to consider all of them, ["reg1", ...] to consider only some regions
    aggregate=False                                                       # Global cost can be plotted (True) or emission for each region (False)
)
