"""
Utopia MODEL
Planning mode
"""

# Import of the Model

from hypatia import Model
import os

#%% 
# Create the model using as input the sets files
OptimizationMode = "Single"                                             # "Single" or "MinEm" or "Multi" objective optimization. Single minimizes NPC, MinEm minimizes Emissions, Multi Objective minimizes NPC and CO2
Number_solutions = 3                                                    # Number of required solution in case of multi-objective optimization
Ensure_Feasibility = "No"                                               # "Yes" allows unmet demand, "No" otherwise                                               

Utopia = Model(
    path="examples/Operation_teaching_1Region/sets",                             # Path to the sets folder
    mode="Operation",                                                    # "Planning" or "Operation" mode
    optimization = OptimizationMode,
    ensure_feasibility = Ensure_Feasibility                                     
)

#%% 
# Create the parameters with default values

# Utopia.create_data_excels(
#     path ='examples/Planning_teaching_2Regions/parameters',                      # Path to the parameters folder
#     force_rewrite=True                                                  # Overwrite the parameters files (True) or not (False)
# )

#%% 
# Read the parameters

Utopia.read_input_data("examples/Operation_teaching_1Region/parameters")         # Path to the parameters folder

#%% 
# Run the model to find the optimal solution

if OptimizationMode == "Multi":    
    Utopia.run_MO(
        solver='gurobi',                                                    # Selection of the solver: 'GUROBI', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS’
        number_solutions = Number_solutions,
        path = "examples/Operation_teaching_1Region/Pareto Froniter",                               # Path to the destination folder for the Pareto Frontier plot
        verbosity=True,
        force_rewrite= True                                                 # Overwrite the parameters files (True) or not (False)
    )
elif OptimizationMode == "Single":
    Utopia.run(
        solver='gurobi',                                                    # Selection of the solver: 'GUROBI', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS’
        verbosity=True,
        force_rewrite= True                                                 # Overwrite the parameters files (True) or not (False)
    )
else:
    Utopia.run_MinEm(
        solver='gurobi',                                                    # Selection of the solver: 'GUROBI', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS’
        verbosity=True,
        force_rewrite= True                                                 # Overwrite the parameters files (True) or not (False)
    )
    
#%%
# Create results folder    
    
if not os.path.exists("examples/Operation_teaching_1Region/results"):
    os.mkdir("examples/Operation_teaching_1Region/results")
    
#%%
# Save the results as csv file in the previous folder

Utopia.to_csv(
    path= "examples/Operation_teaching_1Region/results",                         # Path to the destination folder for the results
    force_rewrite=True,                                                 # Overwrite the parameters files (True) or not (False)
    postprocessing_module="aggregated"                                  # "default" and "aggregated" are the two options
)