"""First run: load an example model, solve it, save the results.

Shown in doc/source/getting_started/first_run.rst and run by tests/test_docs.py.
"""
from hypatia import available_examples, load_example

# The example models shipped with Hypatia.
print(available_examples())

# A small planning model: one region, eleven years, 24 time steps per year.
model = load_example("Atlantis")
print(model)

# Solve with HiGHS, an open-source solver installed with CVXPY.
model.run(solver="highs", verbosity=False)

# Write the results as CSV files, one table per quantity.
model.to_csv(path="results", postprocessing_module="aggregated", force_rewrite=True)
