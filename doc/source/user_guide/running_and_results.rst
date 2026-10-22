.. _results:
.. _running:

Running a model and reading the results
=======================================

Solving
-------

.. code-block:: python

   model.run(solver="highs")

``solver`` is any solver installed for CVXPY (``import cvxpy;
cvxpy.installed_solvers()`` lists them; the name is not case sensitive).
HiGHS is installed with Hypatia and solves both linear and mixed-integer
models; Gurobi and CPLEX need their own installation and licence. Further
keyword arguments are passed to the solver through CVXPY's
``Problem.solve``, for example a time limit or an optimality gap:

.. code-block:: python

   model.run(solver="gurobi", verbosity=True, TimeLimit=3600, MIPGap=0.005)

``verbosity=True`` (the default) prints the solver log. Running a model that
already has results raises an error unless ``force_rewrite=True`` is given;
the previous results are then kept in ``model.backup_results``.

What happens next depends on the solver's answer:

* an optimal solution: the results are kept;
* an *inaccurate* optimum, or a run stopped by a limit (a time limit, say)
  with a feasible solution: the results are kept, with a warning naming the
  solver status. A mixed-integer solution stopped by a time limit is feasible
  but not proven optimal: the gap is in the solver log;
* no usable solution (infeasible, unbounded, or stopped before finding any
  feasible point): Hypatia raises ``NoSolution`` with the solver status, and
  ``model.results`` stays empty. An infeasible model is easiest to diagnose
  with ``ensure_feasibility="Yes"``: the unmet demand shows where supply is
  missing.

The objective is chosen when the model is created
(``optimization=...``) and each has its own run method:

.. list-table::
   :widths: 18 22 60
   :header-rows: 1

   * - ``optimization``
     - Method
     - What it does
   * - ``"Single"``
     - ``run(solver)``
     - Minimises the total discounted cost.
   * - ``"MinEm"``
     - ``run_MinEm(solver)``
     - Minimises the total emissions over the horizon (all emission types
       added together), then the cost among the solutions with minimum
       emissions (within a relative :math:`10^{-6}`).
   * - ``"Multi"``
     - ``run_MO(solver, number_solutions, path)``
     - Solves for the least cost and for the least emissions, then for
       ``number_solutions - 2`` evenly spaced emission limits in between,
       each time at least cost. Every solution is written to
       ``path/1``, ``path/2``, ... with the aggregated module below; the
       first is the least-cost solution, the last the least-emission one. A
       point with no usable solution is skipped with a warning; the others
       are still saved.

The equations are in :ref:`objective-modes`. ``number_solutions`` must be at
least 2.

Saving the results
------------------

.. code-block:: python

   model.to_csv(path="results", postprocessing_module="aggregated")

writes the results as CSV files, creating the folder and its parents if
needed. A folder that already holds files is used only with
``force_rewrite=True``, and then emptied first, so that no file of an earlier
run is left behind. To protect a mistyped path, a folder holding anything
other than CSV files is never emptied: ``to_csv`` stops with the list of
those files.

Two formats are available.

``postprocessing_module="aggregated"`` (recommended)
   One long-format table per quantity, with one value per row and the
   indices as columns. These tables are easy to filter, pivot and plot with
   pandas, and to compare between scenarios.

``postprocessing_module="default"``
   One folder per quantity, one subfolder per region and one file per
   technology category, in wide format (years, or years and time steps, as
   rows; technologies as columns). This is the historical format of Hypatia.

Aggregated results
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 38 32
   :header-rows: 1

   * - File
     - Content
     - Columns
   * - ``tech_production.csv``
     - Production of each technology, by carrier, in each time step.
     - Year, Timesteps, Region, Technology, Carrier, Value
   * - ``tech_production_annual.csv``
     - The same, summed over the year.
     - Year, Region, Technology, Carrier, Value
   * - ``tech_use.csv``, ``tech_use_annual.csv``
     - Carrier consumed by each technology.
     - as above
   * - ``total_capacity.csv``
     - Capacity available in each year.
     - Datetime, Region, Technology, Tech category, Value
   * - ``new_capacity.csv``
     - Planning: new capacity decided in each year.
     - as above
   * - ``tech_cost.csv``
     - Annual costs by item (``fixed_cost``, ``variable_cost``,
       ``investment_cost``, taxes, subsidies, ``CO2_emission_cost``,
       ``decommissioning_cost``), not discounted.
     - Datetime, Region, Technology, Cost, Value
   * - ``emissions.csv``, ``captured_emissions.csv``
     - Emissions released and captured, by technology and emission type.
     - Datetime, Region, Technology, Emission, Value
   * - ``unmet_demand.csv``, ``unmet_demand_annual.csv``
     - Demand not met, by carrier (zero unless ``ensure_feasibility="Yes"``).
     - Years, Timesteps, Region, Carrier, Value
   * - ``line_import.csv``, ``line_export.csv`` and their ``_annual``
       versions
     - Several regions: flows on the lines, by carrier.
     - Year, Timesteps, To reg / From reg, Carrier, Value
   * - ``pareto_point.csv``
     - The total discounted cost (NPC) of the solution and its total
       emissions: one point of the cost-emission trade-off.
     - NPC, Emissions
   * - ``modular_unit_commitment.csv``
     - Planning: units online, installed and new for technologies with unit
       commitment (:ref:`unit-commitment`).
     - Year, Timeslice, Region, Technology, ...

``Datetime`` and ``Year`` hold the calendar year (``Year_name``), while the
unmet demand files use the year code. The costs in ``tech_cost.csv`` do not
include the salvage value and the costs of lines, so their discounted sum is
not the objective; the NPC in ``pareto_point.csv`` is.

Default results
~~~~~~~~~~~~~~~

The default module writes, per region and technology category: production
and use (``production_by_tech``, ``use_by_tech``), capacity
(``total_capacity``, ``new_capacity``, ``decommissioned_capacity``), every
cost item (``fix_cost``, ``variable_cost``, ``investment_cost``,
``salvage_investment_cost``, taxes and subsidies, ``decommissioning_cost``,
``<emission>_emission_cost``), emissions (``<emission>_emission``) and the
demand. With several regions it adds ``imports``, ``exports`` and the
``lines_*`` capacities and costs. Costs are annual and not discounted;
``investment_cost`` is the full cost of the capacity decided in that year,
while the objective spreads it over annuities (:eq:`annuities`).

Working with the results in Python
----------------------------------

``model.results`` holds the solved quantities as CVXPY expressions, indexed
by region and technology category; ``.value`` gives the numbers:

.. code-block:: python

   capacity = model.results.totalcapacity["reg1"]["Supply"].value  # years x technologies
   techs = model.settings.technologies["reg1"]["Supply"]

For most analyses the aggregated CSV files are simpler.

Comparing scenarios
-------------------

:func:`~hypatia.Merge_results` stacks the aggregated results of several runs
into one set of files, with a ``Scenario`` column in front:

.. code-block:: python

   from hypatia import Merge_results

   Merge_results({"reference": "results/reference", "low_cost_pv": "results/low_cost_pv"},
                 path="results/merged")

It merges the production, use, capacity, cost, emission and unmet demand
tables and ``pareto_point.csv``, not the line flows. ``path`` must not exist
unless ``force_rewrite=True``, in which case it is deleted first.

Plots
-----

Hypatia draws interactive plots (HTML files, with Plotly) of capacity,
production, emissions and costs. The plots need a configuration file that
assigns names, colours and groups to technologies and carriers:

.. code-block:: python

   model.create_config_file("my_model/config.xlsx")   # then fill groups and colours

   from hypatia import Plotter
   plots = Plotter(model, config="my_model/config.xlsx", hourly_resolution=False)
   plots.plot_total_capacity(path="plots/total_capacity.html", tech_group="Power Generation")
   plots.plot_prod_by_tech(tech_group="Power Generation", path="plots/production.html")
   plots.plot_emissions(path="plots/emissions.html", tech_group="Resource", emission_type="CO2")
   plots.plot_regional_costs(path="plots/costs.html")

In the configuration file, each technology and carrier gets a group
(``tech_group``, ``fuel_group``), and the plots show one group at a time;
colours are any CSS colour (``#E377C2``, ``rgb(220,176,242)``).
``plot_hourly_prod_by_tech`` needs ``hourly_resolution=True`` and hourly time
steps. The examples with a ``config.xlsx`` (Atlantis, for instance) can be
plotted with ``model.load_plots()``. The methods are listed in the
:ref:`api`.
