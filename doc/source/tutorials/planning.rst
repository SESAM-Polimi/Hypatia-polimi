.. _tutorials:
.. _tutorial-planning:

Tutorial: a planning model
==========================

This tutorial builds the example model *Neverland* from its files, finds its
least-cost development over ten years, and then computes how the cost
changes when emissions are reduced. The complete script is
``doc/examples/planning_tutorial.py`` in the repository; it runs in about a
minute with HiGHS, and the documentation tests run it at every change of
the code, so the outputs shown here are the ones it produces.

The model
---------

Neverland has two regions connected by one line. Each region can use coal
and gas, imported into the model by supply technologies, in coal and gas
power plants, and can build hydro and solar plants. The electricity goes
through a distribution grid to the final demand. Capacities are in GW and
energy in GWh; the horizon is 2025 to 2034, with 24 time steps (one
representative day) per year. Region ``reg1`` starts with 8 GW of coal
plants, ``reg2`` with 2 GW of hydro.

Getting the files
-----------------

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [copy]
   :end-before: # [/copy]

The folder ``neverland`` now contains ``sets/`` (``global.xlsx``,
``reg1.xlsx``, ``reg2.xlsx``) and ``parameters/`` (one file per region,
``parameters_global.xlsx`` and ``parameters_connections.xlsx``). Open the
set files to see the structure described in :ref:`building-a-model`.

Creating the model
------------------

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [model]
   :end-before: # [/model]

``create_data_excels`` is not needed here, since the parameters are already
filled, but it shows the templates a new model starts from: compare
``neverland/templates`` with ``neverland/parameters`` to see the defaults
replaced by data. ``force_rewrite=True`` lets the script run again.

Solving and saving
------------------

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [run]
   :end-before: # [/run]

``to_csv`` creates the last folder of the path but not its parents, hence
``os.makedirs``. The ``assert`` stops the script if the solver did not find
an optimal solution (see :ref:`results`).

Reading the results
-------------------

The aggregated results are long tables, which pandas turns into the view
needed. The capacity of the power plants by region and year:

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [capacity]
   :end-before: # [/capacity]

.. code-block:: text

   Region        reg1                             reg2
   Technology Coal_PP Gas_PP Hydro_PP Solar_PP Coal_PP Gas_PP Hydro_PP Solar_PP
   Datetime
   2025           8.0    0.0      0.0      0.0     0.0    0.0      2.0      0.0
   2026           8.0    0.0      0.0      3.5     0.0    0.0      2.0      0.0
   2027           8.0    0.0      0.0      3.7     0.0    0.0      2.0      0.0
   ...
   2033           8.0    0.0      0.0      5.3     0.0    0.0      2.0      0.0
   2034           8.0    0.0      0.0      5.3     0.0    0.0      2.0      0.0

The cheapest plan keeps the existing plants, builds 3.5 GW of solar in
``reg1`` in 2026 and grows it to 5.3 GW by 2033, following the demand. No gas
plant is built. The costs by
item, summed over the ten years (not discounted):

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [costs]
   :end-before: # [/costs]

.. code-block:: text

   Cost
   variable_cost           8.393382e+09
   fixed_cost              7.326797e+09
   investment_cost         4.773984e+09
   CO2_emission_cost       0.000000e+00
   ...

There is no carbon price in Neverland, so emissions cost nothing: the plan
is cheapest, not cleanest.

Cost against emissions
----------------------

To see what lower emissions would cost, the same data is solved as a
multi-objective model. ``run_MO`` finds the least-cost and the
least-emission solutions, then the least-cost solution under evenly spaced
emission limits in between, and saves each one (:ref:`objective-modes`).

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [trade_off]
   :end-before: # [/trade_off]

.. code-block:: text

               NPC     Emissions
   0  1.514682e+10  9.731457e+07
   1  1.620469e+10  5.448011e+07
   2  2.015622e+10  1.164565e+07

Cutting the emissions by 44 % raises the total discounted cost by 7 %;
cutting them by 88 % raises it by 33 %. The capacity in the last year shows how:

.. literalinclude:: ../../examples/planning_tutorial.py
   :language: python
   :start-after: # [mix]
   :end-before: # [/mix]

.. code-block:: text

               solution 1  solution 2  solution 3
   Technology
   Coal_PP            8.0         8.0         8.0
   Gas_PP             0.0         0.0         0.0
   Hydro_PP           2.0         4.4        10.3
   Solar_PP           5.3         4.1         0.0

The coal plants stay in every solution, because they already exist and cost
little to keep, but in the least-emission solution they do not run (their
production, in ``tech_production_annual.csv``, is zero). Hydro, which can be built
only in ``reg2`` (its resource capacity factor is zero in ``reg1``), replaces
both coal and solar and reaches ``reg1`` through the line. Unlike solar it
produces at every hour, so it needs no storage, which Neverland does not
have.

.. note::

   With other data the last point of ``run_MO``, which asks for emissions no
   higher than the minimum, can be reported as infeasible because the limit
   is exactly the optimum; its folder is then missing. The script reads only
   the solutions that exist. See :ref:`review-notes`.

Going further
-------------

* Add a carbon price in ``Emission_tax`` (regional parameter files) and
  compare the least-cost plan with the trade-off curve.
* Allow the line between the regions to grow (``Max_totalcap`` and ``INV``
  in ``parameters_connections.xlsx``) and see whether hydro from ``reg2``
  replaces solar in ``reg1``.
* Plot the results with the configuration file shipped with the example:
  ``Plotter(model, "neverland/config.xlsx", hourly_resolution=False)``, see
  :ref:`results`.
