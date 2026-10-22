.. _tutorial-operation:

Tutorial: an operation model
============================

An operation model takes the installed capacity as given and finds the
cheapest way to run it, hour by hour, for one year. This tutorial runs the
example *Operation_2Regions*, looks at the dispatch and at the exchanges
between the regions, and then doubles the demand to see where and when the
system falls short. The script is ``doc/examples/operation_tutorial.py``; it
runs in about a minute with HiGHS and is run by the documentation tests.

The model
---------

Two regions, 8760 hourly time steps, year 2020. Energy is in kWh and
capacity in kW; the script converts to GWh and MW for printing.

* ``reg1`` has 704 MW of heavy-fuel-oil plants (``HFO_PP``), 60 MW of gas
  turbines (``OCGT_PP``), 133 MW of hydro and 25 MW of wind;
* ``reg2`` has 624 MW of geothermal, 678 MW of hydro, 100 MW of heavy-fuel-oil
  and 36 MW of biomass plants;
* a 420 MW line connects them.

Running
-------

.. literalinclude:: ../../examples/operation_tutorial.py
   :language: python
   :start-after: # [run]
   :end-before: # [/run]

:func:`~hypatia.load_example` does what :ref:`building-a-model` describes:
it creates the model from the example's set files (an operation model,
because the name starts with ``Operation``) and reads its parameters.

The year in numbers
-------------------

Annual electricity production by plant and region, in GWh:

.. literalinclude:: ../../examples/operation_tutorial.py
   :language: python
   :start-after: # [annual]
   :end-before: # [/annual]

.. code-block:: text

   Region        reg1    reg2
   Technology
   BW_PP          0.0     0.0
   Geo_PP         0.0  5459.0
   HFO_PP      2132.0     0.0
   Hydro_PP     644.0   781.0
   OCGT_PP        0.0     0.0
   Wind_PP       62.0     0.0

Geothermal runs at full output all year (5459 GWh from 624 MW). The gas
turbines and the biomass plant are never used, being more expensive than the
alternatives. And the exchanges:

.. literalinclude:: ../../examples/operation_tutorial.py
   :language: python
   :start-after: # [lines]
   :end-before: # [/lines]

.. code-block:: text

     To reg From reg            Carrier     GWh
   4   reg1     reg2  Final_electricity  3600.0

``reg2`` sends 3600 GWh to ``reg1``, 98 % of what the 420 MW line can carry
in a year. In this example the line carries ``Final_electricity`` only (the
other carriers have zero line capacity), so the exchange happens after the
distribution grids.

A week in July
--------------

The hourly results show how ``reg1`` follows its demand:

.. literalinclude:: ../../examples/operation_tutorial.py
   :language: python
   :start-after: # [week]
   :end-before: # [/week]

.. code-block:: text

   Technology  HFO_PP  Hydro_PP  OCGT_PP  Wind_PP
   mean         255.0      42.0      0.0      4.0
   min            9.0      42.0      0.0      0.0
   max          526.0      42.0      0.0     22.0

Hydro and wind produce what their resource allows (the hydro profile of the
example is flat within the week), and the oil plants fill the rest, between
9 and 526 MW. Plotting ``dispatch`` with pandas or with
:class:`~hypatia.Plotter` (``plot_hourly_prod_by_tech``) shows the daily
cycle.

Doubling the demand
-------------------

*Operation_2Regions_UnMetDemand* has the same plants and twice the demand,
and is built with ``ensure_feasibility="Yes"`` (the name contains
``UnMetDemand``): demand that cannot be met is allowed, at a penalty of
:math:`10^{10}` per kWh. Without it the model would have no solution, and
the solver would only say so.

.. literalinclude:: ../../examples/operation_tutorial.py
   :language: python
   :start-after: # [unmet]
   :end-before: # [/unmet]

.. code-block:: text

           hours     GWh  peak_MW
   Region
   reg1     6451  2634.0   1300.0
   reg2       86     3.0    137.0

Every plant now runs harder, the gas turbines and the biomass plant
included, but ``reg1`` still misses 2634 GWh, a fifth of its demand, in 6451
hours of the year, and up to 1300 MW at the worst hour. ``reg2`` is short for
86 hours only. The penalty makes the objective meaningless as a cost
(:math:`10^{10}` per kWh dominates everything else), but the unmet demand
tells where capacity is missing: here, in ``reg1``, and more than the line
could bring.

Going further
-------------

* Increase the line capacity (``Residual_capacity`` in
  ``parameters_connections.xlsx``) and see how much of the shortfall of
  ``reg1`` it removes.
* Add ramp limits to the oil plants (option ``production_ramp``, see
  :ref:`optional-constraints`) and compare the dispatch.
* For planning the missing capacity instead of just measuring it, the same
  system can be modelled in Planning mode: see :ref:`tutorial-planning`.
