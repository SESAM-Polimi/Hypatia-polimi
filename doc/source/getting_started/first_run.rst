.. _first-run:

A first run
===========

Hypatia ships with example models. The script below loads one, solves it and
saves the results; it takes a few seconds.

.. literalinclude:: ../../examples/first_run.py
   :language: python
   :lines: 5-

``print(model)`` shows what the model contains:

.. code-block:: text

   name = Atlantis
   mode = ModelMode.Planning
   regions= ['reg1']
   techs= {'reg1': {'Supply': ['FossilFuel_supply', 'RES_PP'], 'Conversion': ['FossilFuel_PP'],
           'Transmission': ['Elec_distribution'], 'Demand': ['Electricity_demand']}}
   horizon= ['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10']
   resolution= 24
   optional constraints= none

The folder ``results`` then contains one CSV file per quantity
(``total_capacity.csv``, ``new_capacity.csv``, ``tech_production.csv``,
``tech_cost.csv``, ``emissions.csv`` and others), in a long format that is
easy to filter and plot with pandas or a spreadsheet:

.. code-block:: text

   ,Datetime,Region,Technology,Cost,Value
   0,2025,reg1,FossilFuel_supply,fixed_cost,0.0
   1,2026,reg1,FossilFuel_supply,fixed_cost,0.0

Where to go from here:

* :ref:`concepts` explains how Hypatia represents an energy system;
* :ref:`building-a-model` shows how to build a model of your own;
* :ref:`tutorials` walk through a planning and an operation model;
* :ref:`results` describes every output.

To work on a copy of an example, for instance to change its data, copy its
files to a folder of your own:

.. code-block:: python

   from hypatia import download_example
   download_example("Atlantis", "my_atlantis")
