.. _user-guide:
.. _concepts:

Concepts
========

The reference energy system
---------------------------

A Hypatia model describes an energy system as a network of *technologies*
connected by *energy carriers*: a technology takes carriers in, produces
carriers out, and the carriers produced by one technology are available to
the others in the same region. This network is the *reference energy system*.
It starts from supply technologies, which bring energy into the system (a gas
field, a solar plant, imports from outside the model), and ends with demand
technologies, which consume it (households, industry, transport).

Technology categories
~~~~~~~~~~~~~~~~~~~~~

Every technology belongs to one of six categories, which decide how it is
modelled.

.. list-table::
   :widths: 18 52 30
   :header-rows: 1

   * - Category
     - What it does
     - Diagram
   * - Supply
     - Brings a carrier into the system without consuming any (extraction,
       renewable resources, imports from outside the model).
     - .. image:: ../_static/Tech_ctgry/supply.png
          :width: 100
   * - Conversion
     - Converts one carrier into another with an efficiency (a gas plant
       turning gas into electricity).
     - .. image:: ../_static/Tech_ctgry/conversion.png
          :width: 130
   * - Conversion_plus
     - Converts several carriers into several others, in fixed ratios (a
       refinery, a cogeneration unit modelled as one technology).
     - .. image:: ../_static/Tech_ctgry/conversion_plus.png
          :width: 130
   * - Transmission
     - Moves a carrier within a region, with losses (distribution grids,
       pipelines). Lines *between* regions are modelled separately.
     - .. image:: ../_static/Tech_ctgry/transmission.png
          :width: 130
   * - Storage
     - Stores a carrier and gives it back later, with charging and
       discharging efficiencies. Meant for hourly time steps.
     - .. image:: ../_static/Tech_ctgry/storage.png
          :width: 100
   * - Demand
     - Consumes a carrier according to a demand profile given as input.
     - .. image:: ../_static/Tech_ctgry/demand.png
          :width: 100

Supply technologies have only an output carrier, demand technologies only an
input carrier, and all other categories one input and one output, except
Conversion_plus, which can have several of each. A technology present in
several regions has the same name in all of them.

Carriers
~~~~~~~~

Carriers are any form of energy or fuel the model tracks: electricity, heat,
natural gas, hydrogen, uranium. Each carrier has a type, which helps reading
the model but does not change the equations:

* *Resource*: energy as extracted from nature, not yet processed (raw oil);
* *Intermediate*: a carrier used by technologies other than demand;
* *Demand*: a carrier consumed by demand technologies.

Units are free, carrier by carrier: electricity in MWh, gas in GWh, uranium
in tonnes. Efficiencies and carrier ratios convert between them, so they must
be expressed in the units chosen (see :ref:`nomenclature`).

Regions and lines
-----------------

A model has one or more *regions*. Each region has its own reference energy
system and its own data; a region can be anything from a village to a
country. With several regions, Hypatia creates one *line* between each pair
of regions, through which every carrier can be exchanged, with losses,
capacity limits and costs. Some limits and parameters can then also be set
for the whole system (global parameters).

Time
----

The time horizon is a list of *years*, each divided into the same *time
steps*. The time steps can be the 8760 hours of a year, a set of
representative days, or a few slices (winter days, winter nights, ...): each
has a *fraction* of the year, and the fractions add up to one. Hourly time
steps are needed for storage and ramp limits. The years are assumed to be
consecutive calendar years: lifetimes and discounting count them one by one.

Modes
-----

``mode="Operation"``
   One year, with the installed capacity given. Hypatia decides how each
   technology runs in each time step, at minimum cost. Useful for dispatch
   studies and to check a future energy mix hour by hour.

``mode="Planning"``
   Several years. Hypatia also decides how much new capacity of each
   technology to build each year, with investment costs, lifetimes and
   discounting. This is the mode for transition scenarios.

Objective
~~~~~~~~~

``optimization="Single"`` minimises the total discounted cost.
``optimization="MinEm"`` minimises the total emissions instead, and
``optimization="Multi"`` computes a series of solutions between the cheapest
and the least-emitting one (see :ref:`objective-modes`).

Unmet demand
~~~~~~~~~~~~

With ``ensure_feasibility="No"`` every demand must be met, and a model that
cannot meet it has no solution. With ``ensure_feasibility="Yes"`` part of the
demand can be left unmet, at a very high cost (:math:`10^{10}` per unit of
energy). This is useful to find out *where* a model is short of supply: the
results show the unmet demand by carrier and time step.

Optional constraints
--------------------

Some features are switched on per model: a variable operating range for
combined heat and power plants, ramp limits, minimum output and unit
commitment, green hydrogen from electrolysis, and no simultaneous charging
and discharging of storage. They are listed in :ref:`optional-constraints`.
