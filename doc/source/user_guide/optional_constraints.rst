.. _optional-constraints:

Optional constraints
====================

Five constraints are switched on per model, by name, when the model is
created:

.. code-block:: python

   model = Model(path="my_model/sets", mode="Planning", optimization="Single",
                 ensure_feasibility="No",
                 optional_constraints=["chp_operating_range", "production_ramp"])

.. list-table::
   :widths: 26 44 30
   :header-rows: 1

   * - Name
     - What it adds
     - Sheets it needs
   * - ``chp_operating_range``
     - Heat and power of combined heat and power plants stay inside the
       plant's operating region.
     - ``CHP_variable_techs_selection``
   * - ``production_ramp``
     - Output changes from one hour to the next are limited.
     - ``Max_rate_ramp-up``, ``Max_rate_ramp-down``
   * - ``min_tech_power_output``
     - A running plant produces at least a minimum share of its available
       output; in Planning mode, modular technologies get unit commitment.
     - ``Min_power_output`` (and ``Modular_cap_unit``, always present in
       Planning mode)
   * - ``electrolysis_consumption``
     - Electrolysers use no more electricity in a year than the region's
       renewable plants produce.
     - none (uses ``Renewable_tech_selection``)
   * - ``boolean_for_storage``
     - A storage does not charge and discharge in the same time step.
     - none

The names are not case sensitive; a name that does not exist stops the model
with the list of valid ones. ``print(model)`` and ``model.settings.optional_constraints``
show which options are active. The equations are in
:ref:`formulation-optional`.

.. warning::

   An option adds its sheets to the parameter templates and to the files
   Hypatia reads. The reverse also holds: a sheet in your parameter file is
   not read if its option is not enabled. A model whose files contain a
   filled ``CHP_variable_techs_selection`` sheet, run without
   ``chp_operating_range``, runs without the CHP constraint. Hypatia warns
   when it finds such a sheet holding data (sheets left at their default
   values are ignored quietly, since they change nothing).

``min_tech_power_output`` and ``boolean_for_storage`` add integer or binary
variables: the model becomes a mixed-integer program, which needs a solver
that supports it (HiGHS, Gurobi, CPLEX, CBC) and usually takes much longer
to solve.

.. _optional-chp-usage:

CHP operating range
-------------------

A cogeneration plant whose heat-to-power ratio can vary is represented by
three Conversion technologies:

* the *CHP unit*, which consumes the fuel and produces an intermediate
  carrier (for example ``CHP_Heat``). It carries all the costs, the capacity
  and the emissions of the plant;
* the *power branch*, which turns the intermediate carrier into electricity
  with the maximum electrical efficiency of the plant;
* the *heat branch*, which turns it into useful heat.

The branches are accounting devices: give them no costs and a large residual
capacity, so that only the CHP unit is sized. Each row of
``CHP_variable_techs_selection`` describes one plant:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Column
     - Content
   * - ``CHP group``
     - A row number (1, 2, 3, ...).
   * - ``CHP Tech Name``
     - The CHP unit.
   * - ``Power-branch Tech Name``
     - The power branch.
   * - ``Heat-branch Tech Name``
     - The heat branch.
   * - ``Max Heat Power Coefficient``
     - :math:`\gamma`: the largest heat output per unit of activity of the
       CHP unit.

Rows whose names are not technologies of the region (the template's ``0``)
are skipped, so unused rows can stay. The power branch's efficiency together
with :math:`\gamma` and the heat branch's efficiency set the minimum power
for a given heat output, see :eq:`chp`. The example
``Operation_1Region_CHP-Variable-range`` models a small nuclear CHP plant this
way, and the folder ``hypatia/examples`` has a short guide to the parameters
(``How to use CHP Variable Range.pdf``).

.. _optional-ramp-usage:

Ramp limits
-----------

``Max_rate_ramp-up`` and ``Max_rate_ramp-down`` give, per technology, the
largest change of output per *minute* as a fraction of the available output
(for example 0.02 for 2 %/min). The default, 1, means no limit. The limits
are meant for hourly time steps; Hypatia prints a reminder when the option
is on. See :eq:`ramp` for how the per-minute rate becomes an hourly limit.
A comment in the code warns that ramp limits are not compatible with the
storage cycle constraint as currently formulated; check the results of
models with both (:ref:`review-notes`).

.. _optional-min-output-usage:

Minimum output
--------------

``Min_power_output`` gives, per technology, the minimum output of a running
plant as a fraction of its available output (0 to 1); 0 leaves the
technology unconstrained. How it is applied depends on the mode and on
``Modular_cap_unit``, see :ref:`unit-commitment`.

Green hydrogen
--------------

In each region with a technology named exactly ``Electrolysis``, the
electricity it uses over a year cannot exceed the electricity produced in
that year by the technologies marked as renewable in
``Renewable_tech_selection``. Only the carrier named exactly
``Electricity`` counts. Regions without an electrolyser are not affected.

No simultaneous charge and discharge
------------------------------------

Applies to storage technologies whose input and output carrier are the same.
Without it, a storage may charge and discharge in the same time step when
that is free, which wastes energy through the efficiencies but is
sometimes chosen by the solver to dispose of surplus production. The
constraint uses a large constant (:math:`10^{20}`), which can cause numerical
trouble; see :ref:`optional-boolean-storage`.

.. _optional-capacity-groups:

Capacity groups
---------------

Capacity groups limit the total or new capacity of a set of technologies,
summed over the technologies and over all regions: "at most 30 GW of nuclear
in total", "at most 5 GW of new offshore wind per year". They are not an
option of ``Model``: they are active when their sheets are present and
filled, in Planning mode only. The sheets go in ``parameters_global.xlsx``
for models with several regions and in the regional file otherwise; the
templates contain them, empty.

``Capacity_group_membership``
   Columns ``Group`` and ``Technology``: one row per technology in a group.
   A technology may belong to several groups. Demand technologies are not
   allowed, and all technologies of a group must have the same capacity
   unit (``Tech_cap_unit``).

``Max_totalcap_group``
   Columns ``Group``, ``Year``, ``Max_totalcap``: the upper limit on the
   total capacity of the group in each year.

``Max_newcap_group``
   Columns ``Group``, ``Year``, ``Max_newcap``: the upper limit on the new
   capacity of the group in each year (capacity becoming available that
   year).

``Year`` is the year *code* of the ``Years`` table (``Y0``, ``Y1``, ...).
Every group defined must be used by at least one of the two limit sheets,
and Hypatia checks names, duplicates and years when it reads the data.

A year without a row for a group has no limit for that group, so only the
constrained years need listing.

.. code-block:: text

   Capacity_group_membership        Max_totalcap_group
   Group      Technology            Group     Year   Max_totalcap
   nuclear    SMR                   nuclear   Y0     0
   nuclear    Large_LWR             nuclear   Y1     0
                                    nuclear   Y2     5000
                                    ...
