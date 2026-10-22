.. _unit-commitment:

Modular capacity and unit commitment
====================================

*Based on the notes by Luigi Malco Rossi, who contributed the modular unit
commitment in October 2026.*

Some technologies come in units of a fixed size: a 300 MW reactor, a 50 MW
gas engine. Two inputs describe this:

``Modular_cap_unit`` (Planning mode)
   The size of one unit, in the capacity unit of the technology. With a
   positive value, new capacity can only be built in whole units: Hypatia
   decides an integer number of new units each year. With 0 (the default),
   new capacity is continuous.

``Min_power_output`` (option ``min_tech_power_output``)
   The minimum output of a running plant, as a fraction of its available
   output. With 0 (the default) the technology has no minimum.

Together they decide how a technology is modelled:

.. list-table::
   :widths: 14 22 14 14 36
   :header-rows: 1

   * - Mode
     - ``min_tech_power_output``
     - Unit size
     - Minimum
     - Result
   * - Planning
     - not enabled
     - any
     - (not read)
     - Modular investment if the unit size is positive; no minimum output.
   * - Planning
     - enabled
     - > 0
     - > 0
     - Modular investment and unit commitment: an integer number of units
       online in each time step.
   * - Planning
     - enabled
     - 0
     - > 0
     - Continuous investment; minimum output with one on/off variable for
       the whole technology.
   * - Planning
     - enabled
     - > 0
     - 0
     - Modular investment only.
   * - Operation
     - enabled
     - (not an input)
     - > 0
     - Minimum output with one on/off variable for the whole technology.

Unit commitment
---------------

For a technology with both a unit size :math:`STEP` and a minimum
:math:`\underline{p}`, Hypatia adds an integer variable :math:`n^{on}`, the
number of units online, for every year and time step. Each online unit can
produce between :math:`\underline{p}\,U` and :math:`U`, where :math:`U` is the
production of one unit at full output in that time step (unit size ×
resource capacity factor × ``AnnualProd_perunit_capacity`` × time-step
fraction). The production of the technology must lie between the sums over
the online units, and the online units cannot exceed the installed capacity:

.. math::

   \underline{p}\, U\, n^{on} \;\leq\; p \;\leq\; U\, n^{on},
   \qquad STEP\; n^{on} \leq CAP

So with no unit online the technology produces nothing, and with
:math:`n` units online it produces at least :math:`n` times the minimum of one
unit. The exact formulation is :eq:`unit-commitment`.

Residual capacity need not be a whole number of units; only whole units can
be online, so a fraction of a unit in the residual capacity is never used.
Decide whether the residual capacity should be a multiple of the unit size.

Without a unit size, the minimum is applied to the technology as a whole
with an on/off variable and a large constant (:eq:`min-output`): either the
technology is off, or it produces at least the minimum share of *all* its
available capacity. This is coarser and numerically weaker than the modular
form, and it is the only form available in Operation mode, where the
capacity is given.

What it does not model
----------------------

This is a clustered commitment: it counts how many identical units are
online in each time step, independently from one time step to the next. It
has no start-up or shut-down costs, no minimum up and down times, and no
ramp limits per unit (the ``production_ramp`` option applies to the
technology as a whole). All units of a technology share efficiency, capacity
factor and costs. With representative days or coarse time slices, the number
of units online is an average behaviour rather than a schedule.

For a CHP plant modelled as in :ref:`optional-chp-usage`, apply the unit size
and the minimum to the CHP unit, not to its branches.

Practical points
----------------

* use a solver for mixed-integer problems;
* keep ``Min_power_output`` between 0 and 1, and ``Modular_cap_unit`` in the
  same unit as the capacity;
* check that capacity limits (``Max_totalcap``, ``Max_newcap``) allow at
  least one unit: a maximum smaller than the unit size forbids the
  technology.

Results
-------

The aggregated results (``to_csv(..., postprocessing_module="aggregated")``)
of a Planning model include ``modular_unit_commitment.csv``, with one row per
year, time step, region and committed technology: ``Units_online``,
``Installed_units`` (total capacity / unit size), ``New_units`` and
``Core_production``. The file has only headers when no technology has both a
unit size and a minimum.
