.. _review-notes:

Review notes
============

While writing this documentation from the code (October 2026), the following
points were found where the model does something that may not be intended.
Several were reviewed in October 2026: those fixed are listed in the
:ref:`changelog` and no longer appear here, those kept are marked below. The
others are still open.

Costs
-----

Salvage value when the interest rate differs from the discount rate
   The factors of :eq:`salvage` that convert the investment into the present
   value of its annuities include an extra :math:`(1+d)/(1+i)`. With
   :math:`i = d` it equals one and the salvage value is exactly the share of
   the financed cost that pays for technical life beyond the horizon (the
   technical and economic lifetimes may differ). With :math:`i \neq d` the
   salvage value is smaller than that share, by about 4 % for :math:`i = 7\%`,
   :math:`d = 3\%`.
   *Decision, October 2026: kept as designed.*
Discount factors
   Each year is discounted with its own rate raised to the year index,
   :math:`(1+d_y)^{-y}`, and annuities with the rate of the year of the
   investment. With a discount rate that changes over time this is not the
   usual compounding :math:`\prod_{y' \leq y} (1+d_{y'})^{-1}`.
   *Decision, October 2026: kept as designed.*
Lines are paid upfront
   Investments in lines are paid in full in the year of the decision, with
   no annuities and no salvage value, while technologies have both. The
   sheets ``Interest_rate`` and ``Line_Economic_life`` of the connections
   file are read but not used.

Optimisation modes
------------------

Emission types are added without weights
   in the emission objective and in the :math:`\varepsilon`-constraint of
   the multi-objective mode.
   *Decision, October 2026: kept as designed.*

Time
----

Years are positions, not calendar years
   Lifetimes, construction times and discounting count positions in the list
   of modelled years. A model with, say, 5-year steps would get lifetimes
   and discounting wrong by a factor of 5. Nothing checks that the modelled
   years are consecutive.
Construction time 0
   makes new capacity disappear instead of making it available in the same
   year (the shifted vector is empty). The minimum meaningful value is 1.

Storage
-------

Start and end of a cycle
   The state of charge (:eq:`soc`) is counted from zero at the start of each
   cycle (the storage starts empty), while :eq:`storage-cyclic` sets the end
   of each cycle to the minimum state of charge. The minimum in
   :eq:`storage-cyclic` omits the depth of discharge used in
   :eq:`soc-limits`. With one cycle per year, :eq:`storage-annual-cycle`
   then fixes the first time step to the same value.
   *Decision, October 2026: kept for now; flagged for a future improvement.*
Time steps longer than one hour
   The energy limits multiply the capacity by the length of the time step in
   hours (:math:`8760\,\tau_t`), which is 1 only in hourly models.
Boolean storage
   :eq:`boolean-storage` leaves out the last time step of each year and uses
   :math:`M = 10^{20}`, which is numerically risky.
   *Decision, October 2026: kept for now; flagged for a future improvement.*

Emissions and renewables
------------------------

Sign of emission factors
   The choice between emitting and removing technologies (:eq:`emissions`)
   is made with the first year's factor only.
Renewable share of total demand
   :eq:`renewable-total` compares the activity of renewable technologies
   with the final demand of all carriers added together; it requires all
   carriers in the same unit and counts a renewable supply and a renewable
   conversion of the same energy twice.
Power carriers by name
   Electricity constraints recognise carriers named ``Electricity`` or
   ``Power``; ``ElectrolysisConsumption`` only the exact name
   ``Electricity``. Carriers such as ``Final_electricity`` are not power
   carriers (to confirm that this is intended).

Optional constraints
--------------------

Ramp limits
   compare the first hour of each year after the first with zero, not with
   the last hour of the previous year.
   *Decision, October 2026: kept for now; flagged for a future improvement.*
Ramp limits and storage
   A comment in ``ConstraintList.py`` says that ``production_ramp`` is "not
   compatible with StorageCyclicBoundary as currently formulated". This was
   not checked while writing the documentation.

Inputs, running and results
---------------------------

Fixed in October 2026: Operation models with several years, regions without
demand, time-step fractions, capacity-group years, ``run_MO`` arguments and
``to_csv`` folders are now checked, with clear messages; solver statuses other
than "optimal" are handled explicitly; the exact emission bound of
``run_MinEm`` and ``run_MO`` has a small slack; the variable cost of lines is
right with three or more regions (see the :ref:`changelog`).

Aggregated cost table
   In the aggregated output, ``tech_cost.csv`` has no salvage value and no
   line costs.
Unused sets table
   ``Time_horizon`` is accepted in ``global.xlsx`` but never used.

Examples
--------

Carrier names with a trailing space
   Several examples name the electricity carrier ``"Electricity "``. The
   power-sector constraints ignore the space; anything matching names
   exactly does not.

Code
----

Duplicated objective
   ``ModelVariables`` recomputes the whole cost objective to report total
   costs, in a copy of the code of ``Build``; the two copies must be kept
   identical by hand.
Global production limits
   are added once per region that produces the carrier, so the same rows
   appear several times (harmless, but larger problems).
Unused code
   ``backend/StrData.py`` (about 1200 lines) is unused except for
   ``create_technology_columns``.
Accumulators starting from oversized zero arrays
   (e.g. in ``EmissionConsumedRegional``) give the right result only because
   CVXPY drops constant zeros when adding them to an expression.
