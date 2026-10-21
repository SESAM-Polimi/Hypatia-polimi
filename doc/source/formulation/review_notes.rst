.. _review-notes:

Review notes
============

While writing this documentation from the code (October 2026), the following
points were found where the model does something that may not be intended.
They are documented here as they are; none was changed. Each is a candidate
for a decision by the developers, and some for a fix.

Costs
-----

Salvage value and lifetimes
   The salvage value (:eq:`salvage`) uses the technical lifetime for the
   share of an investment beyond the horizon, the annuities
   (:eq:`annuities`) use the economic lifetime. The salvage value removes
   the right amount only when the two are equal.
Discount factors
   Each year is discounted with its own rate raised to the year index,
   :math:`(1+d_y)^{-y}`, and annuities with the rate of the year of the
   investment. With a discount rate that changes over time this is not the
   usual compounding :math:`\prod_{y' \leq y} (1+d_{y'})^{-1}`.
Lines are paid upfront
   Investments in lines are paid in full in the year of the decision, with
   no annuities and no salvage value, while technologies have both. The
   sheets ``Interest_rate`` and ``Line_Economic_life`` of the connections
   file are read but not used.
Unmet demand penalty
   Fixed at :math:`10^{10}` per unit of energy in the code; it cannot be set
   from the inputs. With the example data it dominates the objective (values
   of :math:`10^{16}` to :math:`10^{19}`).

Optimisation modes
------------------

Exact emission bound in MinEm and Multi
   The second step of ``run_MinEm`` and the last point of ``run_MO`` bound
   the emissions by exactly the minimum found in the first step, which a
   solver can report as infeasible (Atlantis with HiGHS: infeasible with the
   exact bound, optimal with a relative slack of :math:`10^{-9}`). In
   ``run_MinEm`` a relaxed problem (factor 1.000695262) is then built but
   never solved: the method prints "No solution found" and yet sets
   ``model.results``, with every value ``None``. In ``run_MO`` the
   least-emission folder is missing. Neverland is solved in both cases.
Emission types are added without weights
   in the emission objective and in the :math:`\varepsilon`-constraint of
   the multi-objective mode.

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
Time steps longer than one hour
   The energy limits multiply the capacity by the length of the time step in
   hours (:math:`8760\,\tau_t`), which is 1 only in hourly models.
Boolean storage
   :eq:`boolean-storage` leaves out the last time step of each year and uses
   :math:`M = 10^{20}`, which is numerically risky.

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

CHP efficiencies
   are averaged over the years.
Ramp limits
   compare the first hour of each year after the first with zero, not with
   the last hour of the previous year.
Ramp limits and storage
   A comment in ``ConstraintList.py`` says that ``production_ramp`` is "not
   compatible with StorageCyclicBoundary as currently formulated". This was
   not checked while writing the documentation.

Inputs, running and results
---------------------------

Fixed in October 2026: Operation models with several years, regions without
demand, time-step fractions, capacity-group years, ``run_MO`` arguments and
``to_csv`` folders are now checked, with clear messages (see the
:ref:`changelog`).

Solver status
   Results are kept only when the status is exactly "optimal"; a
   mixed-integer run stopped by a time limit with a feasible solution, or an
   "optimal_inaccurate" answer, leaves ``model.results`` at ``None`` with a
   printed message and no exception.
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
