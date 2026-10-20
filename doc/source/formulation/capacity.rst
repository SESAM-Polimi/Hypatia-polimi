.. _capacity:

Capacity
========

In Planning mode the model decides, every year, how much new capacity of each
technology to build. In Operation mode the capacity is given:
:math:`CAP_{r,k} = RC_{r,k}`, and this section does not apply.

Construction, accumulation and retirement
-----------------------------------------

*In plain words.* Capacity decided in year :math:`y` becomes available
:math:`TC` years later, stays in operation for its technical lifetime
:math:`LT`, then retires. The installed capacity of a year is all the new
capacity still within its lifetime plus the residual capacity, which the user
gives year by year (so its own retirement is part of the input).

New capacity available in year :math:`y`:

.. math::
   :label: real-newcap

   \hat N_{r,k,y} =
   \begin{cases}
     N_{r,k,\,y - TC_{r,k}} & \text{if } y \geq TC_{r,k} \\
     0 & \text{otherwise}
   \end{cases}

Total installed capacity:

.. math::
   :label: total-capacity

   CAP_{r,k,y} = \sum_{\substack{y' \leq y \\ y - y' < LT_{r,k}}} \hat N_{r,k,y'} \;+\; RC_{r,k,y}

Capacity reaching the end of its lifetime, which pays the decommissioning
cost of :eq:`annual-cost`:

.. math::
   :label: decommissioning

   D_{r,k,y} =
   \begin{cases}
     \hat N_{r,k,\,y - LT_{r,k}} & \text{if } y \geq LT_{r,k} \\
     0 & \text{otherwise}
   \end{cases}

Capacity built in the last :math:`TC` years of the horizon is paid but never
becomes available, so the optimum does not build it. A construction time of
0 is not supported: it must be at least 1 (the default).

Implemented in ``ModelVariables._calc_variable_planning`` with
``utility.shift_new_cap``, ``newcap_accumulated`` and ``decomcap``.

Limits on capacity
------------------

*In plain words.* The user can bound, for each technology and year, the
total installed capacity and the new capacity that becomes available. The
defaults (0 and :math:`10^9`) do not bind in most models.

.. math::
   :label: total-capacity-limits

   \underline{TC}_{r,k,y} \leq CAP_{r,k,y} \leq \overline{TC}_{r,k,y}

.. math::
   :label: new-capacity-limits

   \underline{NC}_{r,k,y} \leq \hat N_{r,k,y} \leq \overline{NC}_{r,k,y}

The limits on new capacity apply to the capacity *becoming available* in
year :math:`y`, not to the capacity decided in that year. For the first
:math:`TC` years :math:`\hat N = 0`, so a positive minimum there makes the
model infeasible.

New capacity cannot be negative:

.. math::
   :label: newcap-nonneg

   N_{r,k,y} \geq 0

Implemented in ``TotalCapacityRegional``, ``NewCapacityRegional`` and
``NewCapacityNonneg``.

Limits over all regions
-----------------------

With several regions, the same kinds of limits can be set on the sum over the
regions where a technology exists (sheets ``Min_totalcap_global``,
``Max_totalcap_global``, ``Min_newcap_global``, ``Max_newcap_global``):

.. math::
   :label: global-capacity-limits

   \underline{TC}^{glob}_{k,y} \leq \sum_{r} CAP_{r,k,y} \leq \overline{TC}^{glob}_{k,y},
   \qquad
   \underline{NC}^{glob}_{k,y} \leq \sum_{r} \hat N_{r,k,y} \leq \overline{NC}^{glob}_{k,y}

Implemented in ``TotalCapacityGlobal`` and ``NewCapacityGlobal``. Limits on
user-defined groups of technologies are an option, see
:ref:`optional-capacity-groups`.

.. _modular-capacity:

Modular capacity (MILP)
-----------------------

*In plain words.* Some technologies are only built in units of fixed size,
such as a reactor or a turbine. If ``Modular_cap_unit`` gives a positive unit
size :math:`STEP_{r,k}`, the new capacity is an integer number of units:

.. math::
   :label: modular

   N_{r,k,y} = STEP_{r,k}\; n_{r,k,y}, \qquad n_{r,k,y} \in \mathbb{Z}

(and :math:`n \geq 0` by :eq:`newcap-nonneg`). This turns the problem into a
mixed-integer program, which needs a solver that handles integers (e.g.
Gurobi, HiGHS, CBC). With several regions the unit sizes are given in the
global parameter file. Implemented in ``ModelVariables.create_new_capacity``.
The same unit size is used by the unit commitment of
:ref:`optional-min-output`.
