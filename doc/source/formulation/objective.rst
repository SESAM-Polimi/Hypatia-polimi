.. _objective:

Objective function
==================

Hypatia minimises the total cost of the system. In Planning mode this is the
*net present cost* over the time horizon; in Operation mode, the cost of the
single modelled year. Two other modes use the total emissions instead
(:ref:`objective-modes`).

Costs of the technologies, Planning mode
----------------------------------------

*In plain words.* Every year, each region pays fixed costs on its installed
capacity, variable costs on its production, taxes on its emissions,
decommissioning costs on capacity that retires and a penalty on unmet demand;
taxes and subsidies on investments are paid in the year the investment is
decided. These yearly costs are discounted to the first year. The investments
themselves are spread over the economic lifetime as annuities, each discounted
on its own, and the part of an investment that pays for years after the end of
the horizon is subtracted as a salvage value.

The undiscounted cost of region :math:`r` in year :math:`y` is

.. math::
   :label: annual-cost

   \begin{aligned}
   C_{r,y} = \sum_{k \in K_r \setminus K^{dem}_r} \Big[\;
     & (ITAX_{r,k,y} - ISUB_{r,k,y})\, INV_{r,k,y}\, N_{r,k,y}
     && \text{taxes and subsidies on investments} \\
     + & (1 + FTAX_{r,k,y} - FSUB_{r,k,y})\, FOM_{r,k,y}\, CAP_{r,k,y}
     && \text{fixed costs} \\
     + & VOM_{r,k,y}\, P_{r,k,y}
     && \text{variable costs} \\
     + & DEC_{r,k,y}\, D_{r,k,y}
     && \text{decommissioning} \\
     - & SV_{r,k,y} \;\Big]
     && \text{salvage value} \\
     + \sum_{e \in E} \sum_{k \in K^{em}_r} & ETAX_{r,k,e,y}\, EM_{r,k,e,y}
     && \text{emission taxes} \\
     + \sum_{c \in C} PEN_{r,c} \sum_{t \in T} & q_{r,c,y,t}
     && \text{unmet demand}
   \end{aligned}

where :math:`P_{r,k,y} = \sum_t p_{r,k,y,t}` is the annual production and
:math:`CAP`, :math:`D` and :math:`EM` are defined in :eq:`total-capacity`,
:eq:`decommissioning` and :eq:`emissions`. :math:`PEN_{r,c}` is the cost
of one unit of unmet demand of carrier :math:`c` (sheet
``Unmet_demand_penalty``, default :math:`10^{10}`). With
``ensure_feasibility="No"`` the variables :math:`q` do not appear in any
constraint and are zero at the optimum.

*Investment annuities.* An investment decided in year :math:`y` is paid back
in :math:`EL_{r,k}` equal annuities in the years :math:`y+1, \dots, y+EL_{r,k}`,
computed with the capital recovery factor of the technology's interest rate and
discounted with the discount rate of the year of the decision:

.. math::
   :label: annuities

   CRF_{r,k} = \frac{i_{r,k}\,(1+i_{r,k})^{EL_{r,k}}}{(1+i_{r,k})^{EL_{r,k}} - 1},
   \qquad
   I_{r,k} = \sum_{y \in Y} \sum_{j=1}^{EL_{r,k}}
       \frac{CRF_{r,k}\, INV_{r,k,y}\, N_{r,k,y}}{(1+d_{r,y})^{\,y+j}}

All annuities are counted, also those falling after the last modelled year.

*Salvage value.* For capacity whose technical lifetime extends beyond the
last modelled year :math:`Y_{end} = |Y|-1`, a share of the investment is
subtracted:

.. math::
   :label: salvage

   \begin{aligned}
   SV_{r,k,y} &= SF_{r,k,y}\; INV_{r,k,y}\, N_{r,k,y}, \\
   SF_{r,k,y} &=
   \begin{cases}
     \dfrac{(1+d_{r,y})^{\,LT_{r,k} - Y_{end} - 1 + y} - 1}{(1+d_{r,y})^{LT_{r,k}} - 1}
     \cdot
     \dfrac{1 - (1+i_{r,k})^{-1}}{1 - (1+i_{r,k})^{-EL_{r,k}}}
     \cdot
     \dfrac{1 - (1+d_{r,y})^{-EL_{r,k}}}{1 - (1+d_{r,y})^{-1}}
     & \text{if } y + LT_{r,k} > Y_{end} \\
     0 & \text{otherwise.}
   \end{cases}
   \end{aligned}

The two lifetimes have different roles. The annuities are the financing of
the investment, over the economic lifetime :math:`EL` at the interest rate
:math:`i`, and are all counted. The salvage value gives back the part of that
cost that pays for technical life beyond the horizon: the first factor of
:math:`SF` is the share of the technical lifetime :math:`LT` after
:math:`Y_{end}`, weighted with the discount rate, and the last two equal the
present value of the annuities per unit of investment times
:math:`(1+d_{r,y})/(1+i_{r,k})`. For an investment in the last year with
:math:`i = d`, the net cost is exactly one year of use of the asset, whatever
:math:`EL` (for example 0.062 per unit of investment with :math:`d = 5\%`,
:math:`LT = 30`, :math:`EL = 20`).

.. admonition:: Review note

   When :math:`i \neq d` the factor :math:`(1+d)/(1+i)` makes the salvage value
   smaller than the share of the financed cost (by about 4 % for
   :math:`i = 7\%`, :math:`d = 3\%`). See :ref:`review-notes`.

*Net present cost of the technologies.*

.. math::
   :label: npc-technologies

   NPC^{tech} = \sum_{r \in R} \Big[ \sum_{y \in Y} \frac{C_{r,y}}{(1+d_{r,y})^{\,y}}
       + \sum_{k} I_{r,k} \Big]

The discount factor of year :math:`y` uses that year's rate as a single
factor, :math:`(1+d_{r,y})^{-y}`, not the product of the rates of the
previous years; the two coincide when the discount rate is constant.

Costs of the lines, Planning mode
---------------------------------

With several regions, the lines add their investment, fixed, decommissioning
and variable costs, discounted with the global discount rate:

.. math::
   :label: npc-lines

   \begin{aligned}
   NPC^{lines} = \sum_{y \in Y} \frac{1}{(1+d^{glob}_{y})^{\,y}} \Big[
     & \sum_{l \in L} \sum_{c \in C}
       \big( INV_{l,c,y}\, N_{l,c,y} + FOM_{l,c,y}\, CAP_{l,c,y} + DEC_{l,c,y}\, D_{l,c,y} \big) \\
     & + \sum_{r \in R} \sum_{r' \neq r} \sum_{c \in C} VOM_{l(r,r'),c,y}
       \sum_{t \in T} x^{in}_{r,r',c,y,t} \Big]
   \end{aligned}

Unlike the technologies, the investment in lines is paid in full in the year
of the decision, without annuities or salvage value; the sheets
``Interest_rate`` and ``Line_Economic_life`` of the connections file are not
used. The variable cost applies to the energy imported, before losses.

Operation mode
--------------

Operation models have one year and no investment decisions. The capacity is
the residual capacity, :math:`CAP_{r,k} = RC_{r,k}`, and the cost is not
discounted:

.. math::
   :label: cost-operation

   \begin{aligned}
   TC = \sum_{r \in R} \Big[ & \sum_{k \in K_r \setminus K^{dem}_r}
       \big( (1 + FTAX_{r,k} - FSUB_{r,k})\, FOM_{r,k}\, CAP_{r,k} + VOM_{r,k}\, P_{r,k} \big) \\
     & + \sum_{e \in E} \sum_{k \in K^{em}_r} ETAX_{r,k,e}\, EM_{r,k,e}
       + \sum_{c \in C} PEN_{r,c} \sum_{t \in T} q_{r,c,t} \Big] \\
     & + \sum_{l \in L} \sum_{c \in C} FOM_{l,c}\, CAP_{l,c}
       + \sum_{r \in R} \sum_{r' \neq r} \sum_{c \in C} VOM_{l(r,r'),c} \sum_{t \in T} x^{in}_{r,r',c,t}
   \end{aligned}

.. _objective-modes:

Optimisation modes
------------------

The ``optimization`` argument of :class:`hypatia.Model` selects what is
minimised.

``"Single"``
   Minimise the total cost: :math:`NPC^{tech} + NPC^{lines}` in Planning mode,
   :math:`TC` in Operation mode. Run with :meth:`hypatia.Model.run`.

``"MinEm"``
   Minimise the total emissions over all years, regions and emission types,
   run with :meth:`hypatia.Model.run_MinEm`:

   .. math::
      :label: min-emissions

      EM^{tot} = \sum_{y} \sum_{r} \sum_{e} \sum_{k \in K^{em}_r} EM_{r,k,e,y}

   Emission types are added without weights, so they should be expressed in a
   common unit (e.g. CO2-equivalent). The minimum is found in two steps: first
   :math:`EM^{min} = \min EM^{tot}`, then the cost is minimised subject to
   :math:`EM^{tot} \leq \overline{EM}(EM^{min})`, so that the solution returned
   is the cheapest among those with minimum emissions. The bound has a small
   relative slack,

   .. math::
      :label: emission-bound

      \overline{EM}(E) = E + 10^{-6} \max(|E|, 1),

   because a solver can report the exact minimum as infeasible: the minimum it
   found meets its own tolerances, not the bound exactly.

``"Multi"``
   Trace the trade-off between cost and emissions with the
   :math:`\varepsilon`-constraint method, run with :meth:`hypatia.Model.run_MO`
   and ``number_solutions`` :math:`= n`. Hypatia first minimises the cost (its
   emissions give :math:`EM^{max}`), then the emissions (:math:`EM^{min}`),
   then solves :math:`n` problems

   .. math::
      :label: epsilon-constraint

      \min \; \text{cost} \quad \text{s.t.} \quad EM^{tot} \leq
      \overline{EM}\Big( EM^{max} - i\, \frac{EM^{max} - EM^{min}}{n - 1} \Big),
      \qquad i = 0, \dots, n-1

   with the slack of :eq:`emission-bound`, and writes the results of each into
   its own folder.

Implemented in ``Build._set_regional_objective_planning``,
``_set_regional_objective_operation``, ``_set_lines_objective_planning``,
``_set_lines_objective_operation``, ``_set_final_objective_singlenode``,
``_set_final_objective_multinode``, ``_set_regional_emission_objective``,
``_solve_MinEm``, ``_solve_MO`` and ``emission_bound``; cost components in ``ModelVariables._calc_variable_planning`` and
``_calc_variable_operation``; helper functions ``invcosts``,
``invcosts_annuity``, ``salvage_factor`` and ``fixcosts`` in
``hypatia/utility/utility.py``.
