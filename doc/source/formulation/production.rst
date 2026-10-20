.. _production-limits:

Production, renewable shares and land use
=========================================

Annual production of a technology
---------------------------------

*In plain words.* For each technology and output carrier the user can bound
the annual production, in absolute terms or as a share of the region's total
production of that carrier.

The annual production of carrier :math:`c` by technology :math:`k` is
:math:`P^{c}_{r,k,y} = \sum_t p_{r,k,y,t}` for technologies with one output,
and :math:`P^{c}_{r,k,y} = \sum_t CRO_{r,k,c,y,t}\, p_{r,k,y,t}` for
Conversion_plus technologies. With
:math:`PRODA_{r,c,y} = \sum_t PROD_{r,c,y,t}` the region's annual production
of the carrier:

.. math::
   :label: production-limits

   \underline{PR}_{r,k,y} \leq P^{c}_{r,k,y} \leq \overline{PR}_{r,k,y},
   \qquad
   \underline{SR}_{r,k,y}\, PRODA_{r,c,y} \leq P^{c}_{r,k,y} \leq \overline{SR}_{r,k,y}\, PRODA_{r,c,y}

for supply, conversion and Conversion_plus technologies and each of their
output carriers. Conversion_plus technologies read their limits from the
``*_Conv_plus`` sheets, which have one column per (technology, carrier).
Transmission and storage technologies are not limited. Implemented in
``AnnualProductionRegional``.

With several regions, the same four limits can be set on the production of
each technology and carrier summed over all regions, as an absolute value or
as a share of the total production of the carrier over all regions (sheets
``Min_prod_global``, ``Max_prod_global``, ``Min_prod_global_share``,
``Max_prod_global_share``). These also apply to transmission technologies.
Implemented in ``AnnualProductionGlobal``.

.. _renewable-shares:

Renewable shares
----------------

*In plain words.* The user marks which technologies are renewable and can
require a minimum renewable share, of electricity production and of total
final demand.

Let :math:`RES_{r,k} = 1` mark a renewable technology (sheet
``Renewable_tech_selection``; storage, transmission and demand technologies
are never counted).

*Electricity.* The renewable production of power carriers (:math:`C^{pow}`,
carriers named ``Electricity`` or ``Power``) must be at least a share of the
region's production of those carriers:

.. math::
   :label: renewable-electricity

   \sum_{k: RES_{r,k}=1} \; \sum_{c \in C^{pow} \cap C^{out}_{r,k}} P^{c}_{r,k,y}
   \;\geq\; \underline{RS}^{el}_{r,y} \sum_{c \in C^{pow}} PRODA_{r,c,y}

The right-hand side includes all producers of the power carriers, storage
discharge included. Carriers with other names (for example
``Final_electricity``) are not power carriers here.

*All energy.* The annual activity of renewable technologies must be at least
a share of the region's total final demand, all carriers added together:

.. math::
   :label: renewable-total

   \sum_{k: RES_{r,k}=1} P_{r,k,y} \;\geq\; \underline{RS}_{r,y} \sum_{c} \sum_{t} DEM_{r,c,y,t}

Both sides add quantities of different carriers, so they are meaningful only
if all carriers are in the same energy unit. A renewable supply technology
and a renewable conversion technology that uses its output are both counted.

Implemented in ``RenewableProductionRegional``. With several regions,
``RenewableProductionGlobal`` imposes the same two constraints on the sums
over all regions, with the renewable flags and shares of the global file.

Land use
--------

*In plain words.* Each unit of capacity occupies some land; the land used by
a technology can be limited per region and, with several regions, in total.

.. math::
   :label: land-use

   LU_{r,k,y}\, CAP_{r,k,y} \leq \overline{LA}_{r,k,y},
   \qquad
   \sum_{r} LU_{r,k,y}\, CAP_{r,k,y} \leq \overline{LA}^{glob}_{k,y}

Planning mode only. Implemented in ``LandUsageRegional`` and
``LandUsageGlobal``.
