.. _balances:

Energy balances, conversion and availability
============================================

Production and use of each carrier
----------------------------------

Technologies are connected through carriers: what a technology produces in a
carrier is available to the technologies that use that carrier. For each
region, carrier and time step, Hypatia adds up the production and the use of
the carrier by all technologies. Conversion_plus technologies split their
activity among several carriers with the carrier ratios.

.. math::
   :label: carrier-production

   PROD_{r,c,y,t} = \sum_{\substack{k \notin K^{cp}_r \\ c \in C^{out}_{r,k}}} p_{r,k,y,t}
     + \sum_{\substack{k \in K^{cp}_r \\ c \in C^{out}_{r,k}}} CRO_{r,k,c,y,t}\; p_{r,k,y,t}

.. math::
   :label: carrier-use

   USE_{r,c,y,t} = \sum_{\substack{k \notin K^{sup}_r \cup K^{dem}_r \cup K^{cp}_r \\ c \in C^{in}_{r,k}}} u_{r,k,y,t}
     + \sum_{\substack{k \in K^{cp}_r \\ c \in C^{in}_{r,k}}} CRI_{r,k,c,y,t}\; u_{r,k,y,t}

Storage technologies appear in both sums: discharging is production of their
output carrier, charging is use of their input carrier. The final demand of a
carrier is the sum of the demand of the demand technologies that take it as
input:

.. math::

   DEM_{r,c,y,t} = \sum_{\substack{k \in K^{dem}_r \\ c \in C^{in}_{r,k}}} DEM_{r,k,y,t}

Imports and exports (models with several regions) are

.. math::
   :label: imports-exports

   IMP_{r,c,y,t} = \sum_{r' \neq r} \eta_{l,c,y}^{\,L_{l,c,y}/100}\; x^{in}_{r,r',c,y,t},
   \qquad
   EXP_{r,c,y,t} = \sum_{r' \neq r} x^{out}_{r,r',c,y,t}

where :math:`l` is the line between :math:`r` and :math:`r'`: line losses
reduce the energy received, with the efficiency given per 100 units of length.

Implemented in ``ModelVariables._balance_``.

.. _carrier-balance:

Carrier balance
---------------

*In plain words.* In every region, carrier and time step, what is produced
and imported equals what is used, exported and consumed by the final demand.
The balance is an equality, so a technology can produce less than it could
(see :eq:`availability-step`) but nothing can be produced and thrown away.

.. math::
   :label: balance

   PROD_{r,c,y,t} + IMP_{r,c,y,t} - USE_{r,c,y,t} - EXP_{r,c,y,t} - DEM_{r,c,y,t} = 0
   \qquad \forall\, r, c, y, t

With ``ensure_feasibility="Yes"`` the unmet demand is added to the supply
side, so that any demand can be met at the penalty cost of
:eq:`annual-cost`:

.. math::
   :label: balance-unmet

   PROD_{r,c,y,t} + IMP_{r,c,y,t} + q_{r,c,y,t} - USE_{r,c,y,t} - EXP_{r,c,y,t} - DEM_{r,c,y,t} = 0

Implemented in ``Balance`` and ``BalanceUnMetDemand``.

Trade balance
-------------

*In plain words.* What region :math:`r` imports from :math:`r'` is what
:math:`r'` exports to :math:`r`; the losses are applied on the importing side
in :eq:`imports-exports`.

.. math::
   :label: trade-balance

   x^{in}_{r,r',c,y,t} = x^{out}_{r',r,c,y,t} \qquad \forall\, r \neq r', c, y, t

Applies to models with several regions. Implemented in ``TradeBalance``.

Conversion efficiency
---------------------

*In plain words.* Conversion, Conversion_plus and transmission
technologies produce, in each time step, their input times their efficiency.

.. math::
   :label: efficiency

   p_{r,k,y,t} = \eta_{r,k,y}\; u_{r,k,y,t}
   \qquad \forall\, k \in K^{con}_r \cup K^{cp}_r \cup K^{tra}_r

For Conversion_plus technologies :math:`p` and :math:`u` are activities; the
carriers produced and used follow from the carrier ratios in
:eq:`carrier-production` and :eq:`carrier-use`. Storage has its own
efficiencies (:ref:`storage`). Implemented in ``TechEfficency``.

.. _availability:

Production limits from capacity
-------------------------------

*In plain words.* A technology cannot produce more in a time step than its
capacity allows, given the resource available in that time step (sun, wind,
or 1 for dispatchable plants). Over the year, its production must stay
between a minimum and a maximum share of what it could produce, scaled by its
availability.

The production available from the installed capacity in a time step is

.. math::
   :label: available-production

   A_{r,k,y,t} = CAP_{r,k,y}\; CF^{res}_{r,k,y,t}\; APC_{r,k}\; \tau_t

and for every technology except storage and demand

.. math::
   :label: availability-step

   p_{r,k,y,t} \leq A_{r,k,y,t} \qquad \forall\, t

.. math::
   :label: availability-annual

   AV_{r,k,y}\, CF^{min}_{r,k,y} \sum_{t} A_{r,k,y,t}
   \;\leq\; \sum_{t} p_{r,k,y,t} \;\leq\;
   AV_{r,k,y}\, CF^{max}_{r,k,y} \sum_{t} A_{r,k,y,t}

The defaults (:math:`CF^{res} = AV = CF^{max} = 1`, :math:`CF^{min} = 0`)
leave only the capacity limit. In Operation mode the capacity is the
residual capacity. Implemented in ``ResourceTechAvailability``; the
available production in ``utility.available_resource_prod``.
