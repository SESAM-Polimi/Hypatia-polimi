.. _lines:

Inter-regional lines
====================

Models with several regions have one line :math:`l` for each pair of regions,
with a separate capacity for each carrier. Energy can flow both ways; the
flows and the trade balance are in :ref:`balances` (:eq:`imports-exports`,
:eq:`trade-balance`).

Line capacity
-------------

*In plain words.* In Planning mode new line capacity is decided every year,
becomes available after the construction time and retires after its
lifetime, exactly as for technologies. In Operation mode the line capacity is
the residual capacity.

.. math::
   :label: line-capacity

   \hat N_{l,c,y} = N_{l,c,\,y-TC_{l,c}}, \qquad
   CAP_{l,c,y} = \sum_{\substack{y' \leq y \\ y - y' < LT_{l,c}}} \hat N_{l,c,y'} + RC_{l,c,y}, \qquad
   D_{l,c,y} = \hat N_{l,c,\,y - LT_{l,c}}

(with :math:`\hat N` and :math:`D` zero when the index falls before the first
year). Implemented in ``ModelVariables._calc_variable_planning_line``.

Limits on line capacity
-----------------------

.. math::
   :label: line-capacity-limits

   \underline{TC}_{l,c,y} \leq CAP_{l,c,y} \leq \overline{TC}_{l,c,y},
   \qquad
   \underline{NC}_{l,c,y} \leq N_{l,c,y} \leq \overline{NC}_{l,c,y}

Unlike the technologies (:eq:`new-capacity-limits`), the limits on new line
capacity apply to the capacity *decided* in a year. Planning mode only.
Implemented in ``LineTotalCapacity`` and ``LineNewCapacity``.

Line flows
----------

*In plain words.* In each time step, the energy a region imports through a
line cannot exceed what the line capacity can carry in that time step; over
the year, it cannot exceed the line's capacity factor times its full-load
energy.

For region :math:`r` importing from :math:`r'` through line :math:`l`:

.. math::
   :label: line-flows

   x^{in}_{r,r',c,y,t} \leq CAP_{l,c,y}\; APC_{l,c}\; \tau_t,
   \qquad
   \sum_{t} x^{in}_{r,r',c,y,t} \leq CF_{l,c,y}\; CAP_{l,c,y}\; APC_{l,c}

The limits apply to each direction separately, so in the same time step a
line can carry its full capacity both ways. Implemented in
``LineAvailability``.
