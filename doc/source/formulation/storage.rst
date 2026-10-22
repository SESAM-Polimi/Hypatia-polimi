.. _storage:

Storage
=======

Storage technologies charge (use :math:`u`) and discharge (produce :math:`p`)
the same carrier or two different ones, and their capacity :math:`CAP` is an
energy capacity. The constraints below are written for hourly time steps
(:math:`8760\,\tau_t = 1` hour); with longer steps the factor
:math:`8760\,\tau_t` scales the energy limits by the length of the step, which
is not the intended use (see :ref:`review-notes`).

State of charge
---------------

*In plain words.* The year is divided into storage cycles of
:math:`T^{cyc}` time steps (by default one cycle per year). The state of
charge is the energy stored since the start of the current cycle: charging
adds energy times the charging efficiency, discharging removes energy divided
by the discharging efficiency.

For time step :math:`t` of year :math:`y`, belonging to the cycle that starts
at :math:`t_0`:

.. math::
   :label: soc

   SOC_{r,k,y,t} = \sum_{t'=t_0}^{t} \Big( \eta^{ch}_{r,k,y}\, u_{r,k,y,t'}
       - \frac{p_{r,k,y,t'}}{\eta^{dis}_{r,k,y}} \Big)

The number of time steps in a year must be a multiple of :math:`T^{cyc}`,
otherwise Hypatia stops with an error. Implemented in
``ModelVariables._calc_variable_storage_SOC`` with
``utility.storage_state_of_charge``.

Energy limits
-------------

*In plain words.* The stored energy stays between a minimum and the usable
part of the capacity, given by the depth of discharge.

.. math::
   :label: soc-limits

   SOC^{min}_{r,k,y}\; DOD_{r,k}\; CAP_{r,k,y}\; 8760\,\tau_t
   \;\leq\; SOC_{r,k,y,t} \;\leq\;
   DOD_{r,k}\; CAP_{r,k,y}\; 8760\,\tau_t

Implemented in ``StorageMaxMinChange``.

Charging and discharging power
------------------------------

*In plain words.* A storage can charge or discharge its full capacity in
:math:`T^{ch}` or :math:`T^{dis}` hours, which limits the energy that flows in
or out in one time step. Over each year, the storage ends where it started.

.. math::
   :label: storage-flows

   u_{r,k,y,t} \leq \frac{AV_{r,k,y}\; CAP_{r,k,y}\; 8760\,\tau_t}{T^{ch}_{r,k}},
   \qquad
   p_{r,k,y,t} \leq \frac{AV_{r,k,y}\; CAP_{r,k,y}\; 8760\,\tau_t}{T^{dis}_{r,k}}

.. math::
   :label: storage-annual-cycle

   SOC_{r,k,y,\,t_{first}} = SOC_{r,k,y,\,t_{last}}

where :math:`t_{first}` and :math:`t_{last}` are the first and last time steps
of the year. Implemented in ``StorageMaxFlowInOut``.

Cycle boundary
--------------

*In plain words.* At the end of every cycle the stored energy equals the
minimum state of charge.

.. math::
   :label: storage-cyclic

   SOC_{r,k,y,\,t_{end}(m)} = SOC^{min}_{r,k,y}\; CAP_{r,k,y}\; 8760\,\tau_{t_{end}(m)}
   \qquad \text{for every cycle } m

where :math:`t_{end}(m)` is the last time step of cycle :math:`m`.
Implemented in ``StorageCyclicBoundary``.

.. admonition:: Review note

   Because :eq:`soc` counts from zero at the start of each cycle, the storage
   starts every cycle empty, while :eq:`storage-cyclic` makes it end at the
   minimum state of charge, so the two ends of a cycle differ unless
   :math:`SOC^{min} = 0`. Moreover :eq:`storage-cyclic` uses
   :math:`SOC^{min}\, CAP` where :eq:`soc-limits` uses
   :math:`SOC^{min}\, DOD\, CAP`, and with one cycle per year
   :eq:`storage-annual-cycle` then fixes the first time step to the same
   value. See :ref:`review-notes`.

No simultaneous charging and discharging
----------------------------------------

With the option ``boolean_for_storage`` a binary variable decides, in each
time step, whether a storage that charges and discharges the same carrier may
discharge (:math:`s = 1`) or charge (:math:`s = 0`). See
:ref:`optional-boolean-storage`.
