.. _emissions:

Emissions
=========

Emission accounting
-------------------

*In plain words.* Supply, conversion and Conversion_plus technologies emit
in proportion to their annual production. Part of the emissions can be
captured. A technology with a negative emission factor removes emissions from
the atmosphere: its emissions count as negative, and a positive emission tax
then becomes a credit.

For :math:`k \in K^{sup}_r \cup K^{con}_r \cup K^{cp}_r` and each emission
type :math:`e`, with :math:`P_{r,k,y}` the annual production (the activity, for
Conversion_plus technologies):

.. math::
   :label: emissions

   \begin{aligned}
   \text{if } EF_{r,k,e,0} \geq 0: \quad
     & EM_{r,k,e,y} = (1 - \eta^{cap}_{r,k,e,y})\; EF_{r,k,e,y}\; P_{r,k,y},
     && CAPT_{r,k,e,y} = \eta^{cap}_{r,k,e,y}\; EF_{r,k,e,y}\; P_{r,k,y} \\
   \text{if } EF_{r,k,e,0} < 0: \quad
     & EM_{r,k,e,y} = EF_{r,k,e,y}\; P_{r,k,y} \;(\leq 0),
     && CAPT_{r,k,e,y} = 0
   \end{aligned}

:math:`EM` enters the emission taxes of :eq:`annual-cost` and the emission
limits below; :math:`CAPT` is the captured emission, limited by
:eq:`captured-emissions`. Results also report the emissions removed by
negative-emission technologies, :math:`-EF\, P`.

.. admonition:: Review note

   The branch is chosen with the emission factor of the *first* year
   (:math:`EF_{r,k,e,0}`) and then applied to all years. A technology whose
   factor changes sign over the horizon is not handled correctly. See
   :ref:`review-notes`.

Implemented in ``ModelVariables._calc_emission_variables``.

Emission limits
---------------

*In plain words.* The annual emissions of each type can be capped per
region, for the technologies that produce electricity in a region, and over
all regions.

.. math::
   :label: emission-cap-regional

   \sum_{k} EM_{r,k,e,y} \leq \overline{EM}_{r,e,y}

.. math::
   :label: emission-cap-power

   \sum_{k:\; C^{out}_{r,k} \cap C^{pow} \neq \emptyset} EM_{r,k,e,y} \leq \overline{EM}^{pow}_{r,e,y}

.. math::
   :label: emission-cap-global

   \sum_{r} \sum_{k} EM_{r,k,e,y} \leq \overline{EM}^{glob}_{e,y}

The regional cap is read from the column ``<type> Global Cap`` of
``Emission_cap_annual`` (the name is historical: the cap is regional), the
power cap from ``<type> Power Cap`` of ``Emission_cap_annual_power``, the
global cap from ``<type> Global Emission Cap`` of ``Glob_emission_cap_annual``.
The defaults (:math:`10^{30}`) do not bind. Implemented in
``EmissionCapRegional`` and ``EmissionCapGlobal`` (several regions).

Captured emissions
------------------

*In plain words.* The captured emissions of a region cannot exceed the
capacity available to store them.

.. math::
   :label: captured-emissions

   \sum_{k} CAPT_{r,k,e,y} \leq \overline{CS}_{r,e,y}

Read from the column ``<type> Stored Cap`` of ``Emission_Storage_annual``.
Implemented in ``EmissionConsumedRegional``.
