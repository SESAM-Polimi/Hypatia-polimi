.. _formulation-optional:

Optional constraints
====================

These constraints are added only when the model asks for them:
``Model(..., optional_constraints=[...])`` for the first five, and the
presence of the capacity group sheets for the last. Enabling an option also
adds its sheets to the parameter templates (see :ref:`optional-constraints`
for how to use them).

.. _optional-chp:

CHP operating range (``chp_operating_range``)
---------------------------------------------

*In plain words.* A combined heat and power plant is represented by three
conversion technologies: the CHP unit itself, a power branch and a heat
branch. The constraint keeps the heat and power outputs inside the operating
region of the plant: the heat cannot exceed :math:`\gamma` times the activity
of the CHP unit, and the power must be at least a given fraction of the heat.

For each row :math:`(k^{chp}, k^{el}, k^{th}, \gamma)` of the sheet
``CHP_variable_techs_selection`` whose three technologies are conversion
technologies of region :math:`r`:

.. math::
   :label: chp

   p_{r,k^{th},y,t} \leq \gamma\; p_{r,k^{chp},y,t},
   \qquad
   \Big( \frac{\eta^{el}_{r,y}}{\gamma} - \frac{\eta^{el}_{r,y}}{\eta^{th}_{r,y}} \Big)\, p_{r,k^{th},y,t}
   \leq p_{r,k^{el},y,t}

where :math:`\eta^{el}_{r,y}` and :math:`\eta^{th}_{r,y}` are the efficiencies
of the power and heat branches (``Tech_efficiency``) in year :math:`y`. Rows whose technologies are not in the region are skipped, so the
default template rows (names ``0``) have no effect. Implemented in
``CHPOperatingRange``.

.. _optional-ramp:

Ramp limits (``production_ramp``)
---------------------------------

*In plain words.* A plant cannot change its output faster than a given
fraction of its available output per minute. Within one hour, the energy it
can add or remove follows a triangle (slow ramps) or a trapezoid (fast
ramps). Meant for hourly time steps; Hypatia prints a reminder.

For supply, conversion and Conversion_plus technologies with a rate
:math:`\rho < 1` (1, the default, means no limit), let
:math:`\Delta p_{r,k,y,t} = p_{r,k,y,t} - p_{r,k,y,t-1}` and

.. math::
   :label: ramp-amount

   R(\rho, A) =
   \begin{cases}
     30\,\rho\; A & \text{if } \rho \leq 1/60 \\
     \Big(1 - \dfrac{1}{120\,\rho}\Big)\, A & \text{if } \rho > 1/60
   \end{cases}

with :math:`A = A_{r,k,y,t}` the available production of
:eq:`available-production`. Then

.. math::
   :label: ramp

   -R(\rho^{down}_{r,k}, A_{r,k,y,t}) \leq \Delta p_{r,k,y,t} \leq R(\rho^{up}_{r,k}, A_{r,k,y,t})

for all time steps except the first one of the model. The two expressions of
:math:`R` meet at :math:`\rho = 1/60` (:math:`R = A/2`). Implemented in
``ProductionRamp``.

.. admonition:: Review note

   From the second year on, the first time step of each year is compared with
   zero instead of the last time step of the previous year
   (:math:`\Delta p_{r,k,y,0} = p_{r,k,y,0}`), which limits the production of
   the first hour of every year to the ramp amount.

.. _optional-min-output:

Minimum output and unit commitment (``min_tech_power_output``)
--------------------------------------------------------------

*In plain words.* When a plant runs, it must produce at least a minimum
fraction of its available output; it may also be off. For modular
technologies in Planning mode the model counts how many units are online in
each time step, each with its own minimum.

For supply, conversion and Conversion_plus technologies with a minimum
:math:`\underline{p}_{r,k} > 0` (technologies with 0 are not constrained):

*Modular technologies* (Planning mode, :math:`STEP_{r,k} > 0`), with the
production of one unit
:math:`U_{r,k,y,t} = STEP_{r,k}\, CF^{res}_{r,k,y,t}\, APC_{r,k}\, \tau_t`:

.. math::
   :label: unit-commitment

   STEP_{r,k}\; n^{on}_{r,k,y,t} \leq CAP_{r,k,y},
   \qquad
   \underline{p}_{r,k}\; U_{r,k,y,t}\; n^{on}_{r,k,y,t}
   \;\leq\; p_{r,k,y,t} \;\leq\; U_{r,k,y,t}\; n^{on}_{r,k,y,t},
   \qquad n^{on}_{r,k,y,t} \in \mathbb{Z}_{\geq 0}

*Other technologies*, with an on/off variable and :math:`M = 10^{15}`:

.. math::
   :label: min-output

   p_{r,k,y,t} \leq M\, b_{r,k,y,t},
   \qquad
   p_{r,k,y,t} \geq \underline{p}_{r,k}\, A_{r,k,y,t} - M\,(1 - b_{r,k,y,t}),
   \qquad b_{r,k,y,t} \in \{0, 1\}

Either form makes the problem a mixed-integer program. The modular form
(contributed by Luigi Malco Rossi, October 2026) is the one to prefer: it has
no big-M constant and a clear meaning per unit. Implemented in
``MinTechPowerOutput`` and ``ModelVariables.create_units_online``.

.. _optional-electrolysis:

Green hydrogen (``electrolysis_consumption``)
---------------------------------------------

*In plain words.* Electrolysers may only use as much electricity, over the
year, as the region produces from renewable technologies.

In each region with a technology named exactly ``Electrolysis``:

.. math::
   :label: electrolysis

   \sum_{t} u^{el}_{r,\mathrm{Electrolysis},y,t}
   \;\leq\; \sum_{k: RES_{r,k}=1} P^{\mathrm{Electricity}}_{r,k,y}

where :math:`u^{el}` is the electricity used by the electrolyser
(:math:`CRI \cdot u` if it is a Conversion_plus technology) and the right side
is the annual production of the carrier named exactly ``Electricity`` by
renewable technologies. Regions without an electrolyser get no constraint.
Implemented in ``ElectrolysisConsumption``.

.. _optional-boolean-storage:

No simultaneous charge and discharge (``boolean_for_storage``)
--------------------------------------------------------------

*In plain words.* A storage that charges and discharges the same carrier
cannot do both in the same time step.

.. math::
   :label: boolean-storage

   p_{r,k,y,t} \leq M\, s_{r,k,y,t}, \qquad
   u_{r,k,y,t} \leq M\, (1 - s_{r,k,y,t}), \qquad
   s_{r,k,y,t} \in \{0,1\}, \quad M = 10^{20}

for storage technologies with the same input and output carrier.
Implemented in ``BooleanForStorage``.

.. admonition:: Review note

   The last time step of each year is left out of the constraint, and
   :math:`M = 10^{20}` is large enough to cause numerical trouble in most
   solvers (a constant of the order of the largest possible flow would be
   safer).

.. _formulation-capacity-groups:

Capacity groups
---------------

*In plain words.* The user defines groups of technologies (for example "all
renewables" or "all nuclear") and limits the total or new capacity of each
group, summed over technologies and regions.

.. math::
   :label: capacity-groups

   \sum_{r} \sum_{k \in G_g \cap K_r} CAP_{r,k,y} \leq \overline{TC}_{g,y},
   \qquad
   \sum_{r} \sum_{k \in G_g \cap K_r} \hat N_{r,k,y} \leq \overline{NC}_{g,y}

Planning mode only; added when the sheets ``Capacity_group_membership`` and
``Max_totalcap_group`` or ``Max_newcap_group`` are present. Implemented in
``TotalCapacityGroup`` and ``NewCapacityGroup``.
