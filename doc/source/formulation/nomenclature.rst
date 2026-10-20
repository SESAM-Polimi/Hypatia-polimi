.. _nomenclature:

Nomenclature
============

Units
-----

Hypatia does not convert units. Each quantity has a *dimension*, written with
the generic symbols below, and the user chooses the actual unit (MWh or GWh,
EUR or MEUR) as long as related quantities are consistent: if capacity is in
GW and energy in GWh, ``AnnualProd_perunit_capacity`` is 8760 GWh/GW for a
plant that can run all year. The set files have columns ``Tech_cap_unit``,
``Tech_act_unit`` and ``Carr_unit`` where the chosen units can be recorded;
they are labels for the user and are not read by the model.

.. include:: ../_generated/units.rst

Sets and indices
----------------

.. list-table::
   :widths: 12 70
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`r \in R`
     - Regions (sheet ``Regions`` of ``global.xlsx``). A model with one region has
       no lines and no global or connection parameter files.
   * - :math:`y \in Y`
     - Modelled years, numbered :math:`y = 0, 1, \dots, |Y|-1` in the order of
       the sheet ``Years``. Operation models have one year. Lifetimes,
       construction times and discounting count *positions* in this list, so
       the years are assumed to be consecutive calendar years.
   * - :math:`t \in T`
     - Time steps within a year (sheet ``Timesteps``), the same for every year.
   * - :math:`\tau_t`
     - Fraction of the year covered by time step :math:`t`
       (``Timeslice_fraction``, :math:`\sum_t \tau_t = 1`). The length of the
       step in hours is :math:`8760\,\tau_t`.
   * - :math:`k \in K_r`
     - Technologies of region :math:`r`, each in one category:
       :math:`K^{sup}_r` (Supply), :math:`K^{con}_r` (Conversion),
       :math:`K^{cp}_r` (Conversion_plus), :math:`K^{tra}_r` (Transmission),
       :math:`K^{sto}_r` (Storage) and :math:`K^{dem}_r` (Demand).
   * - :math:`c \in C`
     - Energy carriers (sheet ``Carriers_glob``).
   * - :math:`C^{in}_{r,k},\ C^{out}_{r,k}`
     - Input and output carriers of a technology (sheets ``Carrier_input`` and
       ``Carrier_output`` of the regional set file). All categories have one
       input and one output carrier, except Conversion_plus (several of each),
       Supply (no input) and Demand (no output).
   * - :math:`C^{pow}`
     - Carriers named ``Electricity`` or ``Power`` (any capitalisation, spaces
       around the name ignored). Used by
       the constraints that refer to electricity.
   * - :math:`e \in E`
     - Emission types (sheet ``Emissions``).
   * - :math:`l \in L`
     - Inter-regional lines, one per pair of regions, named ``r1-r2`` with the
       regions in alphabetical order.
   * - :math:`g \in G`
     - Capacity groups (optional; see :ref:`optional-capacity-groups`).

Time series are indexed by :math:`(y, t)`: in the code they are matrices with
:math:`|Y| \cdot |T|` rows, the time steps of year 0 first.

Decision variables
------------------

.. list-table::
   :widths: 18 9 9 50
   :header-rows: 1

   * - Variable
     - Unit
     - Domain
     - Meaning
   * - :math:`p_{r,k,y,t}`
     - [E]
     - :math:`\geq 0`
     - Production (output activity) of technology :math:`k \notin K^{dem}_r` in
       time step :math:`t` of year :math:`y`. For storage, the energy discharged.
   * - :math:`u_{r,k,y,t}`
     - [E]
     - :math:`\geq 0`
     - Use (input activity) of technology :math:`k \notin K^{sup}_r \cup K^{dem}_r`.
       For storage, the energy charged.
   * - :math:`N_{r,k,y}`
     - [P]
     - free [#f1]_
     - New capacity decided in year :math:`y` (Planning only).
   * - :math:`n_{r,k,y}`
     - units
     - integer
     - Number of new modular units, if ``Modular_cap_unit`` is positive:
       then :math:`N_{r,k,y} = STEP_{r,k}\, n_{r,k,y}`.
   * - :math:`x^{in}_{r,r',c,y,t}`
     - [E]
     - :math:`\geq 0`
     - Energy of carrier :math:`c` imported into :math:`r` from :math:`r'`,
       before line losses (several regions only).
   * - :math:`x^{out}_{r,r',c,y,t}`
     - [E]
     - :math:`\geq 0`
     - Energy of carrier :math:`c` exported from :math:`r` to :math:`r'`.
   * - :math:`N_{l,c,y}`
     - [P]
     - :math:`\geq 0`
     - New capacity of line :math:`l` for carrier :math:`c` decided in year
       :math:`y` (Planning, several regions).
   * - :math:`q_{r,c,y,t}`
     - [E]
     - :math:`\geq 0`
     - Unmet demand of carrier :math:`c`. It enters the balance only when the
       model is created with ``ensure_feasibility="Yes"``.
   * - :math:`b_{r,k,y,t}`
     - --
     - :math:`\{0,1\}`
     - Technology on (1) or off (0); option ``min_tech_power_output``.
   * - :math:`n^{on}_{r,k,y,t}`
     - units
     - integer :math:`\geq 0`
     - Number of modular units online; option ``min_tech_power_output`` with
       modular capacity, Planning only.
   * - :math:`s_{r,k,y,t}`
     - --
     - :math:`\{0,1\}`
     - Storage may discharge (1) or charge (0); option ``boolean_for_storage``.

.. [#f1] Non-negativity of new capacity is imposed by a constraint,
   :eq:`newcap-nonneg`, rather than in the variable definition.

Derived quantities
------------------

The following expressions are defined once and used in several constraints;
their equations are given in the sections indicated.

.. list-table::
   :widths: 18 9 55 18
   :header-rows: 1

   * - Symbol
     - Unit
     - Meaning
     - Defined in
   * - :math:`\hat N_{r,k,y}`
     - [P]
     - New capacity becoming available in year :math:`y` (after construction)
     - :eq:`real-newcap`
   * - :math:`CAP_{r,k,y}`
     - [P]
     - Total installed capacity
     - :eq:`total-capacity`
   * - :math:`D_{r,k,y}`
     - [P]
     - Capacity reaching the end of its technical lifetime
     - :eq:`decommissioning`
   * - :math:`A_{r,k,y,t}`
     - [E]
     - Production available from the installed capacity in a time step
     - :eq:`available-production`
   * - :math:`P_{r,k,y}`
     - [E]
     - Annual production, :math:`\sum_t p_{r,k,y,t}`
     - --
   * - :math:`PROD_{r,c,y,t},\ USE_{r,c,y,t}`
     - [E]
     - Production and use of carrier :math:`c` by all technologies of a region
     - :eq:`carrier-production`, :eq:`carrier-use`
   * - :math:`SOC_{r,k,y,t}`
     - [E]
     - Net energy stored since the start of the current storage cycle
     - :eq:`soc`
   * - :math:`EM_{r,k,e,y}`
     - [Em]
     - Annual emissions released
     - :eq:`emissions`

Parameters
----------

Every parameter, with the sheet it is read from, is listed below; the
:ref:`input-reference` gives defaults, table layouts and the configurations in
which each sheet is required. Symbols in equations sometimes drop indices that
are clear from context.

.. include:: ../_generated/nomenclature_parameters.rst
