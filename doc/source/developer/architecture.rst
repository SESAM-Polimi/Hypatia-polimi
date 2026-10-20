.. _dev-architecture:

Structure of the code
=====================

From Excel files to results
---------------------------

.. code-block:: text

   Model(path, mode, ...)                       hypatia/core/main.py
     `- read_settings -> ModelSettings          utility/excel.py, backend/ModelSettings.py
          sets, validation, active constraint classes, parameter templates
   create_data_excels -> parameter templates    utility/excel.py
   read_input_data -> ModelData                 backend/ModelData.py
          parameter tables, checked against the templates
   run -> BuildModel                            backend/Build.py
          |- ModelVariables                     backend/ModelVariables.py
          |    decision variables and derived expressions (capacity, costs, emissions)
          |- constraint classes                 backend/constraints/*.py
          |- objective (cost or emissions)
          `- cvxpy Problem.solve -> results (named tuple of expressions)
   to_csv -> post-processing module             postprocessing/*.py
   Plotter                                      analysis/plots.py

The central idea is that constraints declare the parameters they need.
Each constraint class says which sheets it reads, with their index, columns
and default. ``ModelSettings`` collects these declarations from the active
constraint classes to write the templates (``create_data_excels``) and to
know what to read (``read_input_data``). Enabling an optional constraint
therefore adds its sheets everywhere at once, and a sheet can never be read
by the model without being in the templates, or the reverse. A few parameters
used by the variables rather than by one constraint (costs, lifetimes,
efficiencies) are declared directly in ``ModelSettings``.

Main modules
------------

``core/main.py``
   :class:`~hypatia.Model`, the user interface.
``backend/ModelSettings.py``
   The sets, their validation, and the parameter templates.
``backend/ModelData.py``
   The parameter tables and their checks.
``backend/ModelVariables.py``
   The decision variables (production, use, new capacity, line flows, unmet
   demand, integer variables) and every derived quantity: total and
   decommissioned capacity, costs, emissions, storage state of charge. The
   carrier balance terms are built here too.
``backend/Build.py``
   Collects variables and constraints, defines the objectives
   (:ref:`objective`) and solves.
``backend/constraints/``
   One class per constraint family; ``ConstraintList.py`` lists the core
   ones (always active when they apply) and the optional ones (by name).
``utility/utility.py``
   Shared calculations: capacity accumulation and decommissioning, costs,
   annuities and salvage, storage, capacity groups.
``postprocessing/``
   The ``default`` and ``aggregated`` output formats.
``analysis/plots.py``
   :class:`~hypatia.Plotter`.
``examples/``
   The example models and :func:`~hypatia.load_example`.

The :ref:`implementation-map` gives, for every constraint class, its
equations in the formulation.

A constraint class
------------------

A constraint is a subclass of ``Constraint``
(``backend/constraints/Constraint.py``) with:

``MODES``, ``TOPOLOGY_TYPES``, ``ENSURE_FEASIBILITY``
   Where it applies (by default everywhere). The base class skips the
   constraint, and its parameters, in other models.
``_required_regional_parameters(settings)``, ``_required_global_parameters(settings)``, ``_required_trade_parameters(settings)``
   The sheets it reads: a dictionary from an internal key to the sheet name,
   default value, index and columns. Regional ones are given per region.
``_check(self)``
   Assertions on the variables it needs.
``rules(self)``
   The list of CVXPY constraints. Parameters are in
   ``self.model_data.regional_parameters[region][key]`` (and
   ``global_parameters``, ``trade_parameters``), variables in
   ``self.variables``.

``LandUsageRegional`` is a short, complete example:

.. literalinclude:: ../../../hypatia/backend/constraints/LandUsageRegional.py
   :language: python
   :pyobject: LandUsageRegional

Adding a constraint
-------------------

#. Write the equation first, in the notation of the :ref:`formulation`,
   with the units of every term. Decide whether it is always active (core)
   or optional.
#. Create the class in ``backend/constraints/``, declaring its parameters
   with conservative defaults (a default must leave the model unchanged:
   a limit of :math:`10^{20}`, a share of 0).
#. Register it in ``ConstraintList.py``: in ``CORE_CONSTRAINTS``, or in
   ``OPTIONAL_CONSTRAINTS`` under the name users will pass to ``Model``.
#. Make it build in the fingerprint cases: if no example uses it, extend the
   feature model (``tests/data/make_feature_model.py``). Regenerate the
   fingerprints of the cases that change; the others must not.
#. Add a unit test on a case small enough to check by hand.
#. Document it: the parameters in ``doc/tools/parameter_docs.py``, the
   class in ``CONSTRAINTS`` there, and the equation, with a label and a
   plain-language description, in the formulation pages. For an optional
   constraint, also in :ref:`optional-constraints`.
#. Add a line to ``CHANGELOG.rst``.

Things to know
--------------

* All variables of a region and category are matrices: one row per year
  and time step (``years × time_steps`` rows, year-major), one column per
  technology. Annual quantities have one row per year.
* Years are positions in the list of modelled years (see
  :ref:`review-notes`).
* Carrier names are matched exactly, except for electricity
  (``_terms.is_power_carrier``); a trailing space in a name in Excel makes a
  different carrier.
* Write ``a = a + b``, never ``a += b``, on CVXPY expressions; sum lists of
  terms with ``_terms.sum_terms``.
* The cost objective is computed twice, in ``Build`` (to optimise) and in
  ``ModelVariables`` (to report): a change to one must be made in the other
  (:ref:`review-notes`).
* ``backend/StrData.py`` is mostly unused: only ``create_technology_columns``
  is imported elsewhere.
