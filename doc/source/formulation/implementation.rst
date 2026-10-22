.. _implementation-map:

Implementation map
==================

Each constraint of the model is built by one class in
``hypatia/backend/constraints``. The table below is generated from the code
at every build: the list of classes and where each applies come from
``ConstraintList.py`` and the classes' ``MODES``, ``TOPOLOGY_TYPES`` and
``ENSURE_FEASIBILITY`` attributes, so it is always the list the model uses.

.. include:: ../_generated/implementation_map.rst

The variables and derived quantities are built in
``hypatia/backend/ModelVariables.py``, the objective in
``hypatia/backend/Build.py``, and shared calculations (capacity accumulation,
costs, storage) in ``hypatia/utility/utility.py``. How to add a constraint is
explained in :ref:`dev-architecture`.
