Hypatia
=======

Hypatia is an open-source energy system model written in Python. It finds the
cheapest way to meet the energy demand of one or more regions, hour by hour
or with coarser time steps, either for a single year with the capacity given
(*Operation* mode) or over many years, deciding which capacity to build and
when (*Planning* mode). It represents electricity, heat, fuels and any other
energy carrier together, the technologies that produce, convert, transport
and store them, and the lines that connect regions. The model is written with
`CVXPY <https://www.cvxpy.org>`_ and its inputs are Excel files.

The SESAM group of the Department of Energy of Politecnico di Milano develops
Hypatia and uses it for research on national energy scenarios and for
teaching. It was created at Politecnico di Milano by Negar Namazifard and
Mohammad Amin Tahavori (see :ref:`history`).

.. list-table::
   :widths: 30 70

   * - :ref:`installation`
     - Install Hypatia and run a first model in a few minutes.
   * - :ref:`user-guide`
     - How a model is built from Excel files, every input sheet, the options,
       running and reading the results.
   * - :ref:`tutorials`
     - Two complete examples, one planning and one operation model.
   * - :ref:`formulation`
     - The mathematical model: every variable, parameter and equation, with
       units, defaults and the code that implements it.
   * - :ref:`developer-guide`
     - How to contribute: workflow, tests and the structure of the code.

.. toctree::
   :caption: Getting started
   :maxdepth: 1
   :hidden:

   getting_started/installation
   getting_started/first_run

.. toctree::
   :caption: User guide
   :maxdepth: 2
   :hidden:

   user_guide/concepts
   user_guide/building_a_model
   user_guide/input_reference
   user_guide/optional_constraints
   user_guide/unit_commitment
   user_guide/running_and_results

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   tutorials/planning
   tutorials/operation

.. toctree::
   :caption: Model formulation
   :maxdepth: 2
   :hidden:

   formulation/index

.. toctree::
   :caption: Developer guide
   :maxdepth: 1
   :hidden:

   developer/contributing
   developer/testing
   developer/architecture

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   reference/api
   reference/changelog
   about/history
   about/license
