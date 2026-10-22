.. image:: https://github.com/SESAM-Polimi/Hypatia-polimi/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/SESAM-Polimi/Hypatia-polimi/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://www.apache.org/licenses/LICENSE-2.0
   :alt: Licence: Apache 2.0

.. image:: https://raw.githubusercontent.com/SESAM-Polimi/MARIO/767d2c0e9e42ae0b6acf7c3a1cc379d7bcd367fa/doc/source/_static/images/polimi.svg
   :width: 200
   :align: right

Hypatia
=======

Hypatia is an open-source energy system model written in Python. It finds the
cheapest way to meet the energy demand of one or more regions, either for a
single year with the installed capacity given (*Operation* mode) or over many
years, deciding which capacity to build and when (*Planning* mode). It covers
electricity, heat, fuels and any other energy carrier, the technologies that
produce, convert, transport and store them, and the lines between regions,
with any time resolution from one step per year to hourly. Inputs are Excel
files and the optimisation model is written with `CVXPY <https://www.cvxpy.org>`_.

This repository is the version of Hypatia developed and used by the SESAM
group of the Department of Energy of Politecnico di Milano, for research on
national energy scenarios and for teaching.

What it does
------------

* Least-cost dispatch and capacity expansion, with annuities, salvage value,
  taxes and subsidies, and lifetimes per technology.
* Six technology categories: supply, conversion, conversion with several
  inputs and outputs, transmission, storage and demand.
* Several regions, with trade of every carrier over lines that have their
  own capacity, losses and costs.
* Emission caps and taxes, renewable shares, land use, limits on capacity
  and production per technology, per region, per group of technologies or
  for the whole system.
* Minimisation of emissions, and a series of solutions trading cost against
  emissions.
* Optional constraints: variable operating range of combined heat and power
  plants, ramp limits, minimum output with unit commitment, green hydrogen,
  storage that cannot charge and discharge at once.

Installation
------------

Hypatia needs Python 3.10 or later and is tested with Python 3.11.

.. code-block:: bash

   conda create -n hypatia python=3.11
   conda activate hypatia
   git clone https://github.com/SESAM-Polimi/Hypatia-polimi.git
   cd Hypatia-polimi
   pip install -e .

This installs CVXPY with HiGHS, an open-source solver for linear and
mixed-integer problems. Gurobi and CPLEX can be used as well if installed.

.. note::

   The package ``hypatia-py`` on PyPI is an earlier release of Hypatia, not
   this version: install from this repository.

Environments created for earlier versions (Python 3.8, CVXPY 1.1.18) do not
work with the current code; the documentation explains how to reproduce
results obtained with them.

A first run
-----------

.. code-block:: python

   from hypatia import load_example

   model = load_example("Atlantis")
   model.run(solver="highs")
   model.to_csv(path="results", postprocessing_module="aggregated")

``available_examples()`` lists the twelve example models, and
``download_example(name, folder)`` copies one to a folder of your own to
change it.

Documentation
-------------

The documentation is in the ``doc`` folder. It covers installation, how to
build a model from the Excel files, every input sheet with its unit and
default, the optional constraints, the results, two tutorials, and the
complete mathematical formulation, with each equation tied to the code that
implements it. To build it:

.. code-block:: bash

   pip install -r doc/requirements.txt
   cd doc
   sphinx-build -b html source build/html

and open ``doc/build/html/index.html``.

Contributing
------------

Bug reports and contributions are welcome as issues and pull requests. See
`CONTRIBUTING.md <CONTRIBUTING.md>`_ for the workflow and the tests every
change must pass.

Credits
-------

Hypatia was created at Politecnico di Milano by Negar Namazifard and Mohammad
Amin Tahavori, under the scientific coordination of Emanuela Colombo, with
the contribution of EnergyVille (KU Leuven, VITO, imec and UHasselt). The
original repository is `SESAM-Polimi/hypatia
<https://github.com/SESAM-Polimi/hypatia>`_. The documentation lists the
contributors to this version.

Licence
-------

Apache License 2.0, see `LICENSE <LICENSE>`_.
