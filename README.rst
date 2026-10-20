
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. image:: https://readthedocs.org/projects/hypatia-py/badge/?version=latest
    :target: https://hypatia-py.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. image:: https://img.shields.io/pypi/v/hypatia-py
    :target: https://pypi.org/project/hypatia-py/
   
.. image:: https://badges.gitter.im/Hypatia-py/community.svg
    :target: https://gitter.im/Hypatia-py/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link
    :alt: Documentation Status
    
.. image:: https://zenodo.org/badge/434963275.svg
   :target: https://zenodo.org/badge/latestdoi/434963275

.. image:: https://raw.githubusercontent.com/SESAM-Polimi/MARIO/767d2c0e9e42ae0b6acf7c3a1cc379d7bcd367fa/doc/source/_static/images/polimi.svg
   :width: 200
   :align: right

********
Hypatia
********
An Operation and Planning Energy System Modelling Framework


Additional documentation
------------------------

* `Excel input guide <README_INPUT_EXCEL.md>`_
* `Modular unit commitment formulation <README_MODULAR_UNIT_COMMITMENT.md>`_


What is it
-----------
Hypatia is an open source modelling framework written in Python that provides
a technology-rich basis for optimizing both the operation and planning mode of
the energy systems in short-term and long-term time horizons. Hypatia is able
to analyze various energy transition scenarios based on different policies such
as coal phase out, carbon taxes, renewable incentives and other national and
international pledges for the possible future energy systems.

Quickstart
----------
Hypatia needs Python 3.10 or later; it is tested with Python 3.11. We suggest a separate
environment, for example with conda:

.. code-block:: bash

    conda create -n hypatia python=3.11
    conda activate hypatia

Download or clone the repository, then, from its folder:

.. code-block:: bash

    pip install -e .

This also installs CVXPY and the other dependencies. To get exactly the versions the tests are
run with (CVXPY 1.9.2, NumPy 2), install them first:

.. code-block:: bash

    pip install -r ci/requirements.txt
    pip install -e . --no-deps

Environments built for earlier versions of Hypatia (Python 3.10, CVXPY 1.1.18) still work for
now; ``ci/requirements-legacy.txt`` lists that set of versions, which is also tested.

To check an installation, run one of the examples:

.. code-block:: python

    from hypatia import load_example
    model = load_example("Operation_1Region")
    model.run(solver="highs")

With CVXPY 1.1.18, which has no direct HiGHS interface, use
``model.run(solver="scipy", scipy_options={"method": "highs"})`` instead.

Most of the open source solvers that are supported by CVXPY (the optimization library used in Hypatia), will be installed
automatically with the software. For the commercial solvers, you should follow the specific installation methods. 


Python module requirements
--------------------------
Some of the key packages that Hypatia relies on are:

#. `Pandas <https://pandas.pydata.org/>`_
#. `Numpy <https://numpy.org/>`_
#. `Plotly <https://plotly.com/>`_
#. `Cvxpy <https://pypi.org/project/cvxpy/>`_ (domain-specific language)

Hypatia supports different **Open Source** and **Commercial** solvers like:

* `CBC <https://projects.coin-or.org/Cbc>`_
* `GLPK <https://www.gnu.org/software/glpk/>`_
* `OSQP <https://osqp.org/>`_
* `ECOS <https://www.embotech.com/ECOS>`_
* `CVXOPT <http://cvxopt.org/>`_
* `SCS <https://github.com/cvxgrp/scs>`_
* `CPLEX <https://www.ibm.com/products/category/business/commerce>`_
* `GUROBI <https://www.gurobi.com/>`_


.. note::
   * This project is under active development.


License
-------

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://www.apache.org/licenses/


This work is licensed under `Apache 2.0 <https://www.apache.org/licenses/>`_

