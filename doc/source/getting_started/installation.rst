.. _installation:

Installation
============

Hypatia needs Python 3.10 or later and is tested with Python 3.11. It is
installed from its source code, in an environment of its own.

1. Create and activate an environment, for example with conda:

   .. code-block:: bash

      conda create -n hypatia python=3.11
      conda activate hypatia

2. Get the code, either with git or by downloading and unzipping the
   repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/SESAM-Polimi/Hypatia-polimi.git
      cd Hypatia-polimi

3. Install Hypatia and its dependencies (CVXPY, pandas, NumPy, plotly and
   others):

   .. code-block:: bash

      pip install -e .

   The ``-e`` option installs the code in place, so that changes to the
   files, or a ``git pull``, take effect without reinstalling.

To use exactly the versions the tests are run with, install them first and
then Hypatia without its dependencies:

.. code-block:: bash

   pip install -r ci/requirements.txt
   pip install -e . --no-deps

Solvers
-------

CVXPY installs open-source solvers with it, among which HiGHS, a good
default for both linear and mixed-integer problems. Commercial solvers such
as Gurobi or CPLEX are installed separately, with their own licence
(``pip install gurobipy``); CVXPY finds them automatically. The solvers
available in an environment are listed by

.. code-block:: python

   import cvxpy
   print(cvxpy.installed_solvers())

Environments created for earlier versions
-----------------------------------------

Earlier versions of Hypatia ran on Python 3.8 to 3.10 with CVXPY 1.1.18, and
the old installation instructions created a Python 3.8 environment. The
current version does not install on Python 3.8: create a new environment as
above. Results produced with an earlier version can be reproduced with the
version of the code that produced them (for example a git tag) in the
environment used at the time; ``ci/requirements-legacy.txt`` lists the
CVXPY 1.1.18 environment that the tests still check.

With CVXPY 1.1.18 the solver name ``"highs"`` is not available; use
``solver="scipy", scipy_options={"method": "highs"}`` instead.
