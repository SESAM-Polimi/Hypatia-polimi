.. _developer-guide:
.. _contributing:

Contributing
============

Hypatia is developed by the SESAM group at Politecnico di Milano on GitHub,
at `SESAM-Polimi/Hypatia-polimi <https://github.com/SESAM-Polimi/Hypatia-polimi>`_.
Bug reports, questions and contributions are welcome there as issues and
pull requests.

Reporting a problem
-------------------

Open an issue with:

* what you ran (the ``Model(...)`` call, with its options, and the run
  method), and on which example or which of your models;
* what happened: the full error message, or the result that looks wrong
  and why;
* the versions: ``python -c "import hypatia, cvxpy, numpy, pandas;
  print(hypatia.__version__, cvxpy.__version__, numpy.__version__,
  pandas.__version__)"`` and the solver.

A problem that can be shown on one of the examples is solved much faster.

Setting up
----------

.. code-block:: bash

   git clone https://github.com/SESAM-Polimi/Hypatia-polimi.git
   cd Hypatia-polimi
   conda create -n hypatia-dev python=3.11
   conda activate hypatia-dev
   pip install -r ci/requirements.txt     # the exact versions CI uses
   pip install -e . --no-deps
   pytest                                 # about ten minutes

For the documentation, also ``pip install -r doc/requirements.txt``.

Workflow
--------

#. Create a branch from an up-to-date ``main``, named after what it does
   (``fix/line-varcost``, ``feature/storage-losses``, ``docs/tutorials``).
#. Keep each pull request about one thing. A refactoring and a change of
   results in the same pull request cannot be reviewed: the refactoring must
   show identical results, the change must show why the results move.
#. Run the tests (:ref:`testing`) and, if you changed the documentation, build
   it (below).
#. Add a line to ``CHANGELOG.rst`` under *Unreleased*: what changed for a
   user of Hypatia, not how.
#. Open the pull request against ``main``, describing what changed, why,
   and how it was checked. CI runs the tests on two software stacks; both
   must pass.

Rules that keep results trustworthy
-----------------------------------

* A change that alters the built model or a result must say so. The
  build fingerprints and the example objectives (:ref:`testing`) fail on any
  change. When the change is intended, update the reference in the same pull
  request and explain it in the description.
* New behaviour comes with a test that fails without it. For a new
  constraint: a case in the fingerprints that builds it, and a small test
  that checks its rows on a case worked out by hand.
* New parameters are documented in ``doc/tools/parameter_docs.py``
  (symbol, unit, meaning), and new constraints in
  ``CONSTRAINTS`` there and in the formulation pages. ``tests/test_docs.py``
  fails otherwise.
* No silent fallbacks. ``except: pass`` hid two bugs for years in
  Hypatia (both now fixed). Catch only the exception you expect, and fail
  with a message that names the file, the sheet and the technology.
* No results, solver logs or large data in the repository. Example
  models stay small enough to solve in seconds.

Code style
----------

Follow the style of the surrounding code and PEP 8. Names say what a
quantity is (``new_capacity``, ``line_efficiency``), as in the formulation;
units go in comments or docstrings where they are not obvious. Docstrings use
the NumPy format, which the API reference reads. Avoid in-place operators on
CVXPY expressions (``a += b``): write ``a = a + b``, which every CVXPY
version accepts.

Building the documentation
--------------------------

.. code-block:: bash

   cd doc
   sphinx-build -W -b html source build/html      # warnings are errors
   sphinx-build -b latex source build/latex && make -C build/latex   # PDF

The input tables and the implementation map are generated from the code at
every build (``doc/tools/reference.py``), so they need no editing by hand.
``.readthedocs.yaml`` holds the configuration for Read the Docs, with
warnings treated as errors as above.
