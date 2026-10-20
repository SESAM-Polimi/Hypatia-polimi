.. _testing:

Tests
=====

.. code-block:: bash

   pytest                  # everything except the slow examples
   pytest -m slow          # the slow examples (Gurobi recommended)
   pytest tests/unit       # one folder

The suite answers four different questions, each with its own kind of test.

Is the solved model the same? Example objectives
------------------------------------------------

``tests/test_examples.py`` solves every example and compares the optimal
objective with a stored reference (relative tolerance :math:`10^{-6}`).
Six examples solve in seconds with HiGHS and run in CI; the other six take
minutes with open-source solvers, are marked ``slow`` and are meant to be run
locally before a release or after a change to the formulation. The solver is
HiGHS for the fast examples and Gurobi, if installed, for the slow ones;
``HYPATIA_TEST_SOLVER=GUROBI pytest ...`` overrides it.

An objective test passes as long as the optimum is the same, so it does not
see a change in a constraint that is not binding, or in code no example
uses. That is what the next test is for.

Is the built model the same? Build fingerprints
-----------------------------------------------

``tests/test_build_fingerprint.py`` builds 15 models (the examples, two of
them with optional constraints, and a *feature model* that uses what no
example does: modular capacity, ramp limits, minimum output, capacity
groups) and evaluates every constraint, objective term and result
expression at a random point, without solving. The values are summarised by
order-independent statistics and compared with references stored in
``tests/data/fingerprints``. Two versions of the code that build the same
coefficients give the same fingerprint; a change of one coefficient by one
part in a million in any row is reported, with the name of the constraint.
The method is explained at the top of ``tests/fingerprint.py``.

A second test checks that every constraint class produces rows in at least
one of the 15 models, so that no constraint is outside the comparison.

When a change is *meant* to alter the built model, regenerate the references
of the cases concerned and commit them with the change:

.. code-block:: bash

   python -m tests.fingerprint update Neverland Feature_model

and say in the pull request which items changed and why. The fingerprint is
also useful outside the test suite, to check that a new environment or a
refactoring builds the same model for a scenario of your own:

.. code-block:: python

   from tests.fingerprint import fingerprint, save, load, compare
   save(fingerprint(model), "before.json")                    # old code or environment
   print(compare(load("before.json"), fingerprint(model)))    # new one: [] if identical

Does the code do what the documentation says? Formulation tests
---------------------------------------------------------------

``tests/test_formulation.py`` re-implements the central equations of the
:ref:`formulation` with NumPy, written from the documentation and not from
the code, and compares them with the expressions Hypatia builds, at a random
point: capacity accumulation, decommissioning, salvage value, annuities,
emissions, the whole planning objective, and the carrier balance with lines.
Before the comparison the example data is changed so that every term
matters (economic and technical lifetimes differ, capture and taxes are
non-zero, the discount rate changes over time), otherwise a wrong term
multiplied by zero would pass. If the code or the documented equations
change, the test fails until the other is updated.

``tests/test_docs.py`` checks the documentation itself: every parameter
sheet the code can create is described in ``doc/tools/parameter_docs.py``
and no description is left over; every constraint class is listed with
equations that exist in the formulation pages; and the scripts in
``doc/examples`` run (they need HiGHS).

Do the parts work? Unit tests
-----------------------------

``tests/unit`` holds small tests of single functions and options, most on
cases worked out by hand: decommissioning, capacity groups, CHP, electrolysis,
unit commitment, the option names, the date selection of the plots.
``tests/legacy`` holds the original tests of Hypatia, updated to run on both
software stacks.

Two software stacks
-------------------

CI (``.github/workflows/tests.yml``) runs the suite twice:

* *current*: Python 3.11, CVXPY 1.9.2, NumPy 2 (``ci/requirements.txt``);
* *legacy*: Python 3.10, CVXPY 1.1.18, NumPy 1.22
  (``ci/requirements-legacy.txt``), the versions of the environments built
  for earlier versions of Hypatia. It will be dropped in a later release.

The versions are pinned exactly, so that a change in results can always be
traced to a change in the code or to a deliberate update of a pin, made in a
pull request of its own. Deprecation warnings raised by Hypatia's code or by
the tests are errors (``pytest.ini``), so that an incompatibility with a
coming version of pandas, NumPy or CVXPY fails a test instead of scrolling
past in a log.
