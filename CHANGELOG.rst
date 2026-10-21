Release History
===============

Unreleased
----------
* **Bug: variable cost of lines with three or more regions.** ``utility.line_varcost`` tested the
  same line name in its ``if`` and ``elif``, so a region importing over a line named after the
  other region first was charged the variable cost of the last line processed: right by chance
  with two regions, wrong with three or more, and an ``UnboundLocalError`` when the first region
  of the ``Regions`` table was not the first alphabetically. Models with one or two regions are
  unchanged (all fingerprints identical). Test on a three-region case
  (``tests/unit/test_line_varcost.py``).
* **Input checks and clearer errors** (no change to the model built from valid inputs; all
  fingerprints unchanged):

  - an Operation model with several years stops when created, instead of failing at the solve
    with a CVXPY message;
  - a region without a Demand technology stops with a message naming it (it raised
    ``KeyError: 'Demand'``);
  - time-step fractions that do not add up to one give a warning;
  - a year without a row in a capacity-group limit sheet now means no limit for that year (it
    made the solve fail with "Problem data contains NaN");
  - a parameter file holding a filled sheet of an optional constraint that is not enabled gives a
    warning (the sheet was ignored silently: the cause of the 2.5 % difference in the RES-4 rerun
    without ``chp_operating_range``). Sheets left at their default values do not warn;
  - ``to_csv`` creates missing parent folders; with ``force_rewrite=True`` it empties the folder
    first, so no file of an earlier run is left, but only if the folder holds nothing but CSV
    files. An existing empty folder no longer needs ``force_rewrite``. An unknown
    post-processing module is reported before anything is written;
  - ``run_MO`` checks ``optimization="Multi"`` and ``number_solutions >= 2`` before solving, and
    stops with the solver status if the least-cost or least-emission run fails (it failed with
    ``TypeError`` or ``NameError``).
* **Run scripts**: the class scripts ``Run_Planning_MO.py`` and ``Run_Operation_MO.py`` moved out
  of the package to ``scripts/`` (``run_planning.py``, ``run_operation.py``), together with
  ``Tema.json``. They now have one settings block (model folder, optimisation, optional
  constraints, solver), paths relative to the script, copy their example on the first run, and
  are tested (``tests/test_scripts.py``).
* **Example data**: Atlantis had a technical lifetime of 1 year for every technology (the
  template default) with an economic lifetime of 20 years, so capacity built in one year was gone
  the next while its annuities ran for 20 years. Technical lifetimes are now 20 years. The
  least-cost solution now builds renewable capacity from 2026 and the objective falls from
  1.478e10 to 6.528e9. Fingerprint and objective reference updated; only
  lifetime-dependent items changed (total and decommissioned capacity, fixed costs, salvage value,
  the constraints on capacity, the objective). The Excel file was edited at the XML level, so the
  cached values of the formulas in the other sheets are untouched.
* **Documentation**: rewritten. New user guide (concepts, building a model, input reference,
  optional constraints, unit commitment, running and results), two tutorials whose scripts are run
  by the tests (``doc/examples``), the complete model formulation with units, defaults and a
  plain-language description of every constraint, review notes on points of the code that may not
  be intended, and a developer guide. The tables of input sheets and the map from constraint
  classes to equations are generated from the code at every build (``doc/tools``);
  ``tests/test_docs.py`` fails when a sheet or constraint is undocumented, and
  ``tests/test_formulation.py`` checks the main equations against an independent implementation.
  New README, ``CONTRIBUTING.md``, issue and pull request templates. The pages of the previous
  documentation and their images were removed (still in the git history);
  ``README_INPUT_EXCEL.md`` and ``README_MODULAR_UNIT_COMMITMENT.md`` were merged into the user
  guide.
* **Bugs**:

  - Line decommissioning (``line_decommissioned_capacity`` and its cost) was always zero: a hidden
    ``TypeError`` behind a bare ``except``. Fixed; technologies and lines now share one function.
    Results of multi-region planning models change accordingly; constraints and objectives of the
    examples do not (their line decommissioning costs are zero).
  - ``electrolysis_consumption`` crashed on models without an ``Electrolysis`` technology, and in
    multi-region models a region without electrolysers reused the previous region's consumption.
  - ``plot_hourly_prod_by_tech`` failed with its default dates (read by pandas as year 1); errors in
    the dates now keep their cause.
  - Removed the unused ``utility._calc_production_overall``.
* **pandas 2**: ``salvage_factor`` wrote a column-shaped (n, 1) array into a DataFrame column
  (``FutureWarning: Setting an item of incompatible dtype``, an error in a future pandas). Values
  were correct; the array is now flattened. The legacy tests no longer write decimals into
  integer tables. Deprecation warnings triggered by Hypatia's code or by the tests are now test
  errors (``pytest.ini``); those raised inside other libraries stay ignored.
* **Environment**: tested on Python 3.11 with CVXPY 1.9.2, NumPy 2.4, SciPy 1.17 and pandas 2.3
  (the CVXlab stack, ``ci/requirements.txt``). The previous stack (Python 3.10, CVXPY 1.1.18,
  ``ci/requirements-legacy.txt``) is still tested in CI and will be dropped in a later release.
  All 15 build fingerprints and all example objectives are the same on both stacks.
  ``setup.py`` now lists CVXPY as a dependency (it was missing), allows recent xlsxwriter
  (parameter templates checked identical) and requires Python 3.10 or later.
  ``cp.reshape`` calls state ``order="F"``, the current default that CVXPY plans to change.
  README installation instructions updated.
* **Tests**: legacy tests ran SciPy without choosing a method, so the algorithm changed with the
  SciPy version; they now use HiGHS, compare with a tolerance instead of exact equality, and no
  longer use ``DataFrame.append`` (removed in pandas 2).
* **Bug (numpy 2)**: ``decomcap`` hid every error behind a bare ``except: pass``. With numpy 2,
  ``int()`` of a one-element array raises, so all decommissioned capacity (and its cost) silently
  became zero. The retirement year is now computed explicitly; results on the current environment
  are unchanged. Hand-checked unit test added.
* **cvxpy compatibility**: in-place ``+=`` on cvxpy expressions (47 places) rewritten as
  ``a = a + b``, which cvxpy >= 1.4 requires. Same expressions are built (fingerprints unchanged).
  Removed the unused file ``EmissionConsumedRegional - semi original.py``.
* **Tests**: build fingerprint (``tests/fingerprint.py``, ``tests/test_build_fingerprint.py``).
  Every constraint, objective and result expression of 15 models is evaluated without solving, at
  zero and at a random point, and compared with stored references (``tests/data/fingerprints``).
  This catches changes the objective tests cannot see: constraints not binding at the optimum and
  code no example builds. A new feature model (``tests/data/feature_model``, generated by
  ``tests/data/make_feature_model.py`` from Neverland) builds MILP capacity, ramp limits, minimum
  power output and capacity groups; a test checks that every constraint class is covered.
* **Enhancement**: ``load_example(..., optional_constraints=[...])`` to load an example with other
  optional constraints than its default. ``BuildModel.constraints_by_class`` keeps the built
  constraints grouped by class. ``Constraint.get()`` returns an empty list (it returned ``{}``)
  when a constraint does not apply to the model.
* **Bug**: ``Model.create_config_file`` failed with ``NameError: pd`` (pandas not imported in
  ``core/main.py``).
* **Cleanup**: ``Build.py`` tested ``optimization == Multi or MinEm``, which is always true; total
  emissions are now computed unconditionally, as they in fact always were (no change in results).
  Internal key ``golabl_storage_cyclic_boundary`` renamed ``global_storage_cyclic_boundary``
  (sheet names unchanged, input files unaffected).
* **Tests**: test suite moved to a top-level ``tests/`` folder and run with pytest
  (``pytest``; slow examples with ``pytest -m slow``). Regression tests check the objective of
  every example against reference values (Gurobi, 19 October 2026). GitHub Actions runs the fast
  tests on every pull request with the pinned environment in ``ci/requirements.txt``.
  Old tests updated to the current ``ModelSettings`` signature; 18 tests and 3 files that compared
  cvxpy internals or used removed parameters were deleted (see git history).
* **Enhancement**: optional constraints (``chp_operating_range``, ``production_ramp``,
  ``min_tech_power_output``, ``electrolysis_consumption``, ``boolean_for_storage``)
  are now enabled per model with ``Model(..., optional_constraints=[...])`` instead
  of uncommenting lines in ``ConstraintList.py``. The default (none enabled) builds
  the same problem as before. The JRP Nucleare runs correspond to
  ``optional_constraints=["chp_operating_range"]`` (tag ``jrp-nucleare-v1``).
* **CHPOperatingRange**: the number of CHP groups is the number of rows in the
  ``CHP_variable_techs_selection`` sheet (it was fixed to 5 in the code). Multi-region models now
  work: the global template used a key the constraint did not read. Missing or invalid
  efficiencies raise an error instead of silently producing NaN coefficients. Debug prints removed.
* **Parameter templates** can declare ``"variable_index": True``: the rows are user-defined and
  only the columns are validated.
* **Bug**: models without storage technologies failed with ``KeyError: 'storage_cyclic_boundary'``
  (the storage cycle parameter was read even when no region has storage). All examples were affected.
* **Bug**: multi-region models with transmission lines could not be solved by LP solvers: line
  efficiency ``eff ** (length / 100)`` was written with ``cp.exp`` on input data, which cvxpy turned
  into an exponential cone. Now computed with numpy (same values).
* **Examples**: all examples load again. Missing parameter sheets added with their default values
  (which leave the related constraints inactive); Atlantis and Neverland restored from the old main.
  ``available_examples()``, ``load_example(name)`` and ``download_example(name, path)`` are exported
  from ``hypatia`` and work with every example folder.
* **Feature**: upper limits on the total and new capacity of user-defined groups of
  technologies, summed over regions (``TotalCapacityGroup``, ``NewCapacityGroup``).
  Groups go in the sheet ``Capacity_group_membership`` (Group, Technology) and limits in
  ``Max_totalcap_group`` / ``Max_newcap_group`` (Group, Year, value), in the regional
  parameter file for single-region models and in the global file for multi-region ones.
  Years use the model year codes (e.g. ``Y0``). Without these sheets nothing changes.
* **Bug**: EmissionCapRegional/Global, LandUsageGlobal and RenewableProductionRegional/Global
  built an invalid constraint (numpy array instead of cvxpy) when no matching term existed,
  and recognised the electricity carrier only if named exactly ``Electricity``. Fixes from
  May 2026 (old main, ``6b53b2c`` and ``fee908b``) brought over; carrier names are now
  matched ignoring case and surrounding spaces (``Electricity``, ``Power``).
  ``Final_electricity`` is still not treated as electricity.
* **Refactoring**: shared helpers in ``constraints/_terms.py`` replace five local copies;
  terms are summed with ``a + b`` instead of ``a += b``, which recent cvxpy rejects.
* **Bug**: ``print(model)`` failed because ``Model`` had no ``settings`` attribute;
  ``Model.settings`` is now a read-only property.

0.1.1 (04/02/2022)
------------------
* **Bug**: Total capacity plot duplicating regions solved (issue #1).
* **Bug**: Results overwrite bug fixed (issue #3).
* **Enhacement**: Parameter sheets order modfied (issue #4)
* **Enhacement(backward-incompatible)**: Activity and capacity units and CO2 emission table added to the global set files (issue #5)
* **Enhacement**: autocompiled config file with the names and units (issue #6).
* **Enhacement**: Parameter sheet names' error handling improved (issue #7)
* **Docs**: Doc modification (example gallery, building a model,...).
