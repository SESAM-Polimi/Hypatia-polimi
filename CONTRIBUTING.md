# Contributing to Hypatia

Thank you for helping. The full guide is in the documentation
(`doc/source/developer/contributing.rst`, *Developer guide* in the built
documentation); this page is the short version.

## Reporting a problem

Open an issue with what you ran (the `Model(...)` call and its options, and
the run method), what happened (the full error message, or the result that
looks wrong and why), and the versions:

```bash
python -c "import hypatia, cvxpy, numpy, pandas; print(hypatia.__version__, cvxpy.__version__, numpy.__version__, pandas.__version__)"
```

A problem shown on one of the example models is solved much faster.

## Setting up

```bash
git clone https://github.com/SESAM-Polimi/Hypatia-polimi.git
cd Hypatia-polimi
conda create -n hypatia-dev python=3.11
conda activate hypatia-dev
pip install -r ci/requirements.txt
pip install -e . --no-deps
pytest
```

## Pull requests

1. Branch from an up-to-date `main` and keep the pull request about one thing.
   A refactoring must show identical results; a change of results must say
   why the results move.
2. Run `pytest`. The build fingerprints and the example objectives fail on
   any change to the model. When the change is intended, update the
   references in the same pull request (`python -m tests.fingerprint update
   <case>`) and explain it.
3. New behaviour comes with a test that fails without it. New parameters
   and constraints are documented in `doc/tools/parameter_docs.py` and in the
   formulation pages; `tests/test_docs.py` checks this.
4. Add a line to `CHANGELOG.rst` under *Unreleased*.
5. CI runs the tests on two software stacks; both must pass.

Please do not commit results, solver logs or large data files.
