import cvxpy as cp
import numpy as np
import pandas as pd

from hypatia.utility.utility import decomcap, line_decomcap

'''
Decommissioned capacity on a case small enough to check by hand: four years,
technology A with a lifetime of 2 years, technology B with a lifetime longer
than the horizon.

Until October 2026 decomcap hid every error behind a bare "except: pass".
With numpy 2 an error raised for every year, so all decommissioning silently
became zero. This test fails if that happens again.
'''

YEARS = ["Y0", "Y1", "Y2", "Y3"]
NEW_CAPACITY = np.array([[10.0, 1.0],
                         [20.0, 2.0],
                         [30.0, 3.0],
                         [40.0, 4.0]])  # rows: years, columns: A, B


def test_capacity_retires_after_its_lifetime():
    newcap = cp.Variable((4, 2))
    newcap.value = NEW_CAPACITY
    lifetime = pd.DataFrame([[2, 5]], columns=["A", "B"])

    decom = decomcap(newcap, ["A", "B"], YEARS, lifetime).value

    # A: built in Y0 retires in Y2, built in Y1 retires in Y3; later ones
    # retire after the horizon. B never retires within the horizon.
    np.testing.assert_allclose(decom, [[0.0, 0.0],
                                       [0.0, 0.0],
                                       [10.0, 0.0],
                                       [20.0, 0.0]])


def test_line_capacity_retires_after_its_lifetime():
    # Same rule for the carriers of a line. ModelVariables passes the carrier
    # names as a pandas Series and the lifetimes as a one-row table. Before
    # October 2026 line_decomcap always returned zero (a hidden TypeError).
    newcap = cp.Variable((4, 2))
    newcap.value = NEW_CAPACITY
    carriers = pd.Series(["Elec", "Gas"], name="Carrier")
    lifetime = pd.DataFrame([[1, 3]], columns=["Elec", "Gas"])

    decom = line_decomcap(newcap, carriers, YEARS, lifetime).value

    # Elec (1 year): built in Y0, Y1, Y2 retires in Y1, Y2, Y3.
    # Gas (3 years): built in Y0 retires in Y3.
    np.testing.assert_allclose(decom, [[0.0, 0.0],
                                       [10.0, 0.0],
                                       [20.0, 0.0],
                                       [30.0, 1.0]])
