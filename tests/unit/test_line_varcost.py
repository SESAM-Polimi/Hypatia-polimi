"""
Variable cost of the lines between regions (utility.line_varcost).

Each region pays, on what it imports from another region, the variable cost
of the line that joins them. Lines are named after their two regions in
alphabetical order ("reg1-reg2"), so a region reaches half of its lines under
its own name first and half under the other region's name first.

Until October 2026 the second case tested the same name as the first
(an if/elif with one condition), so the cost of the previously processed line
was reused: right by chance with two regions, wrong with three.
"""
import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from hypatia.utility.utility import line_varcost

YEARS = ["Y0", "Y1"]
STEPS = [1, 2]
LINES = ["reg1-reg2", "reg1-reg3", "reg2-reg3"]
COST = {"reg1-reg2": 1.0, "reg1-reg3": 10.0, "reg2-reg3": 100.0}  # per unit imported


def case(regions):
    columns = pd.MultiIndex.from_tuples([(line, "elec") for line in LINES])
    varcost = pd.DataFrame([[COST[line] for line in LINES]] * len(YEARS),
                           index=pd.Index(YEARS), columns=columns)
    # every region imports 1 unit per time step from every other region: 2 per year
    rows = len(YEARS) * len(STEPS)
    imports = {reg: {other: cp.Constant(np.ones((rows, 1))) for other in regions if other != reg}
               for reg in regions}
    return line_varcost(varcost, imports, regions, YEARS, STEPS, LINES)


def line(a, b):
    return "-".join(sorted([a, b]))


@pytest.mark.parametrize("regions", [["reg1", "reg2", "reg3"], ["reg3", "reg1", "reg2"]])
def test_each_import_pays_the_cost_of_its_own_line(regions):
    costs = case(regions)
    for reg in regions:
        for other in regions:
            if other != reg:
                expected = 2 * COST[line(reg, other)]
                np.testing.assert_allclose(costs[reg][other].value, [[expected], [expected]],
                                           err_msg=f"{reg} importing from {other}")
