"""
The model formulation in the documentation (doc/source/formulation) must
describe what the code builds.

These tests re-implement the central equations of the formulation with numpy,
straight from the documentation, and compare them with the expressions
Hypatia builds, evaluated at a random point (no solver). Unlike the
fingerprint tests, which compare the code with its own past, they compare the
code with an independent description of the model: if either changes, they
fail until the other is updated.

Equation numbers refer to the formulation chapter of the documentation.
"""
import contextlib
import io

import cvxpy as cp
import numpy as np
import pytest

from hypatia import load_example
from hypatia.backend.Build import BuildModel
from tests.fingerprint import _random_value

RTOL = 1e-9


def make_every_term_matter(data):
    """Change the example's parameters so that no term of the formulation is
    zero or coincides with another one: in the shipped examples the economic
    and technical lifetimes are equal, capture is zero and the discount rate
    is constant, so a wrong formula for those terms would go unnoticed."""
    def set_table(region, key, values, dtype=float):
        # lifetimes and construction times are used as indices: keep them integer
        if key not in data.regional_parameters[region]:
            return  # e.g. investment parameters in Operation models
        frame = data.regional_parameters[region][key].astype(dtype)
        frame.loc[:, :] = values(frame)
        data.regional_parameters[region][key] = frame

    for region in data.settings.regions:
        parameters = data.regional_parameters[region]
        set_table(region, "emission_capture_efficiency", lambda f: 0.3)
        set_table(region, "tech_decom_cost", lambda f: 5.0)
        set_table(region, "interest_rate", lambda f: 0.08)
        set_table(region, "inv_taxsub", lambda f: [[0.1 if c[0] == "Tax" else 0.05 for c in f.columns]] * len(f))
        set_table(region, "fix_taxsub", lambda f: [[0.07 if c[0] == "Tax" else 0.02 for c in f.columns]] * len(f))
        if "discount_rate" in parameters:
            set_table(region, "discount_rate", lambda f: np.linspace(0.03, 0.07, len(f)).reshape(-1, 1))
        if "economic_lifetime" in parameters:
            lifetime = parameters["tech_lifetime"].values
            set_table(region, "economic_lifetime", lambda f: np.maximum(1, lifetime // 2), int)
            set_table(region, "time_of_construction", lambda f: 2, int)
        # one technology with negative emissions, to test the second branch of eq. (emissions)
        emission = parameters["specific_emission"].astype(float)
        emission.iloc[:, 0] = -0.2
        parameters["specific_emission"] = emission

    # Lines (Planning, several regions): investments, different lifetimes,
    # interest rate different from the global discount rate, which changes.
    trade = data.trade_parameters
    if trade is not None and "line_inv" in trade:
        def set_trade(key, value, dtype=float):
            frame = trade[key].astype(dtype)
            frame.loc[:, :] = value
            trade[key] = frame
        set_trade("line_inv", 1000.0)
        set_trade("line_fixed_cost", 2.0)
        set_trade("line_decom_cost", 3.0)
        set_trade("interest_rate", 0.08)
        set_trade("line_lifetime", 12, int)
        set_trade("line_economic_lifetime", 6, int)
        set_trade("line_time_of_construction", 2, int)
        rate = data.global_parameters["global_discount_rate"].astype(float)
        rate.loc[:, :] = np.linspace(0.03, 0.07, len(rate)).reshape(-1, 1)
        data.global_parameters["global_discount_rate"] = rate


def built_at_random_point(example, seed=7):
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_example(example)._Model__model_data
        make_every_term_matter(data)
        built = BuildModel(data)
    problem = cp.Problem(cp.Minimize(built.global_objective), built.constr)
    rng = np.random.default_rng(seed)
    for v in sorted(problem.variables(), key=lambda v: v.id):
        v.save_value(_random_value(v, rng))
    return built


def table(data, region, key, *columns):
    """A parameter table as a numpy array (years x technologies)."""
    frame = data.regional_parameters[region][key]
    for column in columns:
        frame = frame[column]
    return np.asarray(frame.values, dtype=float)


def available_new_capacity(N, TC):
    """Eq. (real-newcap): capacity decided in y becomes available in y + TC."""
    years = N.shape[0]
    N_hat = np.zeros_like(N)
    for j, tc in enumerate(TC):
        for y in range(years):
            if y >= tc:
                N_hat[y, j] = N[y - tc, j]
    return N_hat


def total_capacity(N_hat, LT, RC):
    """Eq. (total-capacity)."""
    years = N_hat.shape[0]
    CAP = np.array(RC, dtype=float)
    for j, lt in enumerate(LT):
        for y in range(years):
            CAP[y, j] += sum(N_hat[yp, j] for yp in range(y + 1) if y - yp < lt)
    return CAP


def decommissioned(N_hat, LT):
    """Eq. (decommissioning)."""
    D = np.zeros_like(N_hat)
    for j, lt in enumerate(LT):
        for y in range(N_hat.shape[0]):
            if y >= lt:
                D[y, j] = N_hat[y - lt, j]
    return D


def salvage_factor(LT, EL, i, d):
    """Eq. (salvage): years x technologies (or carriers of a line)."""
    years = len(d)
    Y_end = years - 1
    SF = np.zeros((years, len(LT)))
    for j in range(len(LT)):
        for y in range(years):
            if y + LT[j] > Y_end:
                share = ((1 + d[y]) ** (LT[j] - Y_end - 1 + y) - 1) / ((1 + d[y]) ** LT[j] - 1)
                technical = (1 - (1 + i[j]) ** -1) / (1 - (1 + i[j]) ** -EL[j])
                social = (1 - (1 + d[y]) ** -EL[j]) / (1 - (1 + d[y]) ** -1)
                SF[y, j] = share * technical * social
    return SF


def present_value_of_annuities(investment, EL, i, d):
    """Eq. (annuities): all annuities of the investments of every year."""
    CRF = i * (1 + i) ** EL / ((1 + i) ** EL - 1)
    total = 0.0
    for j in range(investment.shape[1]):
        for y in range(investment.shape[0]):
            for k in range(1, EL[j] + 1):
                total += CRF[j] * investment[y, j] / (1 + d[y]) ** (y + k)
    return total


@pytest.mark.parametrize("example", ["Atlantis", "Planning_1Region"])
def test_capacity_and_planning_objective(example):
    built = built_at_random_point(example)
    data, v = built.model_data, built.vars
    years = len(data.settings.years)
    steps = len(data.settings.time_steps)
    region = data.settings.regions[0]
    d = table(data, region, "discount_rate").ravel()
    emission_types = [c.split(" ")[0] for c in data.settings.global_settings["Emissions"]["Emission"]]

    annual_cost = np.zeros(years)
    annuities = 0.0
    for category in v.new_capacity[region]:
        N = v.new_capacity[region][category].value
        TC = table(data, region, "time_of_construction", category).ravel().astype(int)
        LT = table(data, region, "tech_lifetime", category).ravel().astype(int)
        EL = table(data, region, "economic_lifetime", category).ravel().astype(int)
        i = table(data, region, "interest_rate", category).ravel()
        RC = table(data, region, "tech_residual_cap", category)

        N_hat = available_new_capacity(N, TC)
        CAP = total_capacity(N_hat, LT, RC)
        D = decommissioned(N_hat, LT)
        np.testing.assert_allclose(v.totalcapacity[region][category].value, CAP, rtol=RTOL)
        np.testing.assert_allclose(v.decommissioned_capacity[region][category].value, D,
                                   rtol=RTOL, atol=1e-12)

        INV = table(data, region, "tech_inv", category)
        ITAX = table(data, region, "inv_taxsub", "Tax", category)
        ISUB = table(data, region, "inv_taxsub", "Sub", category)
        FOM = table(data, region, "tech_fixed_cost", category)
        FTAX = table(data, region, "fix_taxsub", "Tax", category)
        FSUB = table(data, region, "fix_taxsub", "Sub", category)
        VOM = table(data, region, "tech_var_cost", category)
        DEC = table(data, region, "tech_decom_cost", category)
        p = v.technology_prod[region][category].value
        P = p.reshape(years, steps, -1).sum(axis=1)

        SV = salvage_factor(LT, EL, i, d) * INV * N  # eq. (salvage)

        # Eq. (annual-cost), technology terms
        annual_cost += ((ITAX - ISUB) * INV * N + (1 + FTAX - FSUB) * FOM * CAP
                        + VOM * P + DEC * D - SV).sum(axis=1)

        annuities += present_value_of_annuities(INV * N, EL, i, d)  # eq. (annuities)

        # Eq. (emissions) and emission taxes
        if category in ("Supply", "Conversion", "Conversion_plus"):
            for e in emission_types:
                EF = table(data, region, "specific_emission", e, category)
                capture = table(data, region, "emission_capture_efficiency", e, category)
                ETAX = table(data, region, "emission_tax", e, category)
                emitted = np.where(EF[0] >= 0, (1 - capture) * EF * P, EF * P)
                np.testing.assert_allclose(v.emission_by_region[region][e][category].value,
                                           emitted, rtol=RTOL, atol=1e-12)
                annual_cost += (ETAX * emitted).sum(axis=1)

    # Eq. (annual-cost), unmet demand
    for q in v.unmetdemandbycarrier[region].values():
        q = np.asarray(getattr(q, "value", q), dtype=float).reshape(years, steps)
        annual_cost += 1e10 * q.sum(axis=1)

    # Eq. (npc-technologies)
    npc = (annual_cost / (1 + d) ** np.arange(years)).sum() + annuities
    assert built.global_objective.value == pytest.approx(npc, rel=RTOL)


@pytest.mark.parametrize("example", ["Atlantis", "Neverland", "Operation_2Regions"])
def test_carrier_balance(example):
    """Eqs. (carrier-production), (carrier-use), (imports-exports), (balance)."""
    built = built_at_random_point(example)
    data, v, s = built.model_data, built.vars, built.model_data.settings
    rows = len(s.years) * len(s.time_steps)
    carriers = list(s.global_settings["Carriers_glob"]["Carrier"])
    balances = iter(built.constraints_by_class["Balance"])
    for region in s.regions:
        inputs = s.regional_settings[region]["Carrier_input"]
        outputs = s.regional_settings[region]["Carrier_output"]
        for c in carriers:
            prod, use, dem = np.zeros(rows), np.zeros(rows), np.zeros(rows)
            for category, techs in s.technologies[region].items():
                for j, tech in enumerate(techs):
                    if c in outputs.loc[outputs["Technology"] == tech, "Carrier_out"].values:
                        flow = v.technology_prod[region][category].value[:, j]
                        if category == "Conversion_plus":
                            flow = flow * data.regional_parameters[region]["carrier_ratio_out"][(tech, c)].values
                        prod += flow
                    if c in inputs.loc[inputs["Technology"] == tech, "Carrier_in"].values:
                        if category == "Demand":
                            dem += data.regional_parameters[region]["demand"][tech].values
                        elif category != "Supply":
                            flow = v.technology_use[region][category].value[:, j]
                            if category == "Conversion_plus":
                                flow = flow * data.regional_parameters[region]["carrier_ratio_in"][(tech, c)].values
                            use += flow
            imports, exports = np.zeros(rows), np.zeros(rows)
            if s.multi_node:
                k = carriers.index(c)
                for other in v.line_import[region]:
                    line = f"{region}-{other}" if f"{region}-{other}" in s.lines_list else f"{other}-{region}"
                    eff = data.trade_parameters["line_eff"][(line, c)].values
                    length = data.trade_parameters["line_length"][(line, c)].values
                    loss = np.repeat(np.power(eff, length / 100), len(s.time_steps))
                    imports += loss * v.line_import[region][other].value[:, k]
                    exports += v.line_export[region][other].value[:, k]
            expected = prod + imports - use - exports - dem
            np.testing.assert_allclose(np.ravel(next(balances).expr.value), expected,
                                       rtol=RTOL, atol=1e-9)


def test_line_costs_planning():
    """Eqs. (line-capacity) and (npc-lines): lines with annuities and salvage
    value, like technologies (since October 2026)."""
    built = built_at_random_point("Neverland")
    data, v, s = built.model_data, built.vars, built.model_data.settings
    trade = data.trade_parameters
    years = len(s.years)
    d = np.asarray(data.global_parameters["global_discount_rate"].values, dtype=float).ravel()

    def row(key, line, dtype=float):
        return np.asarray(trade[key][line].values, dtype=dtype).ravel()

    annual = np.zeros(years)
    annuities = 0.0
    for line in s.lines_list:
        N = v.line_new_capacity[line].value
        TC, LT, EL = (row(k, line, int) for k in ("line_time_of_construction", "line_lifetime",
                                                  "line_economic_lifetime"))
        i = row("interest_rate", line)
        RC = np.asarray(trade["line_residual_cap"][line].values, dtype=float)
        N_hat = available_new_capacity(N, TC)
        CAP = total_capacity(N_hat, LT, RC)
        D = decommissioned(N_hat, LT)
        np.testing.assert_allclose(v.line_totalcapacity[line].value, CAP, rtol=RTOL)

        INV, FOM, DEC = (np.asarray(trade[k][line].values, dtype=float)
                         for k in ("line_inv", "line_fixed_cost", "line_decom_cost"))
        SV = salvage_factor(LT, EL, i, d) * INV * N
        np.testing.assert_allclose(v.salvage_inv_line[line].value, SV, rtol=RTOL, atol=1e-9)
        annual += (FOM * CAP + DEC * D - SV).sum(axis=1)
        annuities += present_value_of_annuities(INV * N, EL, i, d)

    assert float(built.inv_lines.value) == pytest.approx(annuities, rel=RTOL)
    # variable costs of the lines, checked on their own in tests/unit/test_line_varcost.py
    for costs in v.cost_variable_line.values():
        for cost in costs.values():
            annual += np.asarray(cost.value).sum(axis=1)
    npc_lines = (annual / (1 + d) ** np.arange(years)).sum() + annuities
    built_npc = float(np.sum(built.totalcost_lines_discounted.value)) + float(built.inv_lines.value)
    assert built_npc == pytest.approx(npc_lines, rel=RTOL)
