"""
Documentation of every input parameter Hypatia reads: symbol used in the
model formulation, unit and plain-language description.

This is the single source for the input reference and for the nomenclature
of the formulation. tests/test_docs.py fails if Hypatia creates a parameter
sheet that has no entry here, so a new sheet cannot ship undocumented.

Keys are (file, parameter key), where file is "regional"
(parameters_<region>.xlsx), "global" (parameters_global.xlsx, multi-region
models only) or "connections" (parameters_connections.xlsx, multi-region
models only); the parameter key is the name Hypatia uses internally.

Units use the generic symbols defined in UNITS. Hypatia does not convert
units: the user chooses them, and only has to keep them consistent.
"""

UNITS = {
    "[E]": "energy or activity, in the unit chosen for the carrier (e.g. MWh, GWh, kt of fuel)",
    "[P]": "capacity, in the unit chosen for the technology (e.g. MW, GW); for storage, an energy capacity",
    "[M]": "money (e.g. EUR, MEUR)",
    "[Em]": "emissions, in the unit chosen for each emission type (e.g. tCO2, ktCO2-eq)",
    "[Ar]": "land area (e.g. km2)",
    "[len]": "length of an inter-regional line, in the unit used for the line efficiency",
    "[yr]": "years, counted as positions in the list of modelled years",
    "[h]": "hours",
    "[min]": "minutes",
    "[ts]": "number of time steps",
    "[-]": "dimensionless (fraction, share, ratio or flag)",
}

# Shorthand for repeated texts.
_ANNUAL_LIMIT = "Applied to the annual sum over the time steps of each year."
_GLOBAL_FILE = ("In multi-region models this sheet is in parameters_global.xlsx "
                "and applies to all regions; in single-region models it is in "
                "the regional file.")

PARAMETERS = {
    # ------------------------------------------------------------ regional
    ("regional", "demand"): dict(
        symbol=r"DEM_{r,k,y,t}", unit="[E]",
        text="Final demand of each demand technology in each time step. Its input "
             "carrier must be supplied exactly (or partly left unmet, see Ensure_feasibility)."),
    ("regional", "tech_fixed_cost"): dict(
        symbol=r"FOM_{r,k,y}", unit="[M/(P·yr)]",
        text="Fixed operation and maintenance cost per unit of total installed capacity and year."),
    ("regional", "tech_var_cost"): dict(
        symbol=r"VOM_{r,k,y}", unit="[M/E]",
        text="Variable cost per unit of production (activity output). For supply "
             "technologies this is where fuel costs go."),
    ("regional", "tech_residual_cap"): dict(
        symbol=r"RC_{r,k,y}", unit="[P]",
        text="Capacity installed before the first modelled year that is still in "
             "operation in each year. In Operation mode it is the whole installed capacity."),
    ("regional", "specific_emission"): dict(
        symbol=r"EF_{r,k,e,y}", unit="[Em/E]",
        text="Emissions of each type per unit of production. A negative value "
             "represents a negative-emission technology (see the emission accounting)."),
    ("regional", "specific_land_usage"): dict(
        symbol=r"LU_{r,k,y}", unit="[Ar/P]",
        text="Land used per unit of total installed capacity."),
    ("regional", "emission_capture_efficiency"): dict(
        symbol=r"\eta^{cap}_{r,k,e,y}", unit="[-]",
        text="Fraction of the emissions of a technology that is captured and not released."),
    ("regional", "emission_tax"): dict(
        symbol=r"ETAX_{r,k,e,y}", unit="[M/Em]",
        text="Tax per unit of emitted (not captured) emission. Applied with its "
             "sign to negative emissions too, which then earn a credit."),
    ("regional", "fix_taxsub"): dict(
        symbol=r"FTAX_{r,k,y},\ FSUB_{r,k,y}", unit="[-]",
        text="Taxes and subsidies on fixed costs, as a fraction of the fixed cost "
             "(two column groups: Tax and Sub)."),
    ("regional", "annualprod_per_unitcapacity"): dict(
        symbol=r"APC_{r,k}", unit="[E/(P·yr)]",
        text="Production of one unit of capacity running at full output for a "
             "whole year (e.g. 8760 MWh per MW). Converts capacity into energy."),
    ("regional", "tech_inv"): dict(
        symbol=r"INV_{r,k,y}", unit="[M/P]",
        text="Investment cost per unit of new capacity decided in a year."),
    ("regional", "inv_taxsub"): dict(
        symbol=r"ITAX_{r,k,y},\ ISUB_{r,k,y}", unit="[-]",
        text="Taxes and subsidies on investments, as a fraction of the investment "
             "cost (two column groups: Tax and Sub). Paid in the year of the decision."),
    ("regional", "tech_decom_cost"): dict(
        symbol=r"DEC_{r,k,y}", unit="[M/P]",
        text="Cost per unit of capacity reaching the end of its technical lifetime."),
    ("regional", "discount_rate"): dict(
        symbol=r"d_{r,y}", unit="[1/yr]",
        text="Discount rate used to bring the costs of each year to present value."),
    ("regional", "tech_lifetime"): dict(
        symbol=r"LT_{r,k}", unit="[yr]",
        text="Technical lifetime: how many modelled years new capacity stays in operation."),
    ("regional", "economic_lifetime"): dict(
        symbol=r"EL_{r,k}", unit="[yr]",
        text="Number of years over which an investment is paid back as annuities."),
    ("regional", "time_of_construction"): dict(
        symbol=r"TC_{r,k}", unit="[yr]",
        text="Years between the investment decision and the moment the capacity "
             "becomes available. Must be at least 1 (the default)."),
    ("regional", "interest_rate"): dict(
        symbol=r"i_{r,k}", unit="[1/yr]",
        text="Interest rate used to turn an investment into annuities (capital recovery factor)."),
    ("regional", "new_capacity_step"): dict(
        symbol=r"STEP_{r,k}", unit="[P]",
        text="Size of one modular unit. If positive, new capacity can only be an "
             "integer number of units (mixed-integer problem); 0 means continuous. "
             + _GLOBAL_FILE),
    ("global", "global_new_capacity_step"): dict(
        symbol=r"STEP_{k}", unit="[P]",
        text="Size of one modular unit (see the regional sheet of the same name), "
             "for multi-region models."),
    ("regional", "storage_charge_efficiency"): dict(
        symbol=r"\eta^{ch}_{r,k,y}", unit="[-]",
        text="Efficiency of charging a storage technology."),
    ("regional", "storage_discharge_efficiency"): dict(
        symbol=r"\eta^{dis}_{r,k,y}", unit="[-]",
        text="Efficiency of discharging a storage technology."),
    ("regional", "storage_min_SOC"): dict(
        symbol=r"SOC^{min}_{r,k,y}", unit="[-]",
        text="Minimum state of charge, as a fraction of the usable storage capacity."),
    ("regional", "storage_charge_time"): dict(
        symbol=r"T^{ch}_{r,k}", unit="[h]",
        text="Hours needed to charge the full storage capacity: the maximum charging "
             "power is capacity divided by this time."),
    ("regional", "storage_discharge_time"): dict(
        symbol=r"T^{dis}_{r,k}", unit="[h]",
        text="Hours needed to discharge the full storage capacity."),
    ("regional", "storage_discharging_depth"): dict(
        symbol=r"DOD_{r,k}", unit="[-]",
        text="Depth of discharge: the fraction of the storage capacity that can be used."),
    ("regional", "storage_cyclic_boundary"): dict(
        symbol=r"T^{cyc}", unit="[ts]",
        text="Length of a storage cycle in time steps (default: one year). The state "
             "of charge is counted from the start of each cycle; the number of time "
             "steps in a year must be a multiple of it. " + _GLOBAL_FILE),
    ("global", "global_storage_cyclic_boundary"): dict(
        symbol=r"T^{cyc}", unit="[ts]", sheet="Storage_cyclic_boundary",
        text="Length of a storage cycle in time steps, for multi-region models."),
    ("regional", "tech_availability"): dict(
        symbol=r"AV_{r,k,y}", unit="[-]",
        text="Availability of a technology over the year (e.g. planned outages); "
             "scales the annual capacity factor limits and the storage flows."),
    ("regional", "min_tech_capacity_factor"): dict(
        symbol=r"CF^{min}_{r,k,y}", unit="[-]",
        text="Minimum annual production as a fraction of the available production."),
    ("regional", "max_tech_capacity_factor"): dict(
        symbol=r"CF^{max}_{r,k,y}", unit="[-]",
        text="Maximum annual production as a fraction of the available production."),
    ("regional", "res_capacity_factor"): dict(
        symbol=r"CF^{res}_{r,k,y,t}", unit="[-]",
        text="Output of one unit of capacity in each time step as a fraction of full "
             "output (resource profile, e.g. solar or wind availability)."),
    ("regional", "tech_efficiency"): dict(
        symbol=r"\eta_{r,k,y}", unit="[E/E]",
        text="Output per unit of input for conversion, Conversion_plus and transmission "
             "technologies. Can exceed 1 when input and output are in different units."),
    ("regional", "carrier_ratio_in"): dict(
        symbol=r"CRI_{r,k,c,y,t}", unit="[E/E]",
        text="Conversion_plus technologies: input of each carrier per unit of the "
             "technology's input activity."),
    ("regional", "carrier_ratio_out"): dict(
        symbol=r"CRO_{r,k,c,y,t}", unit="[E/E]",
        text="Conversion_plus technologies: output of each carrier per unit of the "
             "technology's output activity."),
    ("regional", "Conv_plus_max_production"): dict(
        symbol=r"\overline{PC}_{r,k,c,y}", unit="[E]",
        text="Conversion_plus technologies: maximum annual production of each output carrier."),
    ("regional", "Conv_plus_min_production"): dict(
        symbol=r"\underline{PC}_{r,k,c,y}", unit="[E]",
        text="Conversion_plus technologies: minimum annual production of each output carrier."),
    ("regional", "Conv_plus_max_production_share"): dict(
        symbol=r"\overline{SC}_{r,k,c,y}", unit="[-]",
        text="Conversion_plus technologies: maximum share of the region's annual "
             "production of the carrier."),
    ("regional", "Conv_plus_min_production_share"): dict(
        symbol=r"\underline{SC}_{r,k,c,y}", unit="[-]",
        text="Conversion_plus technologies: minimum share of the region's annual "
             "production of the carrier."),
    ("regional", "tech_max_production"): dict(
        symbol=r"\overline{PR}_{r,k,y}", unit="[E]",
        text="Maximum annual production of a technology. " + _ANNUAL_LIMIT),
    ("regional", "tech_min_production"): dict(
        symbol=r"\underline{PR}_{r,k,y}", unit="[E]",
        text="Minimum annual production of a technology. " + _ANNUAL_LIMIT),
    ("regional", "tech_max_production_share"): dict(
        symbol=r"\overline{SR}_{r,k,y}", unit="[-]",
        text="Maximum share of a technology in the region's annual production of its output carrier."),
    ("regional", "tech_min_production_share"): dict(
        symbol=r"\underline{SR}_{r,k,y}", unit="[-]",
        text="Minimum share of a technology in the region's annual production of its output carrier."),
    ("regional", "tech_maxtotcap"): dict(
        symbol=r"\overline{TC}_{r,k,y}", unit="[P]", text="Maximum total installed capacity."),
    ("regional", "tech_mintotcap"): dict(
        symbol=r"\underline{TC}_{r,k,y}", unit="[P]", text="Minimum total installed capacity."),
    ("regional", "tech_max_newcap"): dict(
        symbol=r"\overline{NC}_{r,k,y}", unit="[P]",
        text="Maximum new capacity becoming available in a year (after the construction time)."),
    ("regional", "tech_min_newcap"): dict(
        symbol=r"\underline{NC}_{r,k,y}", unit="[P]",
        text="Minimum new capacity becoming available in a year (after the construction time)."),
    ("regional", "emission_cap_annual"): dict(
        symbol=r"\overline{EM}_{r,e,y}", unit="[Em]",
        text="Maximum annual emissions of each type in the region (one column per type)."),
    ("regional", "emission_cap_annual_power"): dict(
        symbol=r"\overline{EM}^{pow}_{r,e,y}", unit="[Em]",
        text="Maximum annual emissions of each type from technologies that produce "
             "electricity (carrier named Electricity or Power)."),
    ("regional", "emission_consumed"): dict(
        symbol=r"\overline{CS}_{r,e,y}", unit="[Em]",
        text="Maximum annual captured emissions (storage capacity for captured emissions)."),
    ("regional", "max_land_usage"): dict(
        symbol=r"\overline{LA}_{r,k,y}", unit="[Ar]",
        text="Maximum land used by each technology in the region."),
    ("regional", "renewable_tech"): dict(
        symbol=r"RES_{r,k}", unit="[-]",
        text="1 if the technology counts as renewable, 0 otherwise."),
    ("regional", "min_renewable_production_share"): dict(
        symbol=r"\underline{RS}_{r,y}", unit="[-]",
        text="Minimum annual production of renewable technologies, as a share of the "
             "region's total final demand (all carriers added together)."),
    ("regional", "min_renewable_electric_share"): dict(
        symbol=r"\underline{RS}^{el}_{r,y}", unit="[-]",
        text="Minimum share of renewable technologies in the region's annual "
             "production of electricity (carriers named Electricity or Power)."),
    ("regional", "variable_CHP_tech_selection"): dict(
        symbol=r"k^{chp},\ k^{el},\ k^{th},\ \gamma", unit="names, [-]",
        text="Optional (chp_operating_range). One row per CHP plant: the CHP "
             "technology, its power branch, its heat branch and the maximum heat "
             "output per unit of CHP activity. " + _GLOBAL_FILE),
    ("global", "glob_variable_CHP_tech_selection"): dict(
        symbol=r"k^{chp},\ k^{el},\ k^{th},\ \gamma", unit="names, [-]",
        text="Optional (chp_operating_range). CHP plants, for multi-region models."),
    ("regional", "max_rate_ramp-up"): dict(
        symbol=r"\rho^{up}_{r,k}", unit="[1/min]",
        text="Optional (production_ramp). Fraction of the available output a "
             "technology can increase per minute; 1 or more means no limit. "
             "Meant for hourly time steps. " + _GLOBAL_FILE),
    ("regional", "max_rate_ramp-down"): dict(
        symbol=r"\rho^{down}_{r,k}", unit="[1/min]",
        text="Optional (production_ramp). Fraction of the available output a "
             "technology can decrease per minute; 1 or more means no limit."),
    ("global", "glob_max_rate_ramp-up"): dict(
        symbol=r"\rho^{up}_{k}", unit="[1/min]",
        text="Optional (production_ramp). Ramp-up rate, for multi-region models."),
    ("global", "glob_max_rate_ramp-down"): dict(
        symbol=r"\rho^{down}_{k}", unit="[1/min]",
        text="Optional (production_ramp). Ramp-down rate, for multi-region models."),
    ("regional", "min_tech_power_output"): dict(
        symbol=r"\underline{p}_{r,k}", unit="[-]",
        text="Optional (min_tech_power_output). Minimum output when the technology "
             "runs, as a fraction of its available output; 0 means no minimum. With a "
             "positive Modular_cap_unit (Planning) the minimum applies per unit online. "
             + _GLOBAL_FILE),
    ("global", "glob_min_tech_power_output"): dict(
        symbol=r"\underline{p}_{k}", unit="[-]",
        text="Optional (min_tech_power_output). Minimum output, for multi-region models."),
    # -------------------------------------------------------------- global
    ("global", "global_discount_rate"): dict(
        symbol=r"d^{glob}_{y}", unit="[1/yr]",
        text="Discount rate for the costs of inter-regional lines."),
    ("global", "global_emission_cap_annual"): dict(
        symbol=r"\overline{EM}^{glob}_{e,y}", unit="[Em]",
        text="Maximum annual emissions of each type over all regions."),
    ("global", "global_max_land_usage"): dict(
        symbol=r"\overline{LA}^{glob}_{k,y}", unit="[Ar]",
        text="Maximum land used by each technology over all regions."),
    ("global", "global_max_newcap"): dict(
        symbol=r"\overline{NC}^{glob}_{k,y}", unit="[P]",
        text="Maximum new capacity of each technology over all regions."),
    ("global", "global_min_newcap"): dict(
        symbol=r"\underline{NC}^{glob}_{k,y}", unit="[P]",
        text="Minimum new capacity of each technology over all regions."),
    ("global", "global_maxtotcap"): dict(
        symbol=r"\overline{TC}^{glob}_{k,y}", unit="[P]",
        text="Maximum total capacity of each technology over all regions."),
    ("global", "global_mintotcap"): dict(
        symbol=r"\underline{TC}^{glob}_{k,y}", unit="[P]",
        text="Minimum total capacity of each technology over all regions."),
    ("global", "global_max_production"): dict(
        symbol=r"\overline{PR}^{glob}_{k,c,y}", unit="[E]",
        text="Maximum annual production of each carrier by each technology over all regions."),
    ("global", "global_min_production"): dict(
        symbol=r"\underline{PR}^{glob}_{k,c,y}", unit="[E]",
        text="Minimum annual production of each carrier by each technology over all regions."),
    ("global", "global_max_production_share"): dict(
        symbol=r"\overline{SR}^{glob}_{k,c,y}", unit="[-]",
        text="Maximum share of a technology in the total production of a carrier over all regions."),
    ("global", "global_min_production_share"): dict(
        symbol=r"\underline{SR}^{glob}_{k,c,y}", unit="[-]",
        text="Minimum share of a technology in the total production of a carrier over all regions."),
    ("global", "global_renewable_tech"): dict(
        symbol=r"RES_{k}", unit="[-]",
        text="1 if the technology counts as renewable in the global renewable constraints."),
    ("global", "global_min_renewable_production"): dict(
        symbol=r"\underline{RS}^{glob}_{y}", unit="[-]",
        text="Minimum production of renewable technologies over all regions, as a "
             "share of total final demand."),
    ("global", "global_min_renewable_electric_penetration"): dict(
        symbol=r"\underline{RS}^{glob,el}_{y}", unit="[-]",
        text="Minimum renewable share of electricity production over all regions."),
    # --------------------------------------------------------- connections
    ("connections", "line_fixed_cost"): dict(
        symbol=r"FOM_{l,c,y}", unit="[M/(P·yr)]",
        text="Fixed cost per unit of line capacity and year."),
    ("connections", "line_var_cost"): dict(
        symbol=r"VOM_{l,c,y}", unit="[M/E]",
        text="Variable cost per unit of energy imported through the line."),
    ("connections", "line_residual_cap"): dict(
        symbol=r"RC_{l,c,y}", unit="[P]",
        text="Line capacity existing before the first modelled year."),
    ("connections", "line_eff"): dict(
        symbol=r"\eta_{l,c,y}", unit="[-]",
        text="Line efficiency per 100 units of length: the energy delivered is "
             r"the energy sent times \eta^{L/100}."),
    ("connections", "line_length"): dict(
        symbol=r"L_{l,c,y}", unit="[len]", text="Length of the line."),
    ("connections", "line_inv"): dict(
        symbol=r"INV_{l,c,y}", unit="[M/P]",
        text="Investment cost per unit of new line capacity, paid in full in the "
             "year of the decision (no annuities)."),
    ("connections", "line_decom_cost"): dict(
        symbol=r"DEC_{l,c,y}", unit="[M/P]",
        text="Cost per unit of line capacity reaching the end of its lifetime."),
    ("connections", "line_lifetime"): dict(
        symbol=r"LT_{l,c}", unit="[yr]", text="Technical lifetime of new line capacity."),
    ("connections", "line_economic_lifetime"): dict(
        symbol=r"EL_{l,c}", unit="[yr]",
        text="Economic lifetime of lines. Read but currently not used by the model."),
    ("connections", "line_time_of_construction"): dict(
        symbol=r"TC_{l,c}", unit="[yr]",
        text="Years between the decision and the availability of new line capacity."),
    ("connections", "interest_rate"): dict(
        symbol=r"i_{l,c}", unit="[1/yr]",
        text="Interest rate of lines. Read but currently not used by the model."),
    ("connections", "annualprod_per_unitcapacity"): dict(
        symbol=r"APC_{l,c}", unit="[E/(P·yr)]",
        text="Energy one unit of line capacity can carry in a year at full load."),
    ("connections", "line_capacity_factor"): dict(
        symbol=r"CF_{l,c,y}", unit="[-]",
        text="Maximum annual use of the line as a fraction of its full-load energy."),
    ("connections", "line_maxtotcap"): dict(
        symbol=r"\overline{TC}_{l,c,y}", unit="[P]", text="Maximum total line capacity."),
    ("connections", "line_mintotcap"): dict(
        symbol=r"\underline{TC}_{l,c,y}", unit="[P]", text="Minimum total line capacity."),
    ("connections", "line_max_newcap"): dict(
        symbol=r"\overline{NC}_{l,c,y}", unit="[P]", text="Maximum new line capacity decided in a year."),
    ("connections", "line_min_newcap"): dict(
        symbol=r"\underline{NC}_{l,c,y}", unit="[P]", text="Minimum new line capacity decided in a year."),
}

# Sheets read only if present (they are not in the templates).
OPTIONAL_SHEETS = {
    "Capacity_group_membership": dict(
        symbol=r"G_g", unit="names",
        text="Capacity groups: one row per (group, technology). A technology can "
             "belong to several groups."),
    "Max_totalcap_group": dict(
        symbol=r"\overline{TC}_{g,y}", unit="[P]",
        text="Maximum total capacity of all technologies of a group, summed over all "
             "regions. Every model year must be given for each group listed."),
    "Max_newcap_group": dict(
        symbol=r"\overline{NC}_{g,y}", unit="[P]",
        text="Maximum new capacity of a group in a year, summed over all regions."),
}

# Sheets that exist but that no shipped example model creates, so the
# automatic coverage check cannot find them. Keep this list short.
# Constraint class -> (equation labels in the formulation, short description).
# tests/test_docs.py fails if a class of ConstraintList has no entry here.
CONSTRAINTS = {
    "Balance": (["balance"], "carrier balance"),
    "BalanceUnMetDemand": (["balance-unmet"], "carrier balance with unmet demand"),
    "TradeBalance": (["trade-balance"], "imports equal the exports of the other region"),
    "ResourceTechAvailability": (["availability-step", "availability-annual"],
                                 "production within available capacity"),
    "TotalCapacityRegional": (["total-capacity-limits"], "limits on total capacity"),
    "NewCapacityRegional": (["new-capacity-limits"], "limits on new capacity"),
    "NewCapacityNonneg": (["newcap-nonneg"], "new capacity is not negative"),
    "TechEfficency": (["efficiency"], "output equals input times efficiency"),
    "AnnualProductionRegional": (["production-limits"], "limits on annual production"),
    "AnnualProductionGlobal": (["production-limits"], "limits on annual production over all regions"),
    "EmissionCapRegional": (["emission-cap-regional", "emission-cap-power"], "regional emission caps"),
    "EmissionCapGlobal": (["emission-cap-global"], "emission cap over all regions"),
    "EmissionConsumedRegional": (["captured-emissions"], "limit on captured emissions"),
    "StorageMaxMinChange": (["soc-limits"], "state of charge within limits"),
    "StorageMaxFlowInOut": (["storage-flows", "storage-annual-cycle"], "storage power and annual cycle"),
    "StorageCyclicBoundary": (["storage-cyclic"], "state of charge at the end of each cycle"),
    "LineTotalCapacity": (["line-capacity-limits"], "limits on total line capacity"),
    "LineNewCapacity": (["line-capacity-limits"], "limits on new line capacity"),
    "LineAvailability": (["line-flows"], "line flows within capacity"),
    "TotalCapacityGlobal": (["global-capacity-limits"], "total capacity over all regions"),
    "NewCapacityGlobal": (["global-capacity-limits"], "new capacity over all regions"),
    "TotalCapacityGroup": (["capacity-groups"], "total capacity of a group"),
    "NewCapacityGroup": (["capacity-groups"], "new capacity of a group"),
    "RenewableProductionRegional": (["renewable-electricity", "renewable-total"], "minimum renewable shares"),
    "RenewableProductionGlobal": (["renewable-electricity", "renewable-total"],
                                  "minimum renewable shares over all regions"),
    "LandUsageRegional": (["land-use"], "land use per region"),
    "LandUsageGlobal": (["land-use"], "land use over all regions"),
    "CHPOperatingRange": (["chp"], "CHP heat and power operating region"),
    "ProductionRamp": (["ramp-amount", "ramp"], "ramp limits"),
    "MinTechPowerOutput": (["unit-commitment", "min-output"], "minimum output, unit commitment"),
    "ElectrolysisConsumption": (["electrolysis"], "electrolysis from renewable electricity"),
    "BooleanForStorage": (["boolean-storage"], "no simultaneous charge and discharge"),
}
