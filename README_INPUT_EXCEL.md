# Hypatia Excel input guide

This guide describes the Excel inputs used by the current Hypatia source.
The source code and newly generated parameter templates remain authoritative:
sheet names, index levels, column levels, and labels must match exactly.

## 1. Scenario layout

A scenario normally has this structure:

```text
scenario/
|-- sets/
|   |-- global.xlsx
|   `-- <region>.xlsx
|-- parameters/
|   |-- parameters_<region>.xlsx
|   |-- parameters_global.xlsx       # multi-region only
|   `-- parameters_connections.xlsx  # multi-region only
`-- results/
```

The region code in each file name must match the `Region` value in the
`Regions` table. For example, region `reg1` requires `sets/reg1.xlsx` and
`parameters/parameters_reg1.xlsx`.

## 2. Recommended workflow

The parameter workbooks depend on the model mode, topology, technologies, and
enabled optional constraints. The safest workflow is:

1. prepare the set workbooks;
2. construct the model with the required mode and optional constraints;
3. generate fresh parameter templates in an empty directory;
4. fill the templates without changing their index or header structure;
5. read the completed parameters and run the model;
6. write results to a separate directory.

For a Planning model with the minimum-output/unit-commitment constraint:

```python
from hypatia import Model

model = Model(
    path="scenario/sets",
    mode="Planning",
    optimization="Single",
    ensure_feasibility="Yes",
    optional_constraints=["min_tech_power_output"],
)
model.create_data_excels("scenario/parameter_template")
model.read_input_data("scenario/parameters")
model.run(solver="GUROBI")
model.to_csv(
    "scenario/results",
    postprocessing_module="aggregated",
)
```

Enabling an optional constraint is important when the templates are created:
Hypatia generates and reads only the parameter sheets required by the active
constraint set.

## 3. Set workbooks

Every set workbook contains a worksheet named `Sets`. Data areas are Excel
tables with exact table names; their position in the worksheet is not the
identifier.

### `sets/global.xlsx`

| Excel table | Main columns | Purpose |
| --- | --- | --- |
| `Regions` | `Region`, `Region_name` | Region codes and labels. |
| `Years` | `Year`, `Year_name` | Internal year codes and displayed years. |
| `Timesteps` | `Timeslice`, `Timeslice_name`, `Timeslice_fraction` | Intra-year time representation. |
| `Carriers_glob` | `Carrier`, `Carr_name`, `Carr_type`, `Carr_unit` | Global carriers and units. |
| `Technologies_glob` | `Technology`, `Tech_name`, `Tech_category`, `Tech_cap_unit`, `Tech_act_unit` | Global technologies and units. |
| `Emissions` | `Emission`, `Emission_name`, `Emission_unit` | Emission types and units. |

Technology categories are `Supply`, `Conversion`, `Conversion_plus`,
`Transmission`, `Demand`, and `Storage`. Carrier types are `Resource`,
`Intermediate`, and `Demand`.

Codes, capitalization, underscores, and ordering are significant. Unit fields
are metadata: Hypatia does not automatically convert MW to GW, MWh to GWh, or
mass units.

### `sets/<region>.xlsx`

| Excel table | Main columns | Purpose |
| --- | --- | --- |
| `Technologies` | `Technology`, `Tech_name`, `Tech_category` | Technologies available in the region. |
| `Carriers` | `Carrier`, `Carr_name`, `Carr_type` | Regional carriers. |
| `Carrier_input` | `Technology`, `Carrier_in` | Technology input-carrier mapping. |
| `Carrier_output` | `Technology`, `Carrier_out` | Technology output-carrier mapping. |

All mapped technologies and carriers must exist in the corresponding regional
tables and be consistent with the global definitions.

## 4. Parameter workbooks

Parameter worksheets are pandas matrices serialized to Excel. Many use
multi-level rows or columns. Do not remove apparently blank header cells, add
note columns inside a matrix, rename labels, or leave empty cells. A numerical
zero is an input value; an empty cell becomes `NaN` and is rejected.

Common Planning sheets include:

| Group | Sheets |
| --- | --- |
| Capacity | `Residual_capacity`, `Min_totalcap`, `Max_totalcap`, `Min_newcap`, `Max_newcap` |
| Investment | `INV`, `Decom_cost`, `Tech_lifetime`, `Economic_lifetime`, `Time_of_construction`, `Interest_rate`, `Discount_rate` |
| Operation | `F_OM`, `V_OM`, `AnnualProd_perunit_capacity`, `Tech_availability` |
| Time profiles | `Demand`, `capacity_factor_resource` |
| Conversion | `Tech_efficiency` and category-specific carrier-ratio sheets |
| Emissions | `Specific_emission`, `Emission_capture_efficiency`, `Emission_tax`, emission caps |
| Modular investment | `Modular_cap_unit` |
| Optional minimum output | `Min_power_output` |

Storage, `Conversion_plus`, transmission, CHP, ramping, and other optional
features add their own sheets. Generate a template for the exact configuration
instead of copying sheets from a different scenario.

For multi-region models, shared parameters and constraints are stored in
`parameters_global.xlsx`; connections are stored in
`parameters_connections.xlsx`. In this configuration `Modular_cap_unit` and
`Min_power_output` are global technology parameters.

## 5. Modular capacity and minimum output

For technology `i` in Planning mode:

| `Modular_cap_unit[i]` | `Min_power_output[i]` | Behaviour |
| ---: | ---: | --- |
| `0` | `0` | Continuous investment; no minimum-output commitment. |
| `> 0` | `0` | Integer modular investment; continuous dispatch. |
| `0` | `> 0` | Legacy aggregate binary commitment. |
| `> 0` | `> 0` | Integer modular investment and integer online-unit commitment. |

The minimum-output sheet is included only when the model is constructed with
`optional_constraints=["min_tech_power_output"]`.

In Operation mode, installed capacity is exogenous and `Modular_cap_unit` is
not an input. Enabling `min_tech_power_output` therefore uses the legacy
aggregate binary formulation and does not create online-unit variables.

Recommended checks:

- use `0 <= Min_power_output <= 1`;
- express `Modular_cap_unit` in the same capacity unit as total capacity;
- make capacity bounds compatible with the selected module size;
- decide whether residual capacity must also represent a whole number of
  modules (the model does not enforce this automatically);
- use a MILP-capable solver;
- apply cluster commitment to the physical core of a clustered CHP model,
  rather than duplicating it automatically on every output branch.

## 6. Units and time weighting

Available activity in a timestep is based on

```math
Q^{avail}_{y,t,i} = C_{y,i}\,CF_{y,t,i}\,f_t\,A_i,
```

where `C` is capacity, `CF` is `capacity_factor_resource`, `f` is
`Timeslice_fraction`, and `A` is `AnnualProd_perunit_capacity`. `Q` and the
`Demand` sheet must use a consistent activity unit. Hypatia does not apply an
automatic unit conversion.

## 7. Result file

For a Planning run, aggregated post-processing writes
`modular_unit_commitment.csv`. It is empty when no technology has both a
positive module size and a positive minimum-output fraction. Otherwise it
contains year, timeslice, region, technology, online units, installed units,
new units, and production.

Before solving, compare the completed inputs with a freshly generated
template, check for missing values, verify time-fraction conventions and
units, and confirm that the selected solver supports integer variables.
