# Modular unit commitment formulation

## Scope

This extension combines Hypatia's existing modular investment decision with
an integer count of online modules in each modelled timestep. It is enabled by
the existing optional constraint:

```python
Model(..., optional_constraints=["min_tech_power_output"])
```

The modular formulation is active only in `Planning` mode and only for a
technology for which both `Modular_cap_unit > 0` and
`Min_power_output > 0`. Non-modular technologies retain the legacy aggregate
binary minimum-output formulation. `Operation` mode also retains that legacy
formulation and never reads the Planning-only modular-capacity parameter.

This is a clustered, static commitment formulation. It does not add startup or
shutdown variables, startup costs, minimum up/down times, chronological state
transitions, or ramping equations.

## Symbols

| Symbol | Meaning | Domain/unit |
| --- | --- | --- |
| `r` | Region | set of regions |
| `y` | Model year | set of years |
| `t` | Intra-year timestep | set of timesteps |
| `i` | Technology | technology set |
| `s_i` | Module size (`Modular_cap_unit`) | capacity |
| `k_{y,i}` | New modules selected | non-negative integer |
| `X_{y,i}` | New capacity | capacity |
| `C_{y,i}` | Available total capacity | capacity |
| `n_{y,t,i}` | Online modules | non-negative integer |
| `Q_{y,t,i}` | Technology production/activity | activity |
| `A_i` | `AnnualProd_perunit_capacity` | activity per capacity-year |
| `CF_{y,t,i}` | `capacity_factor_resource` | dimensionless |
| `f_t` | `Timeslice_fraction` | dimensionless |
| `p_i^{min}` | `Min_power_output` | dimensionless |

## Modular investment

The pre-existing modular investment formulation is

```math
k_{y,i} \in \mathbb{Z}_{\ge 0},
```

```math
X_{y,i} = s_i k_{y,i} \qquad \text{if } s_i > 0.
```

When `s_i = 0`, new capacity remains continuous. Modular investment applies to
new capacity; residual capacity is not automatically forced to be a whole
multiple of `s_i`.

## Online-unit variable

For eligible modular technologies, Hypatia creates

```math
n_{y,t,i} \in \mathbb{Z}_{\ge 0}.
```

The variable has one row per `(year, timestep)` and one column per eligible
technology. Sparse technology-name metadata preserves the mapping between
columns and technologies.

The number of committed modules cannot exceed installed capacity:

```math
s_i n_{y,t,i} \le C_{y,i}.
```

This constraint permits residual capacity that is not a whole module, but only
whole modules can be committed by `n`.

## Per-module available production

The production available from one online module is

```math
\bar q_{y,t,i}^{unit}
= s_i\,CF_{y,t,i}\,f_t\,A_i.
```

The upper dispatch bound is

```math
Q_{y,t,i}
\le \bar q_{y,t,i}^{unit} n_{y,t,i},
```

and the minimum stable-output bound is

```math
Q_{y,t,i}
\ge p_i^{min}\bar q_{y,t,i}^{unit} n_{y,t,i}.
```

Therefore `n = 0` forces production to zero. For `n > 0`, aggregate production
lies between the minimum and maximum available production of the committed
modules.

## Non-modular and Operation fallback

For a technology with `Min_power_output > 0` but no positive modular-capacity
step, the existing binary variable `z_{y,t,i}` is used. Conceptually:

```math
Q_{y,t,i} \le M z_{y,t,i},
```

```math
Q_{y,t,i}
\ge p_i^{min}Q_{y,t,i}^{avail} - M(1-z_{y,t,i}).
```

This Big-M fallback is also used in `Operation` mode. It preserves the former
behaviour and avoids reading `Modular_cap_unit`, which is unavailable in that
mode. The Big-M formulation may be numerically weaker than the modular bounds.

## Activation logic

| Mode | Optional constraint | Module size | Minimum output | Result |
| --- | --- | ---: | ---: | --- |
| Planning | disabled | any | any | No minimum-output constraints or online-unit variables. |
| Planning | enabled | `> 0` | `> 0` | Integer modular commitment. |
| Planning | enabled | `0` | `> 0` | Aggregate binary fallback. |
| Planning | enabled | `> 0` | `0` | Modular investment only. |
| Operation | enabled | unavailable | `> 0` | Aggregate binary fallback. |

## Output

Planning results carry `units_online` and its technology-column mapping.
Aggregated post-processing exports `modular_unit_commitment.csv` with:

- `Year` and `Timeslice`;
- `Region` and `Technology`;
- `Units_online`;
- `Installed_units = Total_capacity / Modular_cap_unit`;
- `New_units = New_capacity / Modular_cap_unit`;
- `Core_production`.

The file has headers but no rows when no technology is eligible.

## Implementation map

| File | Responsibility |
| --- | --- |
| `hypatia/backend/ModelVariables.py` | Select eligible technologies and create integer online-unit variables. |
| `hypatia/backend/constraints/MinTechPowerOutput.py` | Apply modular bounds or the aggregate fallback. |
| `hypatia/backend/Build.py` | Include modular variables and metadata in Planning results. |
| `hypatia/postprocessing/AggregatedPostProcessing.py` | Build the diagnostic CSV table. |
| `tests/unit/test_modular_unit_commitment.py` | Check Planning eligibility and Operation isolation. |

## Current limitations

This implementation represents how many identical modules are online within
each aggregated timestep. It is not a full chronological unit-commitment
model. In particular:

- online decisions are independent between timesteps;
- there are no startup/shutdown costs or trajectories;
- there are no minimum up/down times;
- production ramping is a separate optional constraint and is not reformulated
  per committed module;
- all modules of a technology share efficiency, capacity factor, and costs;
- residual capacity is not required to be an integer multiple of module size.

These limitations should be considered when interpreting clustered or
representative-timestep results.
