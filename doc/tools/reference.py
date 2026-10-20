"""
Generate the parameter tables of the documentation from Hypatia's own code.

Which sheets exist, in which file, with which default, rows and columns, is
read from the templates Hypatia uses for create_data_excels, for a set of
synthetic model with one technology of every category, in every
configuration (Planning/Operation, one/several regions, every optional
constraint). Descriptions, symbols and units come from parameter_docs.py.

Run by doc/source/conf.py at every build, so the tables cannot drift from the
code. Also used by tests/test_docs.py to check that every sheet is documented.
"""
import os
import sys
import warnings
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from parameter_docs import CONSTRAINTS, OPTIONAL_SHEETS, PARAMETERS  # noqa: E402

FILES = {
    "regional": "parameters_<region>.xlsx",
    "global": "parameters_global.xlsx",
    "connections": "parameters_connections.xlsx",
}

# A synthetic model with one technology of every category. It is built in
# memory (no Excel files), so the configurations below are complete by
# construction instead of depending on what the shipped examples contain.
TECHNOLOGIES = [  # name, category, input carriers, output carriers
    ("sup", "Supply", [], ["c1"]),
    ("conv", "Conversion", ["c1"], ["c2"]),
    ("cplus", "Conversion_plus", ["c1", "c2"], ["c2", "c3"]),
    ("trans", "Transmission", ["c2"], ["c3"]),
    ("sto", "Storage", ["c3"], ["c3"]),
    ("dem", "Demand", ["c3"], []),
]
CARRIERS = [("c1", "Resource"), ("c2", "Intermediate"), ("c3", "Demand")]
CATEGORIES = [category for _, category, _, _ in TECHNOLOGIES]
MODES = ["Planning", "Operation"]
REQUIRED_CATEGORIES = set()  # filled by collect()
TOPOLOGIES = {"one region": ["reg1"], "several regions": ["reg1", "reg2"]}


def synthetic_settings(mode, regions, without=(), options=()):
    """ModelSettings of the synthetic model, leaving out the categories in `without`."""
    from hypatia.backend.ModelSettings import ModelSettings
    from hypatia.utility.constants import EnsureFeasibility, ModelMode, OptimizationMode

    techs = [t for t in TECHNOLOGIES if t[1] not in without]
    names = [t[0] for t in techs]
    categories = [t[1] for t in techs]
    years = ["Y0"] if mode == "Operation" else ["Y0", "Y1", "Y2"]
    carriers = {"Carrier": [c for c, _ in CARRIERS], "Carr_name": [c for c, _ in CARRIERS],
                "Carr_type": [t for _, t in CARRIERS]}
    global_settings = {
        "Regions": pd.DataFrame({"Region": regions, "Region_name": regions}),
        "Years": pd.DataFrame({"Year": years, "Year_name": [2030 + i for i in range(len(years))]}),
        "Timesteps": pd.DataFrame({"Timeslice": [1, 2], "Timeslice_name": ["a", "b"],
                                   "Timeslice_fraction": [0.5, 0.5]}),
        "Carriers_glob": pd.DataFrame({**carriers, "Carr_unit": ["u"] * len(CARRIERS)}),
        "Technologies_glob": pd.DataFrame({"Technology": names, "Tech_name": names,
                                           "Tech_category": categories,
                                           "Tech_cap_unit": ["u"] * len(techs),
                                           "Tech_act_unit": ["u"] * len(techs)}),
        "Emissions": pd.DataFrame({"Emission": ["CO2"], "Emission_name": ["CO2"],
                                   "Emission_unit": ["t"]}),
    }
    regional = {
        "Technologies": pd.DataFrame({"Technology": names, "Tech_name": names,
                                      "Tech_category": categories}),
        "Carriers": pd.DataFrame(carriers),
        "Carrier_input": pd.DataFrame([(t[0], c) for t in techs for c in t[2]],
                                      columns=["Technology", "Carrier_in"]),
        "Carrier_output": pd.DataFrame([(t[0], c) for t in techs for c in t[3]],
                                       columns=["Technology", "Carrier_out"]),
    }
    return ModelSettings(ModelMode[mode], OptimizationMode.Single, EnsureFeasibility.Yes,
                         global_settings, {r: regional for r in regions},
                         optional_constraints=list(options))


def _sheets(settings):
    """{(file, key): template entry} of every sheet the settings ask for."""
    out = {}
    for key, entry in (settings.global_parameters_template or {}).items():
        out[("global", key)] = entry
    for key, entry in (settings.trade_parameters_template or {}).items():
        out[("connections", key)] = entry
    for region in settings.regions:
        for key, entry in settings.regional_parameters_template[region].items():
            out.setdefault(("regional", key), entry)
    return out


def collect():
    """{(file, key): record} for every parameter sheet any model can ask for.

    A record says in which modes and topologies the sheet appears, which
    optional constraint (if any) adds it, and which technology category it
    needs: a sheet *needs* a category when it disappears from every
    configuration once technologies of that category are removed.
    """
    warnings.filterwarnings("ignore")
    from hypatia.backend.constraints.ConstraintList import OPTIONAL_CONSTRAINTS

    records = defaultdict(lambda: {"modes": set(), "topologies": set(), "options": set(),
                                   "categories": None, "template": None})
    for mode in MODES:
        for topology, regions in TOPOLOGIES.items():
            core = _sheets(synthetic_settings(mode, regions))
            per_option = {o: _sheets(synthetic_settings(mode, regions, options=[o]))
                          for o in OPTIONAL_CONSTRAINTS}
            everything = _sheets(synthetic_settings(mode, regions, options=list(OPTIONAL_CONSTRAINTS)))
            without = {}
            for c in CATEGORIES:
                try:
                    without[c] = _sheets(synthetic_settings(mode, regions, without=[c],
                                                            options=list(OPTIONAL_CONSTRAINTS)))
                except KeyError:
                    REQUIRED_CATEGORIES.add(c)  # Hypatia cannot build a model without it
            for sheet, entry in everything.items():
                record = records[sheet]
                record["modes"].add(mode)
                record["topologies"].add(topology)
                if sheet not in core:
                    record["options"] |= {o for o, s in per_option.items() if sheet in s}
                needs = {c for c in without if sheet not in without[c]}
                record["categories"] = needs if record["categories"] is None else record["categories"] & needs
                if record["template"] is None:
                    record["template"] = entry
    return dict(records)


def _levels(index):
    names = index.names if isinstance(index, pd.MultiIndex) else [index.name]
    return " / ".join(str(n) for n in names if n is not None) or "(one row)"


def _default(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "varies"
    if number.is_integer() and abs(number) < 1e6:
        return str(int(number))
    return f"{number:g}"


def undocumented(records):
    """Sheets created by the code but missing from parameter_docs.py, and stale entries."""
    missing = sorted(set(records) - set(PARAMETERS))
    stale = sorted(set(PARAMETERS) - set(records))
    return missing, stale


def _row(file, key, record):
    doc = PARAMETERS.get((file, key), {"symbol": "", "unit": "", "text": "(not documented)"})
    template = record["template"]
    needed = _needed_in(record)
    return [f"``{template['sheet_name']}``", f":math:`{doc['symbol']}`" if doc["symbol"] else "",
            doc["unit"], _default(template["value"]), _levels(template["index"]),
            _levels(template["columns"]), needed, doc["text"]]


def _needed_in(record):
    """Plain-language conditions under which a sheet is asked for."""
    parts = ["both modes" if record["modes"] == set(MODES) else f"{next(iter(record['modes']))} only"]
    if len(record["topologies"]) == 1:
        parts.append(f"{next(iter(record['topologies']))}")
    if record["categories"]:
        parts.append("with " + " and ".join(sorted(record["categories"])) + " technologies")
    if record["options"]:
        parts.append("option " + ", ".join(f"``{o}``" for o in sorted(record["options"])))
    return "; ".join(parts)


def _table(title, rows, widths="16 10 8 8 10 12 12 24"):
    header = ["Sheet", "Symbol", "Unit", "Default", "Rows", "Columns", "Needed in", "Description"]
    lines = [f".. list-table:: {title}", f"   :widths: {widths}", "   :header-rows: 1",
             "   :class: longtable", ""]
    for row in [header] + rows:
        lines.append("   * - " + (row[0] or " "))
        for cell in row[1:]:
            lines.append("     - " + (str(cell) if cell != "" else " "))
    return "\n".join(lines) + "\n"


def write(out_dir, records=None):
    """Write input_tables.rst and nomenclature_parameters.rst into out_dir."""
    records = records if records is not None else collect()
    os.makedirs(out_dir, exist_ok=True)

    parts = [".. This file is generated by doc/tools/reference.py at every build. Do not edit.\n"]
    for file, label in FILES.items():
        rows = [_row(f, k, r) for (f, k), r in sorted(records.items(),
                key=lambda item: item[1]["template"]["sheet_name"].lower()) if f == file]
        parts.append(f".. _sheets-{file}:\n\n{label}\n{'~' * len(label)}\n\n")
        parts.append(_table(f"Sheets of {label}", rows))
    rows = [[f"``{name}``", f":math:`{d['symbol']}`", d["unit"], "(none)", "", "", "Planning",
             d["text"]] for name, d in OPTIONAL_SHEETS.items()]
    parts.append(".. _sheets-groups:\n\nCapacity groups (optional sheets)\n"
                 "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
                 "Read only if present: in ``parameters_global.xlsx`` for models with several "
                 "regions, in the regional file otherwise. They are plain tables with the "
                 "columns shown in :ref:`optional-capacity-groups`.\n\n")
    parts.append(_table("Capacity group sheets", rows))
    with open(os.path.join(out_dir, "input_tables.rst"), "w") as f:
        f.write("\n".join(parts))

    seen, rows = set(), []
    for (file, key), doc in sorted(PARAMETERS.items(), key=lambda item: item[1]["symbol"].lower()):
        sheet = (records[(file, key)]["template"]["sheet_name"] if (file, key) in records
                 else doc.get("sheet", key))
        signature = (doc["symbol"], sheet)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append([f":math:`{doc['symbol']}`", doc["unit"], f"``{sheet}`` ({file})", doc["text"]])
    lines = [".. This file is generated by doc/tools/reference.py at every build. Do not edit.\n",
             ".. list-table:: Parameters", "   :widths: 13 9 30 40", "   :header-rows: 1",
             "   :class: longtable", ""]
    for row in [["Symbol", "Unit", "Sheet (file)", "Meaning"]] + rows:
        lines.append("   * - " + row[0])
        for cell in row[1:]:
            lines.append("     - " + cell)
    with open(os.path.join(out_dir, "nomenclature_parameters.rst"), "w") as f:
        f.write("\n".join(lines) + "\n")

    from parameter_docs import UNITS
    unit_lines = [".. This file is generated by doc/tools/reference.py at every build. Do not edit.\n",
                  ".. list-table:: Unit symbols", "   :widths: 10 60", "   :header-rows: 1", "",
                  "   * - Symbol", "     - Meaning"]
    for symbol, meaning in UNITS.items():
        unit_lines += [f"   * - {symbol}", f"     - {meaning}"]
    with open(os.path.join(out_dir, "units.rst"), "w") as f:
        f.write("\n".join(unit_lines) + "\n")
    write_implementation_map(out_dir)
    return records


def constraint_classes():
    """{class name: (modes, topologies, feasibility, option name or None)} from ConstraintList."""
    from hypatia.backend.constraints.ConstraintList import CORE_CONSTRAINTS, OPTIONAL_CONSTRAINTS
    option_of = {cls.__name__: name for name, cls in OPTIONAL_CONSTRAINTS.items()}
    topology = {"SingleNode": "one region", "MultiNode": "several regions"}
    out = {}
    for cls in CORE_CONSTRAINTS + list(OPTIONAL_CONSTRAINTS.values()):
        modes = ", ".join(m.name for m in cls.MODES)
        topologies = ", ".join(topology[t.name] for t in cls.TOPOLOGY_TYPES)
        feasibility = {("No",): "ensure_feasibility No", ("Yes",): "ensure_feasibility Yes"}.get(
            tuple(f.name for f in cls.ENSURE_FEASIBILITY), "")
        out[cls.__name__] = (modes, topologies, feasibility, option_of.get(cls.__name__))
    return out


def write_implementation_map(out_dir):
    lines = [".. This file is generated by doc/tools/reference.py at every build. Do not edit.\n",
             ".. list-table:: Constraint classes",
             "   :widths: 22 18 18 30 18", "   :header-rows: 1", "   :class: longtable", "",
             "   * - Class", "     - Equations", "     - Modes", "     - Applies to", "     - Enabled by"]
    for name, (modes, topologies, feasibility, option) in constraint_classes().items():
        labels, _ = CONSTRAINTS.get(name, ([], ""))
        equations = ", ".join(f":eq:`{label}`" for label in labels) or "(not documented)"
        applies = topologies + (f"; {feasibility}" if feasibility else "")
        enabled = f"``{option}``" if option else ("group sheets" if "Group" in name else "always")
        lines += [f"   * - ``{name}``", f"     - {equations}", f"     - {modes}",
                  f"     - {applies}", f"     - {enabled}"]
    with open(os.path.join(out_dir, "implementation_map.rst"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    records = collect()
    missing, stale = undocumented(records)
    print(f"{len(records)} sheets; undocumented: {missing}; stale entries: {stale}")
    if len(sys.argv) > 1:
        write(sys.argv[1], records)
