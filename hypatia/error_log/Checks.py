# -*- coding: utf-8 -*-
"""
This module contains a series of functions for checking the
possible mistakes in the input data to raise
the apporpriate error and inform the user.
"""

import warnings

import numpy as np
import pandas as pd

from hypatia.error_log.Exceptions import (
    WrongIndex,
    NanValues,
    WrongTableName,
    WrongSheetName,
    WrongMappingData,
    WrongMappingTech,
    WrongTechCategory,
    WrongCarrierType,
    WrongNumberOfYears,
)


def check_index(index, table_name, file_name, allowed_indices):
    """Checks if the columns in the set files have valid names
    """

    if not index.equals(allowed_indices):

        raise WrongIndex(
            f"The index/column {list(index)} in the 'table_name' table of '{file_name}' file includes invalid names\
                                       or misses an index/column.Valid index/column names are {list(allowed_indices)}"
        )


def check_index_data(index, sheet_name, file_name, allowed_indices):
    """Checks if the indices and columns in the parameter files have valid names
    """

    if not index.equals(allowed_indices):
        print(index, allowed_indices)
        raise WrongIndex(
            f"The index/column {list(index)} in the '{sheet_name}' sheet of '{file_name}' file includes invalid names\
                                       or misses an index/column.Valid index/column names are {list(allowed_indices)}"
        )


def check_nan(table_name, table_value, file_name):
    """Checks if there are nan values in the set and parameter tables and sheets
    """

    if table_value.isnull().values.any():

        raise NanValues(
            f"There are Nan values in '{table_name}' table in the '{file_name}' file.\
                        Check for missing or extra rows and columns"
        )


def check_table_name(
    file_name, allowed_names, table_name,
):

    """Checks if the tables in the set files have valid names
    """

    if table_name not in allowed_names:

        raise WrongTableName(
            f"The table name '{table_name}' in the '{file_name}' file is not a valid name.\
                               valid table names are {allowed_names}"
        )


def check_sheet_name(path, file_name, ids):

    """Checks if the sheets in the parameter files have valid names when 
    reading the data
    """

    file = pd.ExcelFile(r"{}/{}.xlsx".format(path, file_name))
    given_sheets = set(file.sheet_names)

    neccessary_sheets = set()
    for value in ids.values():
        neccessary_sheets.add(value["sheet_name"])

    differences = neccessary_sheets.difference(given_sheets)

    if differences:
        raise WrongSheetName(
            f"Following necessary sheets are missing in {file_name}. \n {differences}"
        )


def check_tech_category(tech_table, allowed_categories, file_name):

    """Checks if the given technology categories are within the acceptable
    tech categories of the model
    """

    for tech_ctgry in tech_table["Tech_category"]:

        if tech_ctgry not in allowed_categories:

            raise WrongTechCategory(
                f"'{tech_ctgry}' category in Technologies table\
                                    in '{file_name}' file is not a valid technology category.\
                                        Valied categories are {allowed_categories}"
            )


def check_carrier_type(carr_table, allowed_types, file_name):

    """Checks if the given carrier types are within the acceptable
    carrier types of the model
    """

    for carr_type in carr_table["Carr_type"]:

        if carr_type not in allowed_types:

            raise WrongCarrierType(
                f"'{carr_type}' type in Carriers table in '{file_name}' file is not a valid carrier type. Valied types are {allowed_types}"
            )


def check_mapping_values(
    mapping_table,
    mapping_name,
    reference_table,
    reference_name,
    mapping_col,
    reference_col,
    file_name,
):

    """Checks if the specified carriers and technologies in the carrier input
    and carrier output tables have valid names
    """

    if not set(mapping_table[mapping_col]).issubset(
        set(reference_table[reference_col])
    ):
        
        raise WrongMappingData(
            f"There is an invalid name in '{mapping_col}' column of '{mapping_name}' table, in the '{file_name}'\
                               set file. Check the consistency between '{mapping_name}' table and '{reference_name}' table"
        )


def check_mapping_ctgry(
    mapping_table, table_name, technologies_table, tech_ctgry, file_name
):

    """Checks if there are valid technology categories in the carrier input and
    carrier output tables
    """
    for tech in mapping_table["Technology"]:

        if (
            technologies_table.loc[technologies_table["Technology"] == tech][
                "Tech_category"
            ].item()
            == tech_ctgry
        ):

            raise WrongMappingTech(
                f"Technologies with '{tech_ctgry}' category cannot be included in '{table_name}' table.\
                                   Modify '{table_name}' table in the '{file_name}' set file."
            )


def check_years_mode_consistency(mode, main_years):

    """Checks that an Operation model has exactly one year.

    ``mode`` is a ModelMode or its name ("Operation", "Planning").
    """

    mode_name = getattr(mode, "value", mode)
    if mode_name == "Operation" and len(main_years) != 1:
        raise WrongNumberOfYears(
            f"An Operation model covers exactly one year, but the 'Years' table of "
            f"global.xlsx has {len(main_years)} ({list(main_years)}). Use mode='Planning' "
            f"for several years, or keep one year in the sets."
        )


def check_demand_technology(technologies_table, file_name):

    """Checks that a region has at least one Demand technology.

    The parameter templates and the carrier balance need one; regions without
    demand (supply-only or transit regions) are not supported yet.
    """

    if "Demand" not in set(technologies_table["Tech_category"]):
        raise WrongTechCategory(
            f"The region '{file_name}' has no technology of category 'Demand' in the "
            f"'Technologies' table of '{file_name}.xlsx'. Every region needs at least one "
            f"(regions without demand are not supported yet)."
        )


def check_timeslice_fractions(fractions, tolerance=1e-6):

    """Warns when the time-step fractions of a year do not add up to one.

    The fractions scale capacity into energy in every time step, so a sum other
    than one changes how much a unit of capacity can produce in a year. This is
    a warning, not an error, because a partial year can be intended.
    """

    total = float(np.sum(fractions))
    if abs(total - 1) > tolerance:
        warnings.warn(
            f"The time-step fractions in the 'Timesteps' table of global.xlsx add up to "
            f"{total:.6g}, not 1: each unit of capacity can produce {total:.6g} times its "
            f"annual output per year.",
            UserWarning,
            stacklevel=4,
        )


#%%

# %%
