.. _input-reference:

Input reference
===============

This page lists every sheet of the parameter files. The tables are generated
from Hypatia's code at every build of this documentation: the sheet names,
defaults, indices and the conditions under which each sheet appears are read
from the templates that :meth:`~hypatia.Model.create_data_excels` writes, so
they always match the installed version. The symbols link each sheet to the
equations of the :ref:`formulation`, where every constraint that uses it is
written out.

How to read the tables
----------------------

Sheet
   The worksheet name in the parameter file.
Symbol
   The parameter in the :ref:`formulation`. Indices: :math:`r` region,
   :math:`k` technology, :math:`c` carrier, :math:`l` line, :math:`e`
   emission type, :math:`y` year, :math:`t` time step.
Unit
   The dimension of the values, in the symbols of the table below. Hypatia
   lets you choose units carrier by carrier, so units are given as
   dimensions (energy, power, money) rather than as MWh or EUR; [-] marks a
   dimensionless value (a fraction, a share, a ratio of equal units).
Default
   The value written in the template. A default is a placeholder, not a
   recommendation; "varies" marks sheets whose template is not a single
   number.
Rows, Columns
   The index levels of the sheet, from the outermost. ``Years / Timesteps``
   means one row per time step of every year; ``Tech_category /
   Technology`` means two header rows, the category above the technology.
Needed in
   When the sheet is in the files: in which mode, with one region or
   several, when a technology category must be present, and which optional
   constraint adds it.

.. include:: ../_generated/units.rst

Here "energy" is the activity unit of a technology or carrier, and "power"
its capacity unit: for a power plant typically MWh and MW, for a refinery
barrels and barrels per day. One unit of capacity running for a whole year
produces ``AnnualProd_perunit_capacity`` units of energy.

Parameter files
---------------

.. include:: ../_generated/input_tables.rst

Set files
---------

The set files are described in :ref:`building-a-model`.
