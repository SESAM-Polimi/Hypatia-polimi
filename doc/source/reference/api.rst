.. _api:

API reference
=============

The public interface of Hypatia, importable from the package:
``from hypatia import Model, load_example, ...``.

Model
-----

.. autoclass:: hypatia.Model
   :members: create_data_excels, read_input_data, run, run_MinEm, run_MO, to_csv,
             create_config_file, settings

Examples
--------

.. autofunction:: hypatia.available_examples

.. autofunction:: hypatia.load_example

.. autofunction:: hypatia.download_example

Results
-------

.. autofunction:: hypatia.Merge_results

.. autoclass:: hypatia.Plotter
   :members: plot_total_capacity, plot_new_capacity, plot_prod_by_tech,
             plot_use_by_technology, plot_fuel_prod_cons, plot_emissions,
             plot_hourly_prod_by_tech, plot_regional_costs

Exceptions
----------

.. automodule:: hypatia.error_log.Exceptions
   :members:
