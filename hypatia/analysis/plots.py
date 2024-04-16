# -*- coding: utf-8 -*-
"""
This module contains the functions and the class for
result visualization of hypatia model
"""

from copy import deepcopy
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as pltly
import plotly.graph_objects as go
from IPython import get_ipython
from hypatia.analysis.postprocessing import year_slice_index
import plotly.io as io
io.renderers.default='browser'

CARS = "Carriers"
TECHS = "Technologies"
CARS_IN = "Carrier_input"
CARS_OUT = "Carrier_output"

main_index = {
    CARS: "Carr_type",
    TECHS: "Tech_category",
    CARS_IN: "Carrier_in",
    CARS_OUT: "Carrier_out",
}


def run_from_jupyter():
    if get_ipython() is not None:
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return True
        return False


def _plotter(fig, layout, path):

    fig.update_layout(layout)
    if run_from_jupyter():
        pltly.init_notebook_mode(connected=False)
        pltly.iplot({"data": fig.data, "layout": fig.layout})
    fig.write_html(path)


def get_data(panda_data, _format="list"):

    if _format == "list":
        try:
            return panda_data.tolist()
        except AttributeError:
            return [panda_data]
    elif _format == "numpy":
        try:
            return panda_data.values()
        except AttributeError:
            return [panda_data]


def plot(kind, **properties):

    if kind == "bar":
        return go.Bar(**properties)
    elif kind == "area":
        properties["line"] = dict(width=0)
        properties["mode"] = "lines"
        properties["stackgroup"] = "one"

        y = properties["y"]

        if np.any(y < 0):
            y_pos = y.copy()
            y_pos[y_pos < 0] = 0
            properties["y"] = y_pos
            fig = go.Scatter(**properties)

            y_neg = y.copy()
            y_neg[y_neg > 0] = 0
            properties["stackgroup"] = "two"
            properties["y"] = y_neg
            fig = go.Scatter(**properties)

            return fig

        fig = go.Scatter(**properties)

        return fig
    elif kind == "pie":
        return go.Pie(**properties)
    else:
        raise ValueError(f"{kind} is not an acceptable kind.")
    

def set_steps(fig, counter, mode):

    steps = []
    for name, step_upper in counter:
        if name == counter[0][0]:
            step_lower = 0
        steps.append(
            dict(
                label=name,
                method="update",
                args=[
                    {
                        "visible": [
                            True if step_lower <= i < step_upper else False
                            for i in range(len(fig.data))
                        ]
                    }
                ],
            )
        )

        step_lower = step_upper
    if mode == "sliders":
        fig.update_layout({mode: [dict(active=0, steps=steps, pad={"t": 50},)]})
    elif mode == "updatemenus":
        fig.update_layout({mode: [dict(active=0, buttons=steps, pad={"t": 50})]})
    else:
        raise ValueError("Acceptable modes are 'sliders' and 'updatemenus'.")
    return fig


def visible(step_index, aggregate):
    if not step_index:
        return True
    if aggregate:
        return True
    return False


def str2ls(inp):
    if isinstance(inp, str):
        return [inp]
    return inp


class Plotter:

    """A Plotting tool for hypatia"""

    def __init__(self, results, config, hourly_resolution):
        """
        Parameters
        ----------
        results : str, hypatia.Model
            Initializing the plotter object by csv files or directly from
            hypatia model.

        config : str
            path to the excel file defining the config of the plots such as
            colors, names, groups and more.

        hourly_resolution : boolean
            if True, hourly plots will be available else hourly plots cant be used

        """

        if isinstance(results, str):
            self._init_from_csv(results)
        else:
            self._init_from_object(results)

        self._read_config_file(config)
        self._hourly = hourly_resolution
        
    def _init_from_object(self, results):
        """Extracts the data for plots"""

        self.data = deepcopy(results.results)
        self.regions = deepcopy(results.get_model_data().settings.regions)
        self.years = deepcopy(results.get_model_data().settings.years)
        self.time_steps = deepcopy(results.get_model_data().settings.time_steps)
        self.time_fraction = deepcopy(results.get_model_data().settings.timeslice_fraction)
        self.techs = deepcopy(
            results.get_model_data().settings.global_settings["Technologies_glob"]["Technology"].tolist()
        )
        # self.regionalsettings = deepcopy(results.get_model_data().settings.regional_settings)
        self.fuels = deepcopy(
            results.get_model_data().settings.global_settings["Carriers_glob"]["Carrier"].tolist()
        )
        self.emissions = deepcopy(
            results.get_model_data().settings.global_settings["Emissions"]["Emission"].tolist()
        )
        self.mode = deepcopy(
            results.get_model_data().settings.mode
        )
        self.settings = deepcopy(
            results.get_model_data().settings
            )

        reformed_sets = {}
        for region in self.regions:
            reformed_sets[region] = {}
            for key, value in results.get_model_data().settings.regional_settings[region].items():
                reformed_sets[region][key] = value.set_index(
                    [main_index[key]], inplace=False
                )
        self.sets = reformed_sets

        self.glob_mapping = dict(
            techs=results.get_model_data().settings.global_settings["Technologies_glob"].set_index(
                ["Technology"], inplace=False
            ),
            fuels=results.get_model_data().settings.global_settings["Carriers_glob"].set_index(
                ["Carrier"], inplace=False
            ),
            years=results.get_model_data().settings.global_settings["Years"].set_index(
                ["Year"], inplace=False
            ),
        )

        self.mapping = results.get_model_data().settings.regional_settings

        years = results.get_model_data().settings.years
        _years = results.get_model_data().settings.global_settings["Years"]
        years = _years[_years["Year"].isin(years)]["Year_name"]
        time_fraction = results.get_model_data().settings.time_steps
        year_slice = year_slice_index(years, time_fraction)

    def _init_from_csv(self, results):
        """Extracts the data from csv files"""
        raise NotImplementedError("Not implemented in this version")

    def _is_allowed(self, item):
        if item not in self.data:
            raise ValueError("This plot is not implementable for this specific case.")

    def _read_config_file(self, config):
        """Reads the config file and checks the consistency"""
        configs = {}
        if self.settings.multi_node:
            for sheet in ["Techs", "Importexport", "Fuels", "Regions", "Emissions"]:
                data = pd.read_excel(io=config, sheet_name=sheet, index_col=0, header=0,)
    
                if data.isnull().values.any():
                    raise Exception(
                        f"nans are not acceptable in the config file. sheet_name = {sheet}"
                    )
                configs[sheet.lower()] = data
        else:
            for sheet in ["Techs", "Fuels", "Regions", "Emissions"]:
                data = pd.read_excel(io=config, sheet_name=sheet, index_col=0, header=0,)
    
                if data.isnull().values.any():
                    raise Exception(
                        f"nans are not acceptable in the config file. sheet_name = {sheet}"
                    )
                configs[sheet.lower()] = data
        self.configs = configs

    def _take_units(self, _set, item, level):

        try:
            units = self.configs[_set].loc[item, level].unique()
        except AttributeError:
            units = [self.configs[_set].loc[item, level]]
        if len(units) != 1:
            raise Exception(
                "items with different units cant be plotted. "
                f"items {item} have multiple units {units}."
            )
        return units[0]

    def plot_use_by_technology(
        self,
        fuel_group,
        path,
        regions="all",
        kind="bar",
        mode="updatemenus",
        aggregate=False,
    ):
        """Plots the use of fuels by technologies

        Parameters
        ----------

        fuel_group : str
            the name of the fuel group to be plotted based on value given in config file

        regions : str, Optional
            the regions to plot.

        path : str
            where to save the file + the name of the file and extension

        kind : str
            type of plot ['bar','area']

        mode : str
            the mode of plot when more than one region exists in

        """
        
        indexer_time = pd.MultiIndex.from_product(
            [self.years, self.time_steps],
            names=["Years", "Timesteps"],
        )
        
        if regions == "all":
            regions = self.regions

        fuels = self.configs["fuels"]
        fuels = fuels[fuels["fuel_group"].isin(str2ls(fuel_group))].index
        
        if not len(fuels):
            raise ValueError("No fuel found for given fuel name .")
        unit = self._take_units("fuels", fuels, "fuel_unit")
        fig = go.Figure()
        counter = []

        # define if demand and intermediate exist
        tech_type = self.glob_mapping["fuels"].loc[fuels, "Carr_type"].tolist()

        # in aggregated mode, legends will be defined for whole regions
        if aggregate:
            legends = set()
        for step_index, region in enumerate(regions):

            # Identify what techs are consuming that fuel
            carrier_tech = self.mapping[region]["Carrier_input"]
            carrier_tech = carrier_tech[carrier_tech.Carrier_in.isin(fuels)][
                "Technology"
            ].tolist()

            # in disaggregated mode, legends will be separated for regions
            if not aggregate:
                legends = set()
            if "Demand" in tech_type:
                for tt, values in (
                    self.data.demand[region]
                    .groupby(level=0, sort=False)
                    .sum()
                    .items()
                ):
                    if tt not in carrier_tech:
                        continue
                    name = self.configs["techs"].loc[tt, "tech_name"]
                    color = self.configs["techs"].loc[tt, "tech_color"]

                    fig.add_trace(
                        plot(
                            kind=kind,
                            x=values.index,
                            y=values.values,
                            name=name,
                            marker_color=color,
                            visible=visible(step_index, aggregate),
                            showlegend=False if tt in legends else True,
                        ),
                    )

                    legends.add(tt)                 
            
            techuse = self.data.technology_use[region]

            for category, df in techuse.items():

                all_columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==category]["Technology"].values
                columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==category]
                columns = columns[columns.Technology.isin(carrier_tech)]["Technology"].to_list()
                
                if columns != []:
                    tech_use = pd.DataFrame(df.value, indexer_time, all_columns)
                    techuse = tech_use.filter(items=columns,axis=1)
                else:
                    continue

                if category == "Conversion":
                    
                    for tt, values in techuse.groupby(level=0, sort=False).sum().items():
                        if tt not in carrier_tech:
                            continue
                        name = self.configs["techs"].loc[tt, "tech_name"]
                        color = self.configs["techs"].loc[tt, "tech_color"]
                        
                        fig.add_trace(
                            plot(
                                kind=kind,
                                x=values.index,
                                y=values.values,
                                name=name,
                                marker_color=color,
                                visible=visible(step_index, aggregate),
                                showlegend=False if tt in legends else True,
                            )
                        )

                        legends.add(tt)
                if category == "Conversion_plus":

                    for tt, values in techuse.groupby(level=0, sort=False).sum().items():

                        if tt not in carrier_tech:
                            continue

                        ratio = self.data.carrier_ratio_in[region].loc[
                            :, (tt, slice(None))
                        ]

                        for cols, ratios in ratio.items():
                            if cols[1] in fuels:

                                name = self.configs["techs"].loc[tt, "tech_name"]
                                color = self.configs["techs"].loc[tt, "tech_color"]

                                to_plot = (
                                    (values * ratios).groupby(level=0, sort=False).sum()
                                )
                                fig.add_trace(
                                    plot(
                                        kind=kind,
                                        x=to_plot.index,
                                        y=to_plot.values,
                                        name=name,
                                        marker_color=color,
                                        visible=visible(step_index, aggregate),
                                        showlegend=False if tt in legends else True,
                                    )
                                )
                                legends.add(tt)

            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )
        if not aggregate:
            fig = set_steps(fig, counter, mode)
        layout = {
            "title": "Use by technologies",
            "yaxis": {"title": unit},
            "xaxis": dict(tickmode="array", tickvals=values.index, dtick=1),
        }

        if kind == "bar":
            layout["barmode"] = "relative"
        _plotter(fig=fig, layout=layout, path = path)
        # fig.show()

    def plot_new_capacity(
        self,
        path,
        tech_group,
        regions="all",
        kind="bar",
        cummulative=False,
        mode="updatemenus",
        aggregate=False,
    ):
        """Plotes new capacity of a group of technologies

        Parameters
        ----------
        path : str
            the path to save the file with the file extension

        tech_group : str
            the group of the techs to be plotted

        regions : str
            the regions to be plotted

        kind : str
            kind of the plot ['bar','area']

        cummulative : boolean
            if True, plots the cummulative new capacity else will plot new capacity

        mode : str
            defines the model of plots in case multiple restions are given
        """

        # self._is_allowed("real_new_capacity")

        if regions == "all":
            regions = self.regions
        techs = self.configs["techs"]
        techs = techs[techs["tech_group"].isin(str2ls(tech_group))].index

        if not len(techs):
            raise ValueError(f"No tech found for category {tech_group}.")
        unit = self._take_units("techs", techs, "tech_cap_unit")
        tech_type = self.glob_mapping["techs"].loc[techs]["Tech_category"]

        fig = go.Figure()
        counter = []
        if aggregate:
            legends = set()
        for step_index, region in enumerate(regions):
            if not aggregate:
                legends = set()
            for tt in tech_type.unique():
                if tt not in self.data.real_new_capacity[region]:
                    continue
                newcap = self.data.real_new_capacity[region][tt]
                residual = self.data.residual_capacity[region][tt]
                newcap = pd.DataFrame(newcap.value, residual.index, residual.columns)

                if cummulative:
                    newcap = newcap.cumsum()
                    
                for t, values in residual.items():
                    if t not in techs:
                        continue
                    name = self.configs["techs"].loc[t, "tech_name"]
                    color = self.configs["techs"].loc[t, "tech_color"]

                    fig.add_trace(
                        plot(
                            kind=kind,
                            name=name,
                            x=values.index,
                            y=newcap[t],
                            marker_color=color,
                            visible=visible(step_index, aggregate),
                            showlegend=False if t in legends else True,
                        )
                    )
                legends.add(t)
            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )
        layout = {
            "title": "Accumulated New Capacity" if cummulative else "New Capacity",
            "yaxis": {"title": unit},
            "xaxis": dict(tickmode="array", tickvals=values.index, dtick=1),
        }

        if not aggregate:
            fig = set_steps(fig, counter, mode)
        if kind == "bar":
            layout["barmode"] = "relative"
        _plotter(fig=fig, layout=layout, path=path)
        # fig.show()

    def plot_total_capacity(
        self,
        path,
        tech_group,
        regions="all",
        kind="bar",
        decom_cap=True,
        mode="updatemenus",
        aggregate=False,
    ):
        """Plots operative total capacity and total decomissioned capacity

        .. note::
            residual capacities are plotted with the same color but with lower copacity

        Parameters
        ----------
        path : str
            path to save the plot with the extension

        tech_group : str
            the group of the techs to be plotted

        regions : str
            the regions to be plotted

        kind : str
            kind of the plot ['bar','area']

        decom_cap : boolean
            if True, will plot the total decomissioned capacities as negative numbers
        """
        try:
            self._is_allowed("decommissioned_capacity")
        except ValueError:
            decom_cap = False
        techs = self.configs["techs"]
        techs = techs[techs["tech_group"].isin(str2ls(tech_group))].index

        if regions == "all":
            regions = self.regions

        if not len(techs):
            raise ValueError(f"No tech found for category {tech_group}.")
        unit = self._take_units("techs", techs, "tech_cap_unit")
        tech_type = self.glob_mapping["techs"].loc[techs]["Tech_category"]

        fig = go.Figure()
        counter = []
        if aggregate:
            legends = set()
        for step_index, region in enumerate(regions):
            if not aggregate:
                legends = set()
            for tt in tech_type.unique():
                if tt not in self.data.totalcapacity[region]:
                    continue
                
                residual = self.data.residual_capacity[region][tt]
                if self.mode.name == "Planning":
                    totcap = self.data.totalcapacity[region][tt]
                    totcap = pd.DataFrame(totcap.value, residual.index, residual.columns)
                    
                    to_plot = totcap - residual
                    for t, values in residual.items():
                        if t not in techs:
                            continue

                        name = self.configs["techs"].loc[t, "tech_name"]
                        color = self.configs["techs"].loc[t, "tech_color"]

                        name = f"{name}: residual capacity"

                        fig.add_trace(
                            plot(
                                kind=kind,
                                name=name,
                                x=values.index,
                                y=residual[t],
                                marker_color=color,
                                opacity=0.3,
                                visible=visible(step_index, aggregate),
                                showlegend=False if t in legends else True,
                            )
                        )

                        legends.add(name)
                        
                    for t, values in residual.items():
                        if t not in techs:
                            continue
                        
                        name = self.configs["techs"].loc[t, "tech_name"]
                        color = self.configs["techs"].loc[t, "tech_color"]

                        name = f"{name}: new capacity"

                        fig.add_trace(
                            plot(
                                kind=kind,
                                name=name,
                                x=values.index,
                                y=to_plot[t],
                                marker_color=color,
                                visible=visible(step_index, aggregate),
                                showlegend=False if name in legends else True,
                            )
                        )

                        legends.add(name)
                    if decom_cap:
                        to_plot = self.data.decommissioned_capacity[region][tt].cumsum()
                        residul_decom = residual.iloc[0, :] - residual

                        for t, values in to_plot.items():
                            if t not in techs:
                                continue
                            name = self.configs["techs"].loc[t, "tech_name"]
                            color = self.configs["techs"].loc[t, "tech_color"]

                            name = f"{name}: residual capacity"

                            fig.add_trace(
                                plot(
                                    kind=kind,
                                    name=name,
                                    x=values.index,
                                    y=-residul_decom[t],
                                    marker_color=color,
                                    opacity=0.3,
                                    visible=visible(step_index, aggregate),
                                    showlegend=False if name in legends else True,
                                )
                            )
                        for t, values in to_plot.items():
                            if t not in techs:
                                continue
                            name = self.configs["techs"].loc[t, "tech_name"]
                            color = self.configs["techs"].loc[t, "tech_color"]

                            name = f"{name}: new capacity"

                            fig.add_trace(
                                plot(
                                    kind=kind,
                                    name=name,
                                    x=values.index,
                                    y=-values.values,
                                    marker_color=color,
                                    opacity=0.5,
                                    visible=visible(step_index, aggregate),
                                    showlegend=False if name in legends else True,
                                )
                            )

                            legends.add(name)
                else:
                    totcap = residual
                    totcap = pd.DataFrame(totcap.values, residual.index, residual.columns)
                    
                    to_plot = totcap
                    # xx = []
                    for t, values in to_plot.items():
                        if t not in techs:
                            continue
                        x = []
                        name = self.configs["techs"].loc[t, "tech_name"]
                        color = self.configs["techs"].loc[t, "tech_color"]

                        name = f"{name}"
                        x.append(name)
                        fig.add_trace(
                            plot(
                                kind=kind,
                                name=name,
                                x= x,
                                y=to_plot[t],
                                marker_color=color,
                                visible=visible(step_index, aggregate),
                                showlegend=False if name in legends else True,
                            )
                        )

                        legends.add(name)

                
            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )
        
        if self.mode.name == "Planning":
            layout = {
                "title": "Total Capacity",
                "yaxis": {"title": unit},
                "xaxis": dict(tickmode="array", tickvals=values.index, dtick=1),
            }
        else:
            layout = {
                "title": "Total Capacity",
                "yaxis": {"title": unit},
                "xaxis": dict(tickmode="array", dtick=1),
            }
        if not aggregate:
            fig = set_steps(fig, counter, mode)
        if kind == "bar":
            layout["barmode"] = "relative"
        _plotter(fig=fig, layout=layout, path=path)
        # fig.show()

    def plot_prod_by_tech(
        self,
        tech_group,
        path,
        regions="all",
        kind="bar",
        mode="updatemenus",
        aggregate=False,
    ):
        """Plots the total yearly production of techs

        Parameters
        ----------
        tech_group : str
            The tech_group to plot

        regions : list
            The  region/regions to plot

        kind : str
            type of the plot: [bar,area]

        path : str
            Defines the path to save the file with the extension of the file
        """

        if regions == "all":
            regions = self.regions

        techs = self.configs["techs"]
        techs = techs[techs["tech_group"].isin(str2ls(tech_group))].index

        if not len(techs):
            raise ValueError(f"No tech found for category {tech_group}.")
        unit = self._take_units("techs", techs, "tech_production_unit")
        tech_type = self.glob_mapping["techs"].loc[techs]["Tech_category"]

        fig = go.Figure()
        counter = []

        if aggregate:
            legends = set()
        for step_index, region in enumerate(regions):
            if not aggregate:
                legends = set()
            for tt in tech_type.unique():
                if tt not in self.data.technology_prod[region]:
                    continue
                techprod = self.data.technology_prod[region][tt]                
                columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==tt]["Technology"].values
                
                indexer_time = pd.MultiIndex.from_product(
                    [self.years, self.time_steps],
                    names=["Years", "Timesteps"],
                )
 
                techprod = pd.DataFrame(techprod.value, indexer_time, columns)
                techprod = techprod.groupby(level=0, sort=False).sum()
                
                for t, values in techprod.items():
                    if t not in techs:
                        continue
                    name = self.configs["techs"].loc[t, "tech_name"]
                    color = self.configs["techs"].loc[t, "tech_color"]

                    fig.add_trace(
                        plot(
                            kind=kind,
                            name=name,
                            x=values.index,
                            y=techprod[t],
                            marker_color=color,
                            visible=visible(step_index, aggregate),
                            showlegend=False if t in legends else True,
                        )
                    )

                    legends.add(t)
            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )
        
        layout = {
            "title": "Production by technology",
            "yaxis": {"title": unit},
            "xaxis": dict(tickmode="array", tickvals=values.index, dtick=1),
        }
        if not aggregate:
            fig = set_steps(fig, counter, mode)
        if kind == "bar":
            layout["barmode"] = "relative"
        _plotter(fig=fig, layout=layout, path=path)
        # fig.show()

    def plot_fuel_prod_cons(
        self,
        path,
        years,
        fuel_group,
        regions="all",
        trade=True,
        mode="updatemenus",
        aggregate=False,
    ):
        """Plots the share of a fuel group production and consumption

        .. note::

           data will be aggregated over the given years

        Parameters
        ----------
        years : list
            the years to be aggregaed

        regions : list
            regions to be plotted

        trade : boolean
            if True, imports and exports will be considered in production & consumption
        """
        
        indexer_time = pd.MultiIndex.from_product(
            [self.years, self.time_steps],
            names=["Years", "Timesteps"],
        )
        
        # aggregate = False
        fuels = self.configs["fuels"]
        fuels = fuels[fuels["fuel_group"].isin(str2ls(fuel_group))].index

        if regions == "all":
            regions = self.regions

        if not len(fuels):
            raise ValueError(f"No fuel found for category {fuel_group}.")
        _ = self._take_units("fuels", fuels, "fuel_unit")

        specs = [[{"type": "domain"}, {"type": "domain"}]]
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["Production", "Consumption"], specs=specs,
        )

        counter = []

        if aggregate:
            legends = set()
        # Define the production
        for step_index, region in enumerate(regions):

            if not aggregate:
                legends = set()
            prod_techs = []

            for ff in fuels:
                if ff in self.sets[region]["Carrier_output"].index:
                    prod_techs.extend(
                        get_data(
                            self.sets[region]["Carrier_output"].loc[ff, "Technology"]
                        )
                    )
            # remove the transmission
            trans = get_data(
                self.sets[region]["Technologies"].loc["Transmission", "Technology"]
            )

            for tran in trans:
                if tran in prod_techs:
                    prod_techs.remove(tran)
            # Creating production
            labels = []
            colors = []
            production = pd.Series(dtype="float64")

            for category, df in self.data.technology_prod[region].items():
                if category in ["Transmission", "Storage"]:
                    continue
                
                sliced_techs = get_data(
                    self.sets[region]["Technologies"].loc[category, "Technology"]
                )
                df = pd.DataFrame(df.value, indexer_time, sliced_techs)
                techs_plt = [i for i in prod_techs if i in sliced_techs]
                techs_plt = list(set(techs_plt))

                if category == "Conversion_plus":
                    _df = pd.Series(dtype="float64")
                    for tt, values in df.loc[years, techs_plt].items():
                        activity = values

                        ratio = self.data.carrier_ratio_out[region].loc[
                            (years, slice(None)), (tt, slice(None))
                        ]

                        res = 0
                        for cols, ratios in ratio.items():

                            if cols[1] in fuels:
                                res += (activity * ratios).sum().sum()
                        _df.loc[tt] = res

                else:
                    _df = df.filter(items=techs_plt,axis=1).groupby(level=0, sort=False).sum().loc[years, techs_plt].sum()

                production = production.append(_df)

                labels.extend(
                    [self.configs["techs"].loc[tt, "tech_name"] for tt in _df.index]
                )
                colors.extend(
                    [self.configs["techs"].loc[tt, "tech_color"] for tt in _df.index]
                )
                
            if trade and not aggregate:
                for regions in self.configs["regions"].index:
                    if(regions == region):
                        continue
                    
                    columns = self.glob_mapping["fuels"]["Carr_name"].index
                    imports = self.data.line_import[region][regions]
                    imports = pd.DataFrame(imports.value, index = indexer_time, columns = columns)

                    imports = pd.DataFrame(imports[fuels].values, 
                                            index = indexer_time, 
                                            columns = ["Import to " + region + " from " + regions]*len(fuels))
                    
                    imports = imports.loc[years].sum(axis=1).sum()
                    production.loc[regions] = imports

                    labels.append(
                        "Import from "
                        + self.configs["regions"].loc[regions, "region_name"]
                    )
                    colors.append(self.configs["regions"].loc[regions, "region_color"])
      
            fig.add_trace(
                go.Pie(
                    values=production.values,
                    labels=labels,
                    visible=visible(step_index, aggregate),
                    marker=dict(colors=colors),
                    sort=False,
                ),
                1,
                1,
            )

            consumption = pd.Series(dtype="float64")
            labels = []
            colors = []
            # Identify what techs are consuming that fuel
            carrier_tech = self.mapping[region]["Carrier_input"]
            carrier_tech = carrier_tech[carrier_tech.Carrier_in.isin(fuels)][
                "Technology"
            ].tolist()
            # define if demand and intermediate exist
            tech_type = self.glob_mapping["fuels"].loc[fuels, "Carr_type"].tolist()

            if "Demand" in tech_type:
                for tt, values in (
                    self.data.demand[region]
                    .groupby(level=0, sort=False)
                    .sum()
                    .items()
                ):
                    if tt not in carrier_tech:
                        continue
                    labels.append(self.configs["techs"].loc[tt, "tech_name"])
                    colors.append(self.configs["techs"].loc[tt, "tech_color"])
                    consumption.loc[tt] = values.loc[years].sum()
                    
            if trade and not aggregate:
                for regions in self.configs["regions"].index:
                    if(regions == region):
                        continue
                    
                    columns = self.glob_mapping["fuels"]["Carr_name"].index
                    exports = self.data.line_export[region][regions]
                    exports = pd.DataFrame(exports.value, index = indexer_time, columns = columns)

                    exports = pd.DataFrame(exports[fuels].values,
                                           index = indexer_time, 
                                           columns = ["Export from " + region + " to " + regions]*len(fuels))
                    
                    exports = exports.loc[years].sum(axis=1).sum()
                    consumption.loc[regions] = exports

                    labels.append(
                        "Export to " + self.configs["regions"].loc[regions, "region_name"]
                    )
                    colors.append(self.configs["regions"].loc[regions, "region_color"])
            
            for category, df in self.data.technology_use[region].items():
                if category in ["Transmission", "Storage"]:
                    continue
                
                columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==category]["Technology"].values
                df = pd.DataFrame(df.value, indexer_time, columns)
                
                if category == "Conversion":
                    df = df.groupby(level=0, sort=False).sum().loc[years, :].sum()
                    for tt, values in df.items():
                        if tt not in carrier_tech:
                            continue
                        labels.append(self.configs["techs"].loc[tt, "tech_name"])
                        colors.append(self.configs["techs"].loc[tt, "tech_color"])
                        consumption.loc[tt] = values
                elif category == "Conversion_plus":

                    for tt, values in df.loc[years, :].items():

                        if tt not in carrier_tech:
                            continue
                        labels.append(self.configs["techs"].loc[tt, "tech_name"])
                        colors.append(self.configs["techs"].loc[tt, "tech_color"])

                        activity = values
                        ratio = self.data.carrier_ratio_in[region].loc[
                            (years, slice(None)), (tt, slice(None))
                        ]
           
                        res = 0
                        for cols, ratios in ratio.items():
                            if cols[1] in fuels:
                                res += (activity * ratios).sum()
                        consumption.loc[tt] = res
          
            fig.add_trace(
                go.Pie(
                    values=consumption.values,
                    labels=labels,
                    visible=visible(step_index, aggregate),
                    marker=dict(colors=colors),
                    sort=False,
                ),
                1,
                2,
            )

            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )

        if not aggregate:
            fig = set_steps(fig, counter, mode)
        fig.update_annotations(yshift=20, xshift=-200)
        layout = {}
        _plotter(fig=fig, layout=layout, path=path)
        # fig.show()

    def plot_emissions(
        self,
        path,
        tech_group,
        emission_type,
        regions="all",
        kind="bar",
        mode="updatemenus",
        aggregate=False,
    ):
        """ Plots the emissions for a tech_group

        Parameters
        -----------
        path : str
            Defines the path and the name of the file to save
        tech_group : str
            the techn_group to plot based on config excel file.

        regions : list[str]
            Which regions to plot
        kind : str, optional
            Type of the plot can be area or bar. The default is "bar".
        mode : str, optional
            defined the mode when multiple regions are plotted. Acceptable values are "updatemenus" and "sliders". The default is "updatemenus".
        aggregate : boolean, optional
            If True will aggregated the regions into one singel region. The default is False.
        """
        
        indexer_time = pd.MultiIndex.from_product(
            [self.years, self.time_steps],
            names=["Years", "Timesteps"],
        )
        
        techs = self.configs["techs"]
        techs = list(techs[techs["tech_group"].isin(str2ls(tech_group))].index)
        emission_name = self.configs["emissions"]
        emission_name = list(emission_name[emission_name["emission_name"].isin(str2ls(emission_type))].index)
        unit = list(self.configs["emissions"].loc[emission_name, "emission_unit"])

        if regions == "all":
            regions = self.regions

        if not len(techs):
            raise ValueError(f"No tech found for category {tech_group}.")
        fig = go.Figure()
        counter = []
        
        if aggregate:
            legends = set()
        for step_index, types in enumerate(emission_name):
            if not aggregate:
                legends = set()
            for step_index_reg, region in enumerate(regions):
                for category, df in self.data.emission_by_type[types][region].items():
                    columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==category]["Technology"].values
                    to_plot = pd.DataFrame(df.value, self.years, columns)
                    for t, values in to_plot.items():
                        if t not in techs:
                            continue
                        name = self.configs["techs"].loc[t, "tech_name"]
                        color = self.configs["techs"].loc[t, "tech_color"]
                        fig.add_trace(
                            plot(
                                kind=kind,
                                name=name,
                                x=values.index,
                                y=to_plot[t],
                                marker_color=color,
                                visible=visible(step_index, aggregate),
                                showlegend=True,
                            )
                        )
                        
                        legends.add(t)

                counter.append(
                    (self.configs["regions"].loc[region, "region_name"], len(fig.data))
                )
                
            layout = {
                "title": types + " emission",
                "xaxis": dict(tickmode="array", tickvals=values.index, dtick=1),
                "yaxis_title": unit[step_index],
            }
    
            if not aggregate:
                fig = set_steps(fig, counter, mode)
            if kind == "bar":
                layout["barmode"] = "relative"
            _plotter(fig=fig, layout=layout, path=path)
            # fig.show()

    def plot_hourly_prod_by_tech(
        self,
        path,
        tech_group,
        fuel_group,
        year,
        regions="all",
        kind="area",
        freq="h",
        function="sum",
        start="01-01 00:00:00",
        end="12-29 23:00:00",
        mode="updatemenus",
        aggregate=False,
    ):
        """Plots hourly tech production


        Parameters
        ----------
        path : str
            Defines the path and the name of the file to save
        tech_group : str
            the techn_group to plot based on config excel file.
        regions : list[str]
            Which regions to plot
        year : int,str
            hourly plot can be used for one year only
        kind : str, optional
            Type of the plot can be area or bar. The default is "area".
        freq : str, optional
            Defines the frequencey of data.Follows the pd.Datetime rules. The default is "h". for example for a 3 hours frequency '3h' should be passed.
        function : str, optional
            In case of resampling data, the function can be specified. The default is "sum".
        start : str, optional
            the starting datetime for the plot. The default is "01-01 00:00:00".
        end : str, optional
            the ending datetime for the plot. The default is "12-29 23:00:00".
        mode : str, optional
            defined the mode when multiple regions are plotted. Acceptable values are "updatemenus" and "sliders". The default is "updatemenus".
        aggregate : boolean, optional
            If True will aggregated the regions into one singel region. The default is False.

        """
        
        multiplier = len(self.time_steps)/8760
        
        indexer_time = pd.MultiIndex.from_product(
            [self.years, self.time_steps],
            names=["Years", "Timesteps"],
        )

        if not self._hourly:
            raise Exception("Is only applicapable with hourly resolution")

        techs = self.configs["techs"]
        techs = techs[techs["tech_group"].isin(str2ls(tech_group))].index
        if not len(techs):
            raise ValueError(f"No tech found for category {tech_group}.")

        unit = self._take_units("techs", techs, "tech_production_unit")

        if regions == "all":
            regions = self.regions

        tech_type = self.glob_mapping["techs"].loc[techs]["Tech_category"]

        fig = go.Figure()
        counter = []
        if aggregate:
            legends = set()
        if freq != "h":
            fig.update_layout({"xaxis": {"tickformat": "%b"}})
        for step_index, region in enumerate(regions):
            if not aggregate:
                legends = set()
            
            if self.settings.multi_node:
                to_plots_trade = []   
                for regions in self.configs["regions"].index:
                    if(regions == region):
                        continue
                    
                    fuel = self.configs["fuels"]
                    fuel = fuel[fuel["fuel_group"].isin(str2ls(fuel_group))].index
                    
                    columns = self.glob_mapping["fuels"]["Carr_name"].index
                    imports = self.data.line_import[region][regions]
                    imports = pd.DataFrame(imports.value, index = indexer_time, columns = columns)
    
                    imports = pd.DataFrame(imports[fuel].values, 
                                           index = indexer_time, 
                                           columns = ["Import to " + region + " from " + regions]*len(fuel))
    
                    to_plots_trade.append(
                        imports.loc[year]
                    )
                      
                for regions in self.configs["regions"].index:
                    if(regions == region):
                        continue
                    
                    fuel = self.configs["fuels"]
                    fuel = fuel[fuel["fuel_group"].isin(str2ls(fuel_group))].index
                    
                    columns = self.glob_mapping["fuels"]["Carr_name"].index
                    exports = self.data.line_export[region][regions]
                    exports = pd.DataFrame(exports.value, index = indexer_time, columns = columns)
    
                    exports = pd.DataFrame(exports[fuel].values,
                                           index = indexer_time, 
                                           columns = ["Export from " + region + " to " + regions]*len(fuel))
    
                    to_plots_trade.append(
                        -exports.loc[year]
                    )
                    
            to_plot_unmets = []
            fuel = self.configs["fuels"]
            fuel = fuel[fuel["fuel_group"].isin(str2ls(fuel_group))].index
            
            for carr in fuel:
                if carr in self.glob_mapping["fuels"].loc[self.glob_mapping["fuels"]["Carr_type"]=="Demand"].index:
                    
                    unmet_demand = self.data.unmetdemandbycarrier[region][carr]
                    unmet_demand = pd.DataFrame(unmet_demand.value, index = indexer_time, columns = [carr])
        
                    unmet_demand = pd.DataFrame(unmet_demand[carr].values, 
                                           index = indexer_time, 
                                           columns = ["Unmet Demand " +str(carr)])
        
                    to_plot_unmets.append(
                        unmet_demand.loc[year]
                    )
            
            to_plots = []
            # add if there are some storages consumption
            for tt in tech_type.unique():

                if tt not in self.data.technology_prod[region]:
                    continue
                
                techprod = self.data.technology_prod[region][tt]                
                columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==tt]["Technology"].values

                techprod = pd.DataFrame(techprod.value, indexer_time, columns)   
                to_plots.append(
                    techprod.loc[year]
                )

                if tt == "Storage":
                    
                    storage = self.data.technology_use[region][tt]                
                    columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==tt]["Technology"].values
                    
                    indexer_time = pd.MultiIndex.from_product(
                        [self.years, self.time_steps],
                        names=["Years", "Timesteps"],
                    )
     
                    storage = pd.DataFrame(storage.value, indexer_time, columns)
                    to_plots.append(
                        -storage.loc[year]
                    )

            for to_plot in to_plots:
                
                _year_ = self.glob_mapping["years"].loc[year]["Year_name"].to_list()
                # To avoid leap years,
                time_index = pd.date_range(
                    start= str(_year_[0])+"-01-01 00:00:00", periods=len(to_plot), freq="1h"
                )
                to_plot.index = time_index

                try:
                    to_plot = to_plot.loc[start:end]
                except:
                    raise ValueError("incorrect start or end.")
                # print(to_plot)
                    
                # try:
                #     yy = int(year) - 2021
                #     to_plot.index = to_plot.index + pd.offsets.DateOffset(years=yy)
                # except:
                #     pass

                to_plot = to_plot.resample(freq)
                to_plot = eval(f"to_plot.{function}()")
                

                for t, values in to_plot.items():
                    if t not in techs:
                        continue
                    name = self.configs["techs"].loc[t, "tech_name"]
                    color = self.configs["techs"].loc[t, "tech_color"]

                    fig.add_trace(
                        plot(
                            kind=kind,
                            name=name,
                            x=values.index,
                            y=values.values*multiplier,
                            marker_color=color,
                            visible=visible(step_index, aggregate),
                            showlegend=False if t in legends else True,
                        )
                    )

                    legends.add(t)
                    
            if not aggregate:
                if self.settings.multi_node:                
                    for to_plot_trade in to_plots_trade:
        
                        _year_ = self.glob_mapping["years"].loc[year]["Year_name"].to_list()
                        # To avoid leap years,
                        time_index = pd.date_range(
                            start= str(_year_[0])+"-01-01 00:00:00", periods=len(to_plot_trade), freq="1h"
                        )
        
                        to_plot_trade.index = time_index
                        
                        try:
                            to_plot_trade = to_plot_trade.loc[start:end]
                        except:
                            raise ValueError("incorrect start or end.")
        
                        # try:
                        #     yy = int(year) - 2021
                        #     to_plot_trade.index = to_plot_trade.index + pd.offsets.DateOffset(years=yy)
                        # except:
                        #     pass
        
                        to_plot_trade = to_plot_trade.resample(freq)
                        to_plot_trade = eval(f"to_plot_trade.{function}()")
        
                        for tt, values in to_plot_trade.items():
        
                            name = self.configs["importexport"].loc[tt, "line_name"]
                            color = self.configs["importexport"].loc[tt, "line_color"]
        
                            fig.add_trace(
                                plot(
                                    kind=kind,
                                    name=name,
                                    x=values.index,
                                    y=values.values*multiplier,
                                    marker_color=color,
                                    visible=visible(step_index, aggregate),
                                    showlegend=False if tt in legends else True,
                                )
                            )
        
                            legends.add(tt)
                            
            for to_plot_unmet in to_plot_unmets:
                
                _year_ = self.glob_mapping["years"].loc[year]["Year_name"].to_list()
                # To avoid leap years,
                time_index = pd.date_range(
                    start= str(_year_[0])+"-01-01 00:00:00", periods=len(to_plot_unmet), freq="1h"
                )
                to_plot_unmet.index = time_index

                try:
                    to_plot_unmet = to_plot_unmet.loc[start:end]
                except:
                    raise ValueError("incorrect start or end.")
                # print(to_plot)
                    
                # try:
                #     yy = int(year) - 2021
                #     to_plot.index = to_plot.index + pd.offsets.DateOffset(years=yy)
                # except:
                #     pass

                to_plot_unmet = to_plot_unmet.resample(freq)
                to_plot_unmet = eval(f"to_plot_unmet.{function}()")
            
                for tt, values in to_plot_unmet.items():
                    
                    fig.add_trace(
                        plot(
                            kind=kind,
                            name="Unmet Demand",
                            x=values.index,
                            y=values.values*multiplier,
                            marker_color="#0c0c0d",
                            visible=visible(step_index, aggregate),
                            showlegend=False if tt in legends else True,
                        )
                    )

                    legends.add("Unmet Demand")
                    
                
            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )
        
        layout = {
            "title": "Hourly production by technologies",
            "yaxis": {"title": unit},
            "xaxis": dict(tickmode="array", tickvals=values.index, dtick=1),
        }
        if not aggregate:
            fig = set_steps(fig, counter, mode)
        if kind == "bar":
            layout["barmode"] = "relative"
        _plotter(fig=fig, layout=layout, path=path)
        # fig.show()

    def plot_regional_costs(
        self,
        path,
        regions="all",
        stacked_by="techs",
        aggregate=False,
        mode="updatemenus",
        exclude_tech_groups=[],
        exclude_cost_items=[],
    ):
        """


        Parameters
        ----------
        path : str
            The path to save the plot.
        regions :  list[str], optional
            regions to plot,. The default is "all".
        stacked_by : str, optional
            defines the stacking opttion. if techs, will stack techs. if items will stack to cost items. The default is "techs".
        aggregate : boolean, optional
            if True will aggregate the regions. The default is False.
        mode : str, optional
            defines the mode if multiple regions exists. The default is "updatemenus".
        exclude_tech_groups : list[str], optional
            the tech groupes to not plot. The default is [].
        exclude_cost_items : list[str], optional
            the cost items not to plot. The default is [].
        """

        cost_items = {
            "cost_fix": {"name": "Fix Cost", "color": "red"},
            "cost_variable": {"name": "Variable Cost", "color": "blue"},
            "cost_inv": {"name": "Investment Cost", "color": "green"},
            "decommissioning_cost": {"name": "Decommissioning Cost", "color": "yellow"},
            "cost_fix_tax": {"name": "Fix Tax Cost", "color": "brown"},
            "cost_fix_sub": {"name": "Fix subsidies", "color": "khaki"},
            "cost_inv_tax": {"name":"Investment Tax Cost", "color":"orange"},
            "cost_inv_sub": {"name": "Investment subsidies", "color":"grey"},
            "emission_cost_by_region": {"name": "Emission tax", "color":"black"},            
        }
        
        # cost_items_line = {
        #     "cost_inv_line":{"name":"Investment Cost Lines", "color":"purple"},
        #     "cost_variable_line":{"name":"Variable Cost Lines","color":"pink"},
        # }
            

        techs = self.configs["techs"]
        exclude_techs = techs[
            techs["tech_group"].isin(str2ls(exclude_tech_groups))
        ].index

        if regions == "all":
            regions = self.regions
            
        emission_name = self.configs["emissions"]
        emission_name = list(emission_name.index)

        fig = go.Figure()
        counter = []
        if aggregate:
            legends = set()

        for step_index, region in enumerate(regions):
            if not aggregate:
                legends = set()
                
            data=self.data._asdict()
            
            for cost_item, info in cost_items.items():
                if (cost_item not in data.keys()) or (cost_item in exclude_cost_items):
                    continue                    
                
                tech_type_ = self.mapping[region]["Technologies"]["Tech_category"].to_list()

                tech_type = []
                for elem in tech_type_:
                    if elem not in tech_type:
                        tech_type.append(elem)
                        
                to_plot = pd.DataFrame()
                
                if cost_item in ["emission_cost_by_region"]:
                    for type_emissions in emission_name:
                        for tt in tech_type:
                            if tt == "Demand" or tt =="Transmission" or tt == "Storage":
                                continue
                            columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==tt]["Technology"].to_list()
                            df = pd.DataFrame(data[cost_item][region][type_emissions][tt].value,index = self.years, columns = columns)
                            to_plot = pd.concat([to_plot,df], axis=1)
                else:            
                    for tt in tech_type:
                        if tt == "Demand":
                            continue
                        columns = self.mapping[region]["Technologies"].loc[self.mapping[region]["Technologies"]["Tech_category"]==tt]["Technology"].to_list()
                        df = pd.DataFrame(data[cost_item][region][tt].value,index = self.years, columns = columns)
                        to_plot = pd.concat([to_plot,df], axis=1)
                    
                if cost_item in ["cost_fix_sub", "cost_inv_sub"]:
                    to_plot = -to_plot

                if stacked_by == "techs":

                    for t, values in to_plot.items():
                        if t in exclude_techs:
                            continue

                        name = self.configs["techs"].loc[t, "tech_name"]
                        color = self.configs["techs"].loc[t, "tech_color"]

                        fig.add_trace(
                            plot(
                                kind="bar",
                                name=name,
                                x=[values.index, [info["name"]] * len(values)],
                                y=values.values,
                                marker_color=color,
                                legendgroup=name,
                                visible=visible(step_index, aggregate),
                                showlegend=False if t in legends else True,
                            )
                        )

                        legends.add(t)

                elif stacked_by == "items":
                    for t, values in to_plot.items():
                        if t in exclude_techs:
                            continue
                        name = info["name"]
                        color = info["color"]
                        tech_name = self.configs["techs"].loc[t, "tech_name"]

                        fig.add_trace(
                            plot(
                                kind="bar",
                                name=name,
                                x=[values.index, [tech_name] * len(values)],
                                y=values.values,
                                marker_color=color,
                                legendgroup=name,
                                visible=visible(step_index, aggregate),
                                showlegend=False if cost_item in legends else True,
                            )
                        )
                        legends.add(cost_item)

                else:
                    raise ValueError(
                        "acceptable stacked_by inputs are ['techs','items']"
                    )

            counter.append(
                (self.configs["regions"].loc[region, "region_name"], len(fig.data))
            )

        layout = {
            "title": "Regional costs",
            "barmode": "relative"}
        if not aggregate:
            fig = set_steps(fig, counter, mode)
        _plotter(fig=fig, layout=layout, path=path)
        # fig.show()
