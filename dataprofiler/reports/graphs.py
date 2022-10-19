"""Contains functions for generating graph data report."""
# !/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, List, Optional, Union, cast

if TYPE_CHECKING:
    from dataprofiler.profilers.float_column_profile import FloatColumn
    from dataprofiler.profilers.int_column_profile import IntColumn

import numpy as np

try:
    import matplotlib
    import matplotlib.patches
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    # don't require if using graphs will below recommend to install if not
    # installed
    pass

from dataprofiler.profilers.profile_builder import (
    StructuredColProfiler,
    StructuredProfiler,
)

from . import utils


@utils.require_module(["matplotlib", "seaborn"])
def plot_histograms(
    profiler: StructuredProfiler,
    column_names: Optional[List[Union[int, str]]] = None,
    column_inds: Optional[List[int]] = None,
) -> matplotlib.pyplot.figure:
    """
    Plot the histograms of column names that are int or float columns.

    :param profiler: StructuredProfiler variable
    :type profiler: StructuredProfiler
    :param column_names: List of column names to be plotted. Can only specify
        columns or column_inds, but not both
    :type column_names: list[Union[int,str]]
    :param column_inds: List of column indexes to be plotted
    :type column_inds: list[int]
    :return: matplotlib figure of where the graph was plotted
    :rtype: matplotlib.pyplot.Figure
    """
    if column_names and column_inds:
        raise ValueError(
            "Can only specify either `column_names` or `column_inds` but not "
            "both simultaneously"
        )
    elif column_names is not None and not (
        isinstance(column_names, list) and all(isinstance(x, str) for x in column_names)
    ):
        raise ValueError(
            "`column_names` must be a list integers or strings "
            "matching the names of columns in the profiler."
        )
    elif column_inds is not None and not (
        isinstance(column_inds, list) and all(isinstance(x, int) for x in column_inds)
    ):
        raise ValueError(
            "`column_inds` must be a list of integers matching "
            "column indexes in the profiler"
        )

    # get all inds to graph, raise error if user specified doesn't exist
    inds_to_graph = column_inds if column_inds else []
    if not column_names and not column_inds:
        inds_to_graph = list(range(len(profiler.profile)))
    elif not column_inds:
        for column in cast(List[Union[str, int]], column_names):
            col = column
            if isinstance(col, str):
                col = col.lower()
            if col not in profiler._col_name_to_idx:
                raise ValueError(
                    'Column "' + str(col) + '" is not found as a ' "profiler column"
                )
            inds_to_graph.extend(profiler._col_name_to_idx[col])

    # sort the column indexes to be in the same order as the original profiler.
    sorted(inds_to_graph)

    # get all columns which are of type [int, float]
    def is_index_graphable_column(ind_to_graph: int) -> bool:
        """
        Filter ind_to_graph.

        Return false if there is a data type that is not a int or float,
        otherwise true.
        """
        col_profiler = profiler.profile[ind_to_graph]
        data_compiler = col_profiler.profiles["data_type_profile"]
        if data_compiler.selected_data_type not in ["int", "float"]:
            return False
        return True

    inds_to_graph = list(filter(is_index_graphable_column, inds_to_graph))

    if not inds_to_graph:
        warnings.warn(
            "No plots were constructed"
            " because no int or float columns were found in columns"
        )
        return

    # get proper tile format for graph
    n = len(inds_to_graph)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols)

    # flatten axes for inputting graphs into the plot
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()

    # graph the plots
    for col_ind, ax in zip(inds_to_graph, axs):
        col_profiler = profiler.profile[col_ind]
        data_compiler = col_profiler.profiles["data_type_profile"]
        data_type = data_compiler.selected_data_type
        data_type_profiler = data_compiler._profiles[data_type]
        ax = plot_col_histogram(
            data_type_profiler, ax=ax, title=str(data_type_profiler.name)
        )

        # remove x/ylabel on subplots
        ax.set(xlabel=None)
        ax.set(ylabel=None)

    # turn off all unused axes
    for ax in axs[n:]:
        ax.axis("off")

    # add figure x/ylabel and formatting
    fig.text(0.5, 0.01, "bins", ha="center", va="center")
    fig.text(0.01, 0.5, "Count", ha="center", va="center", rotation=90)
    fig.tight_layout()
    return fig


@utils.require_module(["matplotlib", "seaborn"])
def plot_col_histogram(
    data_type_profiler: Union[IntColumn, FloatColumn],
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """
    Take input of a Int or Float Column and plot the histogram.

    :param data_type_profiler: the Int or Float column we pass in
    :type data_type_profiler: Union[IntColumn, FloatColumn]
    :param ax: matplotlib axes of where to plot the graph
    :type ax: matplotlib.axes.Axes
    :param title: title ot set for the graph
    :type title: str
    :return: matplotlib axes of where the graph was plotted
    """
    histogram = data_type_profiler._get_best_histogram_for_profile()
    if histogram["bin_counts"] is None or histogram["bin_edges"] is None:
        raise ValueError(
            "The column profiler, "
            + str(data_type_profiler.name)
            + ", had no data and therefore could not be plotted."
        )
    ax = sns.histplot(
        x=histogram["bin_edges"][:-1],
        bins=histogram["bin_edges"].tolist(),
        weights=histogram["bin_counts"],
        ax=ax,
    )

    ax.set(xlabel="bins")
    if title is None:
        title = str(data_type_profiler.name)
    ax.set_title(title)
    return ax


@utils.require_module(["matplotlib", "seaborn"])
def plot_missing_values_matrix(
    profiler: StructuredProfiler,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
) -> matplotlib.pyplot.figure:
    """
    Generate matrix of bar graphs for missing value locations in cols of struct dataset.

    A colored line indicates the value does not exist.

    :param profiler: structured profiler to be plotted
    :type profiler: StructuredProfiler
    :param ax: matplotlib axes where to plot the graph
    :type ax: matplotlib.axes.Axes
    :param title: title ot set for the graph
    :type title: str
    :return: matplotlib figure of where the graph was plotted
    """
    if not isinstance(profiler, StructuredProfiler):
        raise ValueError("`profiler` must of type StructuredProfiler.")
    return plot_col_missing_values(profiler.profile, ax=ax, title=title)


@utils.require_module(["matplotlib", "seaborn"])
def plot_col_missing_values(
    col_profiler_list: List[StructuredColProfiler],
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
) -> matplotlib.pyplot.figure:
    """
    Generate bar graph of missing value locations within a col.

    A colored line indicates the value does not exist.

    :param col_profiler_list:
    :type col_profiler_list: list[StructuredColProfiler]
    :param ax: matplotlib axes where to plot the graph
    :type ax: matplotlib.axes.Axes
    :param title: title ot set for the graph
    :type title: str
    :return: matplotlib figure of where the graph was plotted
    """
    if not (
        isinstance(col_profiler_list, list)
        and all(
            isinstance(col_profile, StructuredColProfiler)
            for col_profile in col_profiler_list
        )
    ):
        raise ValueError(
            "`col_profiler_list` must be a list of " "StructuredColProfilers"
        )
    elif not col_profiler_list:
        warnings.warn(
            "There was no data in the profiles to plot missing " "column values."
        )
        return

    # bar width settings and height settings for each null value
    # width = 1, height = 1 would be no gaps
    width = 0.8
    height = 1

    # determine the colors for the plot, first color reserved for background of
    # each column, subsequent colors are for null values.
    nan_to_color = dict()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]

    # setup plot
    is_own_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        is_own_fig = True
    # in case user passed their own axes
    fig = ax.figure

    # loop through eac column plotting their null values
    for col_id, col_profiler in enumerate(col_profiler_list):

        # plot the values background bar
        sample_size = col_profiler.sample_size
        ax.add_patch(
            matplotlib.patches.Rectangle(
                xy=(col_id - width / 2 + 0.5, -height / 2),
                width=width,
                height=sample_size,
                linewidth=1,
                color="blue",
                fill=True,
            )
        )

        # get the list of null values in the column and plot contiguous nulls
        # as a single value
        null_data = col_profiler.null_types_index
        for i, null_type in enumerate(null_data):

            # sorted null indexes for plotting contiguous
            null_indexes = sorted(null_data[null_type])

            # get the color for this nan value, if it hasn't been determined,
            # determine it now
            if null_type not in nan_to_color:
                nan_to_color[null_type] = color_cycle.pop(0)
                # restart cycle if empty
                if not len(color_cycle):
                    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
            nan_color = nan_to_color[null_type]

            # loop through contiguous patches to plot a single bar
            null_ind = 0
            num_nulls = len(null_indexes)
            y_start = null_indexes[null_ind]
            y_end = y_start
            while null_ind < num_nulls - 1:
                value = null_indexes[null_ind]
                next_value = null_indexes[null_ind + 1]
                if next_value - value == 1:
                    y_end = next_value
                else:
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            xy=(col_id - width / 2 + 0.5, y_start - height / 2),
                            width=width,
                            height=y_end - y_start + 1,
                            linewidth=1,
                            color=nan_color,
                            fill=True,
                            label='"' + null_type + '"',
                        )
                    )
                    y_start = next_value
                null_ind += 1
            y_end = null_indexes[-1]
            # plot the patch of the last null value
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    xy=(col_id - width / 2 + 0.5, y_start - height / 2),
                    width=width,
                    height=y_end - y_start + 1,
                    linewidth=1,
                    color=nan_color,
                    fill=True,
                    label='"' + null_type + '"',
                )
            )

    # setup the graph axes and labels
    column_names = ['"' + str(col.name) + '"' for col in col_profiler_list]
    xticks = [0.5 + i for i in range(col_id + 1)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(column_names, rotation=90, ha="right")
    ax.set_xbound(0, col_id + 1)
    ax.autoscale(enable=True)

    # limit legend to singles and not duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("column name")
    ax.set_ylabel("row index")
    if title:
        ax.set_title(title)

    if is_own_fig:
        fig.set_tight_layout(True)

    return fig
