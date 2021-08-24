#!/usr/bin/env python3
import math
import warnings

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..profilers.profile_builder import StructuredProfiler, \
    StructuredColProfiler


def plot_histograms(profiler, columns=None):
    """
    Take a input of StructuredProfiler class and a list of specified column
    names and then plots the histograms of those that are int or float
    columns.

    :param profiler: StructuredProfiler variable
    :type profiler: StructuredProfiler
    :param columns: list of column names to be plotted
    :type columns: list
    :return: matplotlib figure of where the graph was plotted
    :rtype: matplotlib.figure.Figure
    """

    # get all inds to graph, raise error if user specified doesn't exist (part 1)
    inds_to_graph = []
    if not columns:
        inds_to_graph = list(range(len(profiler.profile)))
    else:
        for column in columns:
            col = column
            if isinstance(col, str):
                col = col.lower()
            if col not in profiler._col_name_to_idx:
                raise ValueError("Column \"" + str(col) + "\" is not found as a "
                                                       "profiler column")
            inds_to_graph.extend(profiler._col_name_to_idx[col])
        sorted(inds_to_graph)
    # get all columns which are either int or float (part2)
    def is_index_graphable_column(ind_to_graph):
        """
        This function filters ind_to_graph by returning false if there is a
        data type that is not a int or float, otherwise true
        """
        col_profiler = profiler.profile[ind_to_graph]
        data_compiler = col_profiler.profiles['data_type_profile']
        if data_compiler.selected_data_type not in ['int', 'float']:
            return False
        return True
    inds_to_graph = list(filter(is_index_graphable_column, inds_to_graph))

    if not inds_to_graph:
        warnings.warn("No plots were constructed"
                      " because no int or float columns were found in columns")
        return
    # get proper tile format for graph
    n = len(inds_to_graph)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols)  # this will need to be flattened into a list
    # flatten axes for inputing graphs into the plot
    if not isinstance(axs, (np.ndarray)):
        axs = np.array([axs])
    axs.flatten()

    # graph the plots (part 3)
    for col_ind, ax in zip(inds_to_graph, axs):
        col_profiler = profiler.profile[col_ind]
        data_compiler = col_profiler.profiles['data_type_profile']
        data_type = data_compiler.selected_data_type
        data_type_profiler = data_compiler._profiles[data_type]
        plot_col_histogram(data_type_profiler, ax=ax,
                           title=str(data_type_profiler.name))
    plt.show()
    return fig


def plot_col_histogram(data_type_profiler, ax=None, title=None):
    """
    Take a input of a Int or Float Column and plots the histogram

    :param data_type_profiler: the Int or Float column we pass in
    :type data_type_profiler: Union[IntColumn, FloatColumn]
    :param ax: matplotlib axes of where to plot the graph
    :type ax: matplotlib.axes.Axes
    :param title: title ot set for the graph
    :type title: str
    :return: matplotlib axes of where the graph was plotted
    """

    histogram = data_type_profiler._get_best_histogram_for_profile()
    if histogram['bin_counts'] is None or histogram['bin_edges'] is None:
        raise ValueError("The column profiler, " + str(
            data_type_profiler.name) + ", provided had no data and "
                                       "therefore could not be plotted.")
    ax = sns.histplot(
        x=histogram['bin_edges'][:-1], bins=histogram['bin_edges'],
        weights=histogram['bin_counts'], ax=ax)
    if title is None:
        title = str(data_type_profiler.name)
    ax.set_title(title)
    return ax


def plot_missing_values_matrix(profiler, ax=None, title=None):
    """
    Generates a matrix of bar graphs for the missing value locations within
    each column in a structured dataset. A color line indicates the value does
    not exist.

    :param profiler: structured profiler to be plotted
    :type profiler: StructuredProfiler
    :param ax: matplotlib axes where to plot the graph
    :type ax: matplotlib.axes.Axes
    :param title: title ot set for the graph
    :type title: str
    :return: matplotlib figure of where the graph was plotted
    """
    if not isinstance(profiler, StructuredProfiler):
        raise ValueError('`profiler` must of type StructuredProfiler.')
    return plot_col_missing_values(profiler.profile, ax=ax, title=title)


def plot_col_missing_values(col_profiler_list, ax=None, title=None):
    """
    Generates a bar graph of the missing value locations within a column where
    a color line indicates the value does not exist.

    :param col_profiler_list:
    :type col_profiler_list: list[StructuredColProfiler]
    :param ax: matplotlib axes where to plot the graph
    :type ax: matplotlib.axes.Axes
    :param title: title ot set for the graph
    :type title: str
    :return: matplotlib figure of where the graph was plotted
    """
    if not (isinstance(col_profiler_list, list)
            and all(isinstance(col_profile, StructuredColProfiler)
                    for col_profile in col_profiler_list)):
        raise ValueError('`col_profiler_list` must be a list of '
                         'StructuredColProfilers')

    # bar width settings and height settings for each null value
    # width = 1, height = 1 would be no gaps
    width = 0.8
    height = 1

    # determine the colors for the plot, first color reserved for background of
    # each column, subsequent colors are for null values.
    nan_to_color = dict()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]

    # setup plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # in case user passed their own axes
    fig = ax.figure

    # loop through eac column plotting their null values
    for col_id, col_profiler in enumerate(col_profiler_list):

        # plot the values background bar
        sample_size = col_profiler.sample_size
        ax.add_patch(matplotlib.patches.Rectangle(
            xy=(col_id - width / 2 + 0.5, - height / 2),
            width=width, height=sample_size + height,
            linewidth=1, color='blue', fill=True))

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
                    color_cycle = (
                        plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
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
                    ax.add_patch(matplotlib.patches.Rectangle(
                        xy=(col_id - width / 2 + 0.5, y_start - height / 2),
                        width=width, height=y_end - y_start + 1,
                        linewidth=1, color=nan_color, fill=True,
                        label='"' + null_type + '"'))
                    y_start = next_value
                null_ind += 1
            y_end = null_indexes[-1]
            # plot the patch of the last null value
            ax.add_patch(matplotlib.patches.Rectangle(
                xy=(col_id - width / 2 + 0.5, y_start - height / 2),
                width=width, height=y_end - y_start + 1,
                linewidth=1, color=nan_color, fill=True,
                label='"' + null_type + '"'))

    # setup the graph axes and labels
    column_names = ['"' + str(col.name) + '"' for col in col_profiler_list]
    xticks = [0.5 + i for i in range(col_id + 1)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(column_names, rotation=90, ha='right')
    ax.set_xbound(0, col_id + 1)
    ax.autoscale(enable=True)

    # limit legend to singles and not duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    return fig
