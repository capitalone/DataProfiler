#!/usr/bin/env python3
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_histograms(profiler, columns=None):
    """
        Take a input of StructuredProfiler class and a list of specified column
        names and then plots the histograms of those that are int or float
        columns.

        :param profiler: StructuredProfiler variable
        :param columns: list of column names to be plotted
        :type profiler: StructuredProfiler
        :type columns: list
        :return:
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
    for col_ind in reversed(inds_to_graph):
        # get data_type
        data_compiler = profiler.profile[col_ind].profiles['data_type_profile']
        data_type = data_compiler.selected_data_type
        data_type_profiler = data_compiler._profiles[data_type]
        if data_type not in ['int', 'float']:
            inds_to_graph.pop()

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


def plot_col_histogram(data_type_profiler, ax=None, title=""):
    histogram = data_type_profiler._get_best_histogram_for_profile()
    if not histogram['bin_counts'] or not histogram['bin_edges']:
        raise ValueError("The column profiler, " + str(
            data_type_profiler.name) + ", provided had no data and "
                                       "therefore could not be plotted.")
    plot = sns.histplot(x=histogram['bin_edges'][:-1], bins=histogram['bin_edges'],
                 weights=histogram['bin_counts'], ax=ax)
    plot.set_title(title)

