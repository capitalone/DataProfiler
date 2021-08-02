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
    :type profiler: StructuredProfiler
    :param columns: list of column names to be plotted
    :type columns: list
    :return: returns fig
    :rtype: fig
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
    :param ax: ax as in seaborn ax
    :type ax: list
    :param title: name of a individual histogram
    :type title: str
    :return: ax
    """

    histogram = data_type_profiler._get_best_histogram_for_profile()
    if histogram['bin_counts'] is None or histogram['bin_edges'] is None:
        raise ValueError("The column profiler, " + str(
            data_type_profiler.name) + ", provided had no data and "
                                       "therefore could not be plotted.")
    ax = sns.histplot(x=histogram['bin_edges'][:-1], bins=histogram['bin_edges'],
                 weights=histogram['bin_counts'], ax=ax)
    if title is None:
        title = str(data_type_profiler.name)
    ax.set_title(title)
    return ax

