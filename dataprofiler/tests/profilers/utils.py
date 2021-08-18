import os
import shutil
import random
import json
from unittest import mock

import numpy as np

import dataprofiler as dp


def set_seed(seed=None):
    """
    Sets the seed for all possible random state libraries
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    dp.set_seed(seed)


def delete_folder(path):
    """
    Deletes the folder, if exists
    :param path:
    :type path: str
    :return:
    """

    # Remove the generated model folder if exists
    if os.path.exists(path):
        shutil.rmtree(path)


def delete_file(path):
    """
    Deletes the file, if exists
    :param path:
    :type path: str    :return:
    """

    if os.path.exists(path):
        os.remove(path)


def list_files_in_path(path):
    """
    List all files in a directory
    :param path: the path to a directory
    :type path: str
    :return: a list of strings with file and subdirectory names
    """

    return os.listdir(path)


def find_col_index_with_name(name, trained_schema):
    """
    finds the index of the column with name 'name' and returns the index
    :param name: name of the column to look for
    :type name: str
    :param trained_schema:
    :type trained_schema: List(dict)
    :return: index of the element in trained_schema that has the given name
    :rtype: int
    """
    for i in range(len(trained_schema)):
        if trained_schema[i]['name'] == name:
            return i
    return None


def generate_random_date_sample(start_date,
                                end_date,
                                date_formats=["%Y-%m-%d %H:%M:%S"],
                                t=None):
    """
    Generate a synthetic date
    :param start_date: earliest date in date
    :type start_date: pandas.tslib.Timestamp
    :param end_date: latest date in date
    :type end_date: pandas.tslib.Timestamp
    :param date_formats: possible formats for date
    :type date_formats: list[str]
    :return: sythetic date
    :rtype: str
    """
    if not t:
        t = random.random()

    random_format = random.choice(date_formats)
    ptime = start_date + t * (end_date - start_date)

    return ptime.strftime(random_format)


def get_depth(od):
    """Function to determine the depth of a nested dictionary.

    Parameters:
        od (dict): dictionary or dictionary-like object

    Returns:
        int: max depth of dictionary
    """
    if isinstance(od, dict):
        return 1 + (max(map(get_depth, od.values())) if od else 0)
    return 0


def clean_report(report):
    """
    Clean report for comparison of profiles

    :param report: profile report
    :return:
    """
    global_stats = report["global_stats"]
    if "correlation_matrix" in global_stats and \
            report["global_stats"]["correlation_matrix"] is not None:
        report["global_stats"]["correlation_matrix"] = \
            report["global_stats"]["correlation_matrix"].tolist()

    if "chi2_matrix" in global_stats and \
            report["global_stats"]["chi2_matrix"] is not None:
        report["global_stats"]["chi2_matrix"] = \
            report["global_stats"]["chi2_matrix"].tolist()

    data_stats = report["data_stats"]
    for i in range(len(data_stats)):
        stats = data_stats[i]["statistics"]
        if "histogram" in stats:
            if "bin_counts" in stats["histogram"]:
                stats["histogram"]["bin_counts"] = \
                    stats["histogram"]["bin_counts"].tolist()
            if "bin_edges" in stats["histogram"]:
                stats["histogram"]["bin_edges"] = \
                    stats["histogram"]["bin_edges"].tolist()

    return report


def mock_timeit(*args, **kwargs):
    """
    Creates a mock for the time.time function that increments the time for
    every call.
    """
    def increment_counter():
        counter = 0
        while True:
            counter += 1
            yield counter
    counter = increment_counter()
    return mock.patch('time.time', side_effect=lambda: next(counter))
