#!/usr/bin/env python
"""
coding=utf-8

Profiles the data.
"""

from __future__ import print_function
from __future__ import division

import abc
import time
from collections import OrderedDict, defaultdict
from future.utils import with_metaclass
import functools

import numpy as np


class BaseColumnProfiler(with_metaclass(abc.ABCMeta, object)):
    """
    Abstract class for profiling a column of data.
    """
    col_type = None

    # This specifies the minimum percent of elements in a column to meet the
    # matching condition so that the column is classified as that type.
    _COLUMN_MATCH_THRESHOLD = 0.9

    _SAMPLING_RATIO = 0.20
    _MIN_SAMPLING_COUNT = 500

    def __init__(self, name):
        """
        Initialization of base class properties for the subclass.

        :param name: Name of the dataset
        :type name: String
        """
        self.name = name
        self.col_index = np.nan
        self.sample_size = 0
        self.metadata = dict()
        self.times = defaultdict(float)

    @staticmethod
    def _combine_unique_sets(a, b):  # TODO: Not needed for data labeling
        """
        Method to union two lists.

        :type a: list
        :type b: list
        :rtype: list
        """
        if not a and not b:
            return list()
        elif not a:
            return list(OrderedDict.fromkeys(b))
        elif not b:
            return list(OrderedDict.fromkeys(a))
        return list(OrderedDict.fromkeys(a + b))

    @staticmethod
    def _timeit(method=None, name=None):
        """
        Measure execution time of provided method
        Records time into times dictionary

        :param method: method to time
        :type method: Callable
        :param name: key argument for the times dictionary
        :type name: str
        """

        def decorator(method, name_dec=None):
            @functools.wraps(method)
            def wrapper(self, *args, **kw):
                # necessary bc can't reassign external name
                name_dec = name
                if not name_dec:
                    name_dec = method.__name__
                ts = time.time()
                result = method(self, *args, **kw)
                te = time.time()
                self.times[name_dec] += (te - ts)
                return result

            return wrapper

        if callable(method):
            return decorator(method, name_dec=name)
        return decorator

    @staticmethod
    def _filter_properties_w_options(calculations, options):
        """
        Cycles through the calculations and turns off the ones that are
        disabled.

        :param calculations: Contains all the column calculations.
        :type calculations: Dict
        :param options: Contains all the options.
        :type options: BaseColumnOptions
        """
        for property in list(calculations):
            if options and not options.is_prop_enabled(property):
                del calculations[property]

    def _perform_property_calcs(self, calculations, df_series,
                                prev_dependent_properties, subset_properties):
        """
        Cycles through the properties of the columns and calculate them.

        :param calculations: Contains all the column calculations.
        :type calculations: dict
        :param df_series: Data to be profiled
        :type df_series: pandas.Dataframe
        :param prev_dependent_properties: Contains all the previous properties 
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        for property in calculations:
            calculations[property](self,
                                   df_series,
                                   prev_dependent_properties,
                                   subset_properties)

    def _add_helper(self, other1, other2):
        """
        Merges the properties of two BaseColumnProfile objects

        :param other1: first BaseColumn profile
        :param other2: second BaseColumn profile
        :type other1: BaseColumnProfiler
        :type other2: BaseColumnProfiler
        """
        if np.isnan(other1.col_index) and np.isnan(other2.col_index):
            pass
        elif other1.col_index == other2.col_index:
            self.col_index = other1.col_index
        else:
            raise ValueError("Column indexes unmatched: {} != {}"
                             .format(other1.col_index, other2.col_index))
        if other1.name == other2.name:
            self.name = other1.name
        else:
            raise ValueError("Column names unmatched: {} != {}"
                             .format(other1.name, other2.name))

        self.times = defaultdict(
            float, {key: (other1.times.get(key, 0)
                          + other2.times.get(key, 0)
                          + self.times.get(key, 0))
                    for key in (set(other1.times) | set(other2.times)
                                | set(self.times))}
        )

        self.sample_size = other1.sample_size + other2.sample_size

    def _update_column_base_properties(self, profile):
        """
        Updates the base properties with the base schema.

        :param profile: profile dictionary of data type
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def __getitem__(self, item):
        """
        Override for the [] operator to allow access to class properties.
        NOTE: Will be removed when switched over, only used as a method to
        integrate with current setup.
        """
        if not hasattr(self, item):
            raise ValueError("The property '{} does not exist.".format(item))
        return getattr(self, item)

    @abc.abstractmethod
    def _update_helper(self, df_series_clean, profile):
        """
        Private abstract method for updating the profile.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, df_series):
        """
        Private abstract method for updating the profile.

        :param df_series: Data to profile.
        :type df_series: Pandas Dataframe
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def profile(self):
        """
        Property for profile. Returns the profile of the column.
        """
        raise NotImplementedError()


class BaseColumnPrimitiveTypeProfiler(with_metaclass(abc.ABCMeta,
                                                     BaseColumnProfiler)):
    """
    Abstract class for profiling the primative data type for a column of data.
    """

    def __init__(self, name):
        """
        Initialization of base class properties for the subclass.

        :param name: Name of the data
        :type name: String
        """
        BaseColumnProfiler.__init__(self, name)
        self.match_count = 0

    def _update_column_base_properties(self, profile):
        """
        Updates the base properties with the base schema.

        :param profile: profile containg base properties
        :type profile: base data profile dict
        :return: None
        """
        self.match_count += profile.pop("match_count")
        BaseColumnProfiler. \
            _update_column_base_properties(self, profile)

    def _add_helper(self, other1, other2):
        """
        Merges the properties of two objects inputted

        :param other1: first profile
        :param other2: second profile
        :type other1: BaseColumnPrimitiveTypeProfiler
        :type other2: BaseColumnPrimitiveTypeProfiler
        """
        BaseColumnProfiler._add_helper(self, other1, other2)
        self.match_count = other1.match_count + other2.match_count
