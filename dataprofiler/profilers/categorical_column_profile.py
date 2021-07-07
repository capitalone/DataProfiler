from collections import defaultdict
from operator import itemgetter

from . import BaseColumnProfiler
from .profiler_options import CategoricalOptions
from . import utils


class CategoricalColumn(BaseColumnProfiler):
    """
    Categorical column profile subclass of BaseColumnProfiler. Represents a
    column int the dataset which is a categorical column.
    """

    type = "category"

    # If total number of unique values in a column is less than this value,
    # that column is classified as a categorical column.
    _MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL = 10

    # Default value that determines if a given col is categorical or not.
    _CATEGORICAL_THRESHOLD_DEFAULT = 0.2

    def __init__(self, name, options=None):
        """
        Initialization of column base properties and itself.

        :param name: Name of data
        :type name: String
        """
        if options and not isinstance(options, CategoricalOptions):
            raise ValueError("CategoricalColumn parameter 'options' must be of"
                             " type CategoricalOptions.")
        super(CategoricalColumn, self).__init__(name)
        self._categories = defaultdict(int)
        self.__calculations = {}
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other):
        """
        Merges the properties of two CategoricalColumn profiles

        :param self: first profile
        :param other: second profile
        :type self: CategoricalColumn
        :type other: CategoricalColumn
        :return: New CategoricalColumn merged profile
        """
        if not isinstance(other, CategoricalColumn):
            raise TypeError("Unsupported operand type(s) for +: "
                            "'CategoricalColumn' and '{}'".format(
                                other.__class__.__name__))

        merged_profile = CategoricalColumn(None)
        merged_profile._categories = \
            utils.add_nested_dictionaries(self._categories, other._categories)
        BaseColumnProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(merged_profile.__calculations,
                                 self.__calculations,
                                 other.__calculations)
        return merged_profile

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.
        For categorical_count, it will display the top k categories most
        frequently occurred in descending order.
        """
        top_k_categories = 5

        profile = dict(
            categorical=self.is_match,
            statistics=dict([
                ('unique_count', len(self.categories)),
                ('unique_ratio', self.unique_ratio),
            ]),
            times=self.times
        )
        if self.is_match:
            profile["statistics"].update(
                dict(categories=self.categories)
            )
            profile["statistics"]['gini_impurity'] = self.gini_impurity
            profile["statistics"]['categorical_count'] = dict(
                sorted(self._categories.items(), key=itemgetter(1),
                       reverse=True)[:top_k_categories])
        return profile

    @property
    def categories(self):
        """
        Property for categories.
        """
        return list(self._categories.keys())

    @property
    def unique_ratio(self):
        """
        Property for unique_ratio. Returns ratio of unique 
        categories to sample_size
        """
        unique_ratio = 1.0
        if self.sample_size:
            unique_ratio = len(self.categories) / self.sample_size
        return unique_ratio

    @property
    def is_match(self):
        """
        Property for is_match. Returns true if column is categorical.
        """
        is_match = False
        unique = len(self._categories)
        if unique <= self._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL:
            is_match = True
        elif self.sample_size \
                and self.unique_ratio <= self._CATEGORICAL_THRESHOLD_DEFAULT:
            is_match = True            
        return is_match

    @BaseColumnProfiler._timeit(name="categories")
    def _update_categories(self, df_series, prev_dependent_properties=None, 
                           subset_properties=None):
        """
        Check whether column corresponds to category type and adds category
        parameters if it is.

        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :param df_series: Data to be profiled
        :type df_series: pandas.DataFrame
        :return: None
        """
        category_count = df_series.value_counts(dropna=False).to_dict()
        self._categories = utils.add_nested_dictionaries(self._categories,
                                                         category_count)

    def _update_helper(self, df_series_clean, profile):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known profile of the dataset.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: categorical profile dictionary
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series):
        """
        Updates the column profile.

        :param df_series: Data to profile.
        :type df_series: pandas.core.series.Series
        :return: None
        """
        if len(df_series) == 0:
            return self
        
        profile = dict(
            sample_size=len(df_series)
        )
        CategoricalColumn._update_categories(self, df_series)
        BaseColumnProfiler._perform_property_calcs(
            self, self.__calculations, df_series=df_series,
            prev_dependent_properties={}, subset_properties=profile)

        self._update_helper(df_series, profile)

        return self

    @property
    def gini_impurity(self):
        """
        Property for gini impurity
        :return: None or gini impurity probability
        """
        if not self._categories:
            return None
        summation = 0
        total = sum(self._categories.values())
        for i in self._categories:
            summation += (self._categories[i]/total) * \
                         (1 - (self._categories[i]/total))
        return summation
