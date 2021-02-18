import numpy as np

from .numerical_column_stats import NumericStatsMixin
from .base_column_profilers import BaseColumnProfiler, \
    BaseColumnPrimitiveTypeProfiler
from .profiler_options import FloatOptions


class FloatColumn(NumericStatsMixin, BaseColumnPrimitiveTypeProfiler):
    """
    Float column profile mixin with of numerical stats. Represents a column in
    the dataset which is a float column.
    """

    col_type = "float"

    def __init__(self, name, options=None):
        """
        Initialization of column base properties and itself.
        :param name: Name of the data
        :type name: String
        :param options: Options for the float column
        :type options: FloatOptions
        """
        if options:
            if not isinstance(options, FloatOptions):
                raise ValueError("options must be of type FloatOptions.")
        NumericStatsMixin.__init__(self, options)
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)
        self.precision = 0
        self.__calculations = {
            "precision": FloatColumn._update_precision,
        }
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other):
        """
        Merges the properties of two FloatColumn profiles
        :param self: first profile
        :param other: second profile
        :type self: FloatColumn
        :type other: FloatColumn
        :return: New FloatColumn merged profile
        """
        if not isinstance(other, FloatColumn):
            raise TypeError("Unsupported operand type(s) for +: "
                            "'FloatColumn' and '{}'"
                            .format(other.__class__.__name__))

        merged_profile = FloatColumn(None)
        BaseColumnPrimitiveTypeProfiler._add_helper(merged_profile, self, other)
        NumericStatsMixin._add_helper(merged_profile, self, other)
        return merged_profile

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.
        :return:
        """
        histogram_method = self.histogram_bin_method_names[0]
        if self.histogram_selection is not None:
            histogram_method = self.histogram_selection

        profile = dict(
            min=self.min,
            max=self.max,
            mean=self.mean,
            median=None,
            variance=self.variance,
            stddev=self.stddev,
            histogram=self.histogram_methods[histogram_method]['histogram'],
            quantiles=self.quantiles,
            times=self.times,
            precision=self.precision
        )
        return profile

    @property
    def data_type_ratio(self):
        """
        Calculates the ratio of samples which match this data type.
        :return: ratio of data type
        :rtype: float
        """
        if self.sample_size:
            return float(self.match_count) / self.sample_size
        return None

    @classmethod
    def _get_float_precision(cls, df_series):
        """
        Determines the precision of the numeric value
        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :return: string representing its precision print format
        :rtype: int
        """
        integer_decimal_loc = -1
        float_precision = 0
        for value in df_series:
            decimal_loc = value.rfind('.')

            # integers will not have a '.'
            if decimal_loc == integer_decimal_loc:
                continue

            # since has a '.', subtract the str len from the position,
            # since indexes start at 0: len - pos - 1
            value_len = len(value)
            value_precision = value_len - decimal_loc - 1

            if float_precision < value_precision:
                float_precision = value_precision
        return float_precision

    @classmethod
    def _is_each_row_float(cls, df_series):
        """
        Determines if each value in a dataframe is a float. Integers and NaNs
        can be considered a float.
        e.g.
        For column [1, 1, 1] returns [True, True, True]
        For column [1.0, np.NaN, 1.0] returns [True, True, True]
        For column [1.0, "a", "b"] returns [True, False, False]
        :param df_series: series of values to evaluate
        :type df_series: pandas.core.series.Series
        :return: is_float_col
        :rtype: list
        """
        len_df = len(df_series)
        if len_df == 0:
            return list()

        return [NumericStatsMixin.is_float(x) for x in df_series]

    @BaseColumnProfiler._timeit(name='precision')
    def _update_precision(self, df_series, prev_dependent_properties,
                          subset_properties):
        """
        Updates the precision value of the column.

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
        self.precision = max(
            self.precision, self._get_float_precision(df_series)
        )

    def _update_helper(self, df_series_clean, profile):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known profile of the dataset.
        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: float profile dictionary
        :type profile: dict
        :return: None
        """
        if self._NumericStatsMixin__calculations:
            NumericStatsMixin._update_helper(self, df_series_clean, profile)
        self._update_column_base_properties(profile)

    def update(self, df_series):
        """
        Updates the column profile.
        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: None
        """
        if len(df_series) == 0:
            return
        df_series = df_series.reset_index(drop=True)
        is_each_row_float = self._is_each_row_float(df_series)
        sample_size = len(is_each_row_float)
        float_count = np.sum(is_each_row_float)
        profile = dict(match_count=float_count, sample_size=sample_size)

        BaseColumnProfiler._perform_property_calcs(
            self, self.__calculations, df_series=df_series[is_each_row_float],
            prev_dependent_properties={}, subset_properties=profile)

        self._update_helper(
            df_series_clean=df_series[is_each_row_float],
            profile=profile
        )

    def _update_numeric_stats(self, df_series, prev_dependent_properties,
                              subset_properties):
        """
        Calls the numeric stats update function. This is a wrapper to allow
        for modularity.
        :param prev_dependent_properties: Contains all the previous properties 
        that the calculations depend on.
        :type prev_dependent_properties: Dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: Dict
        :param df_series: Data to be profiled
        :type df_series: Pandas Dataframe
        :return: None 
        """
        super(FloatColumn, self)._update_helper(df_series, subset_properties)
