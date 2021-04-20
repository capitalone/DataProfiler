import re
import copy
import math
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
        if options and not isinstance(options, FloatOptions):
            raise ValueError("FloatColumn parameter 'options' must be of type"
                             " FloatOptions.")
        NumericStatsMixin.__init__(self, options)
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)

        self.precision = {
            'min': None,
            'max': None,
            'mean': None,
            'var': None,
            'std': None,
            'sum': None,
            'sample_size': None,
            'margin_of_error': None,
            'confidence_level': 0.999
        }
        
        # https://www.calculator.net/confidence-interval-calculator.html
        self.__z_value_precision = 3.291

        self.__precision_sample_ratio = None
        if options and options.precision and options.precision.is_enabled:
            self.__precision_sample_ratio = options.precision.sample_ratio
        
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
        
        self._merge_calculations(merged_profile.__calculations,
                                 self.__calculations,
                                 other.__calculations)

        if "precision" in merged_profile.__calculations:

            if self.precision['min'] is None:
                merged_profile.precision = copy.deepcopy(other.precision)
            elif other.precision['min'] is None:
                merged_profile.precision = copy.deepcopy(self.precision)                
            else:
                merged_profile.precision['min'] = min(
                    self.precision['min'], other.precision['min'])
                merged_profile.precision['max'] = max(
                    self.precision['max'], other.precision['max'])
                merged_profile.precision['sum'] = \
                    self.precision['sum'] + other.precision['sum']
                merged_profile.precision['sample_size'] = \
                    self.precision['sample_size'] + other.precision['sample_size']

                merged_profile.precision['var'] = self._merge_variance(
                    self.precision['sample_size'],
                    self.precision['var'], self.precision['mean'],
                    other.precision['sample_size'], other.precision['var'],
                    other.precision['mean'])
                merged_profile.precision['mean'] = \
                    merged_profile.precision['sum'] \
                    / merged_profile.precision['sample_size']
            
                merged_profile.precision['std'] = math.sqrt(
                    merged_profile.precision['var'])

                # Margin of error, 99.9% confidence level
                merged_profile.precision['margin_of_error'] = \
                    merged_profile.__z_value_precision * (
                        merged_profile.precision['std']
                        / math.sqrt(merged_profile.precision['sample_size'])
                )
            
        return merged_profile

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.
        :return:
        """

        profile = dict(
            min=self.np_type_to_type(self.min),
            max=self.np_type_to_type(self.max),
            mean=self.np_type_to_type(self.mean),
            variance=self.np_type_to_type(self.variance),
            stddev=self.np_type_to_type(self.stddev),
            histogram=self._get_best_histogram_for_profile(),
            quantiles=self.quantiles,
            times=self.times,
            precision=dict(
                min=self.np_type_to_type(self.precision['min']),
                max=self.np_type_to_type(self.precision['max']),
                mean=self.np_type_to_type(self.precision['mean']),
                var=self.np_type_to_type(self.precision['var']),
                std=self.np_type_to_type(self.precision['std']),
                sample_size=self.np_type_to_type(self.precision['sample_size']),
                margin_of_error=self.np_type_to_type(self.precision['margin_of_error']),
                confidence_level=self.np_type_to_type(self.precision['confidence_level'])
            )
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
    def _get_float_precision(cls, df_series_clean, sample_ratio=None):
        """
        Determines the precision of the numeric value.
        
        :param df_series_clean: df series with nulls removed, assumes all values
            are floats as well
        :type df_series_clean: pandas.core.series.Series
        :param sample_ratio: Ratio of samples used for float precision
        :type sample_ratio: float (between 0 and 1)
        :return: string representing its precision print format
        :rtype: int
        """
        len_df = len(df_series_clean)
        if not len_df: return None

        # Lead zeros: ^[+-.0\s]+ End zeros: \.?0+(\s|$)
        # Scientific Notation: (?<=[e])(.*) Any non-digits: \D
        r = re.compile(r'^[+-.0\s]+|\.?0+(\s|$)|(?<=[e])(.*)|\D')

        # DEFAULT: Sample the dataset. If small use full dataset,
        # OR 20k samples or 5% of the dataset which ever is larger.
        # If user sets sample ratio, utilize their request
        sample_size = min(len_df, max(20000, int(len_df * 0.05)))
        if sample_ratio is not None and sample_ratio > 0:
            sample_size = int(len_df * sample_ratio)

        # length of sampled cells after all punctuation removed
        len_per_float = df_series_clean.sample(sample_size).replace(
            to_replace=r, value='').map(len)

        # Determine statistics precision
        precision_sum = len_per_float.sum()
        subset_precision = {
            'min': len_per_float.min(),
            'max': len_per_float.max(),
            'mean': precision_sum / sample_size,
            'var': float(len_per_float.var()),
            'sum': precision_sum,
            'sample_size': sample_size
        }
        
        return subset_precision
    
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
        if len(df_series) == 0: return list()
        return df_series.map(NumericStatsMixin.is_float)

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

        sample_ratio = None
        if self.__precision_sample_ratio is not None:
            sample_ratio = self.__precision_sample_ratio
        
        # (min, max, var, sum, sample_size)
        subset_precision = self._get_float_precision(df_series, sample_ratio)
        if subset_precision is None:
            return
        elif self.precision['min'] is None:
            self.precision.update(subset_precision)
        else:        
            # Update the calculations as data is valid
            self.precision['min'] = min(
                self.precision['min'], subset_precision['min'])
            self.precision['max'] = max(
                self.precision['max'], subset_precision['max'])            
            self.precision['sum'] += subset_precision['sum']
            
            self.precision['var'] = self._merge_variance(
                self.precision['sample_size'], self.precision['var'],
                self.precision['mean'],
                subset_precision['sample_size'], subset_precision['var'],
                subset_precision['mean'])
            
            self.precision['sample_size'] += subset_precision['sample_size']            
            self.precision['mean'] = self.precision['sum'] \
                / self.precision['sample_size']

        # Calculated outside
        self.precision['std'] = math.sqrt(self.precision['var'])

        # Margin of error, 99.9% confidence level
        self.precision['margin_of_error'] = self.__z_value_precision *(
            self.precision['std'] / math.sqrt(self.precision['sample_size']))

        # Set the significant figures
        sigfigs = int(self.precision['max'])
        for key in ['mean', 'var', 'std', 'margin_of_error']:
            self.precision[key] = \
                float('{:.{p}g}'.format(self.precision[key], p=sigfigs))
                        
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
        
    def update(self, df_series):
        """
        Updates the column profile.
        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: None
        """
        if len(df_series) == 0:
            return self
        
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

        return self
