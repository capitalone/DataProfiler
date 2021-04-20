from .numerical_column_stats import NumericStatsMixin
from .base_column_profilers import BaseColumnProfiler, \
    BaseColumnPrimitiveTypeProfiler
from .profiler_options import TextOptions
from . import utils
import itertools


class TextColumn(NumericStatsMixin, BaseColumnPrimitiveTypeProfiler):
    """
    Text column profile subclass of BaseColumnProfiler. Represents a column in
    the dataset which is a text column.
    """
    col_type = "text"
    
    def __init__(self, name, options=None):
        """
        Initialization of column base properties and itself.
        
        :param name: Name of the data
        :type name: String
        :param options: Options for the Text column
        :type options: TextOptions
        """
        if options and not isinstance(options, TextOptions):
            raise ValueError("TextColumn parameter 'options' must be of type"
                             " TextOptions.")
        NumericStatsMixin.__init__(self, options)
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)
        self.vocab = list()
        self.__calculations = {
            "vocab": TextColumn._update_vocab
        }
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other):
        """
        Merges the properties of two TextColumn profiles
        
        :param self: first profile
        :param other: second profile
        :type self: TextColumn
        :type other: TextColumn
        :return: New TextColumn merged profile
        """
        if not isinstance(other, TextColumn):
            raise TypeError("Unsupported operand type(s) for +: "
                            "'TextColumn' and '{}'".format(other.__class__.__name__))
        merged_profile = TextColumn(None)
        NumericStatsMixin._add_helper(merged_profile, self, other)
        BaseColumnPrimitiveTypeProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(merged_profile.__calculations,
                                 self.__calculations,
                                 other.__calculations)
        if "vocab" in merged_profile.__calculations:
            merged_profile.vocab = self.vocab.copy()
            merged_profile._update_vocab(other.vocab)
        return merged_profile

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.
        
        :return:
        """

        profile = dict(
            min=self.min,
            max=self.max,
            mean=self.mean,
            variance=self.variance,
            stddev=self.stddev,
            histogram=self._get_best_histogram_for_profile(),
            quantiles=self.quantiles,
            vocab=self.vocab,
            times=self.times
        )
        return profile

    @property
    def data_type_ratio(self):
        """
        Calculates the ratio of samples which match this data type.
        NOTE: all values can be considered string so always returns 1 in this
        case.
        
        :return: ratio of data type
        :rtype: float
        """
        return 1.0 if self.sample_size else None
    
    @BaseColumnProfiler._timeit(name='vocab')
    def _update_vocab(self, data, prev_dependent_properties=None,
                      subset_properties=None):
        """
        Finds the unique vocabulary used in the text column.

        :param data: list or array of data from which to extract vocab
        :type data: Union[list, numpy.array, pandas.DataFrame]
        :param prev_dependent_properties: Contains all the previous properties
            that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
            subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        
        data_flat = list(itertools.chain(*data))
        self.vocab = utils._combine_unique_sets(self.vocab, data_flat)
        

    def _update_helper(self, df_series_clean, profile):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known null parameters of the dataset.
        
        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: text profile dictionary
        :type profile: dict
        :return: None
        """
        if self._NumericStatsMixin__calculations:
            text_lengths = df_series_clean.str.len()
            NumericStatsMixin._update_helper(self, text_lengths, profile)
        self._update_column_base_properties(profile)
        if self.max:
            self.col_type = 'string' if self.max <= 255 else 'text'

    def update(self, df_series):
        """
        Updates the column profile.
        
        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: None
        """
        len_df = len(df_series)
        if len_df == 0:
            return self
        
        profile = dict(match_count=len_df, sample_size=len_df)

        BaseColumnProfiler._perform_property_calcs(
            self, self.__calculations, df_series=df_series,
            prev_dependent_properties={}, subset_properties=profile)

        self._update_helper(df_series, profile)

        return self
