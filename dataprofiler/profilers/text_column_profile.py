"""Text profile analysis for individual col within structured profiling.."""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import utils
from .base_column_profilers import BaseColumnPrimitiveTypeProfiler, BaseColumnProfiler
from .numerical_column_stats import NumericStatsMixin
from .profiler_options import TextOptions


class TextColumn(NumericStatsMixin, BaseColumnPrimitiveTypeProfiler):
    """
    Text column profile subclass of BaseColumnProfiler.

    Represents a column in the dataset which is a text column.
    """

    type = "text"

    def __init__(self, name: Optional[str], options: TextOptions = None) -> None:
        """
        Initialize column base properties and itself.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Text column
        :type options: TextOptions
        """
        if options and not isinstance(options, TextOptions):
            raise ValueError(
                "TextColumn parameter 'options' must be of type" " TextOptions."
            )
        NumericStatsMixin.__init__(self, options)
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)
        self.vocab: List = list()
        self.__calculations = {"vocab": TextColumn._update_vocab}
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other: TextColumn) -> TextColumn:
        """
        Merge properties of two TextColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: TextColumn
        :type other: TextColumn
        :return: New TextColumn merged profile
        """
        if not isinstance(other, TextColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'TextColumn' and '{}'".format(other.__class__.__name__)
            )
        merged_profile = TextColumn(None)
        NumericStatsMixin._add_helper(merged_profile, self, other)
        BaseColumnPrimitiveTypeProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )
        if "vocab" in merged_profile.__calculations:
            merged_profile.vocab = self.vocab.copy()
            merged_profile._update_vocab(other.vocab)
        return merged_profile

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """Report profile attribute of class; potentially pop val from self.profile."""
        calcs_dict_keys = self._TextColumn__calculations.keys()
        profile = self.profile

        if remove_disabled_flag:
            profile_keys = list(profile.keys())
            for profile_key in profile_keys:
                if profile_key == "vocab":
                    if "vocab" in calcs_dict_keys:
                        continue
                profile.pop(profile_key)

        return profile

    @property
    def profile(self) -> Dict:
        """
        Return the profile of the column.

        :return:
        """
        profile = NumericStatsMixin.profile(self)
        # remove num_zeros and num_negative updated from numeric profile
        profile.pop("num_zeros")
        profile.pop("num_negatives")
        # and add the vocab update for text profile
        profile.update(dict(vocab=self.vocab))
        return profile

    def diff(self, other_profile: TextColumn, options: Dict = None) -> Dict:
        """
        Find the differences for text columns.

        :param other_profile: profile to find the difference with
        :type other_profile: TextColumn Profile
        :return: the text columns differences
        :rtype: dict
        """
        differences = NumericStatsMixin.diff(self, other_profile, options)
        del differences["psi"]
        vocab_diff = utils.find_diff_of_lists_and_sets(self.vocab, other_profile.vocab)
        differences["vocab"] = vocab_diff
        return differences

    @property
    def data_type_ratio(self) -> Optional[float]:
        """
        Calculate the ratio of samples which match this data type.

        NOTE: all values can be considered string so always returns 1 in this
        case.

        :return: ratio of data type
        :rtype: float
        """
        return 1.0 if self.sample_size else None

    @BaseColumnProfiler._timeit(name="vocab")
    def _update_vocab(
        self,
        data: Union[List, np.ndarray, pd.DataFrame],
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """
        Find the unique vocabulary used in the text column.

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

    def _update_helper(self, df_series_clean: pd.Series, profile: Dict) -> None:
        """
        Update col profile properties with clean dataset and its known null parameters.

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
            self.type = "string" if self.max <= 255 else "text"

    def update(self, df_series: pd.Series) -> TextColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: updated TextColumn
        :rtype: TextColumn
        """
        len_df = len(df_series)
        if len_df == 0:
            return self

        profile = dict(match_count=len_df, sample_size=len_df)

        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series,
            prev_dependent_properties={},
            subset_properties=profile,
        )

        self._update_helper(df_series, profile)

        return self
