"""Contains class for categorical column profiler."""
from __future__ import annotations

import math
from collections import defaultdict
from operator import itemgetter
from typing import cast

import datasketches
from pandas import DataFrame, Series

from .. import dp_logging
from . import profiler_utils
from .base_column_profilers import BaseColumnProfiler
from .profiler_options import CategoricalOptions

logger = dp_logging.get_child_logger(__name__)


class CategoricalColumn(BaseColumnProfiler["CategoricalColumn"]):
    """
    Categorical column profile subclass of BaseColumnProfiler.

    Represents a column int the dataset which is a categorical column.
    """

    type = "category"

    # If total number of unique values in a column is less than this value,
    # that column is classified as a categorical column.
    _MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL = 10

    # Default value that determines if a given col is categorical or not.
    _CATEGORICAL_THRESHOLD_DEFAULT = 0.2

    def __init__(self, name: str | None, options: CategoricalOptions = None) -> None:
        """
        Initialize column base properties and itself.

        :param name: Name of data
        :type name: String
        """
        if options and not isinstance(options, CategoricalOptions):
            raise ValueError(
                "CategoricalColumn parameter 'options' must be of"
                " type CategoricalOptions."
            )
        super().__init__(name)
        self._categories: dict[str, int] = defaultdict(int)
        self.__calculations: dict = {}
        self._filter_properties_w_options(self.__calculations, options)
        self._top_k_categories: int | None = None

        # Conditions to stop categorical profiling
        self.max_sample_size_to_check_stop_condition = None
        self.stop_condition_unique_value_ratio = None
        self._stop_condition_is_met = False

        self._stopped_at_unique_ratio: float | None = None
        self._stopped_at_unique_count: int | None = None

        self._cms_max_num_heavy_hitters: int | None = 5000
        self.cms_num_hashes: int | None = None
        self.cms_num_buckets: int | None = None
        self.cms: datasketches.countminsketch | None = None
        if options:
            self._top_k_categories = options.top_k_categories
            self.stop_condition_unique_value_ratio = (
                options.stop_condition_unique_value_ratio
            )
            self.max_sample_size_to_check_stop_condition = (
                options.max_sample_size_to_check_stop_condition
            )

            if options.cms:
                self._cms_max_num_heavy_hitters = options.cms_max_num_heavy_hitters
                self.cms_num_hashes = datasketches.count_min_sketch.suggest_num_hashes(
                    options.cms_confidence
                )
                self.cms_num_buckets = (
                    datasketches.count_min_sketch.suggest_num_buckets(
                        options.cms_relative_error
                    )
                )
                self.cms = datasketches.count_min_sketch(
                    self.cms_num_hashes, self.cms_num_buckets
                )

    def __add__(self, other: CategoricalColumn) -> CategoricalColumn:
        """
        Merge the properties of two CategoricalColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: CategoricalColumn
        :type other: CategoricalColumn
        :return: New CategoricalColumn merged profile
        """
        if not isinstance(other, CategoricalColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'CategoricalColumn' and '{}'".format(other.__class__.__name__)
            )

        merged_profile = CategoricalColumn(None)
        BaseColumnProfiler._add_helper(merged_profile, self, other)

        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )

        if self.cms and other.cms:

            assert isinstance(self._cms_max_num_heavy_hitters, int)
            assert isinstance(other._cms_max_num_heavy_hitters, int)
            cms_max_num_heavy_hitters: int = min(
                self._cms_max_num_heavy_hitters, other._cms_max_num_heavy_hitters
            )

            (
                merged_profile.cms,
                merged_profile._categories,
                merged_profile._cms_max_num_heavy_hitters,
            ) = self._merge_categories_cms(
                self.cms,
                self._categories,
                self.sample_size,
                {},
                other.cms,
                other._categories,
                other.sample_size,
                cms_max_num_heavy_hitters,
            )

        elif not self.cms and not other.cms:
            # If both profiles have not met stop condition
            if not (self._stop_condition_is_met or other._stop_condition_is_met):
                merged_profile._categories = profiler_utils.add_nested_dictionaries(
                    self._categories, other._categories
                )

                # Transfer stop condition variables of 1st profile object to
                # merged profile if they are not None else set to 2nd profile
                profile1_product = self.sample_size * self.unique_ratio
                profile2_product = other.sample_size * other.unique_ratio
                if profile1_product < profile2_product:
                    merged_profile.max_sample_size_to_check_stop_condition = (
                        self.max_sample_size_to_check_stop_condition
                    )
                    merged_profile.stop_condition_unique_value_ratio = (
                        self.stop_condition_unique_value_ratio
                    )
                else:
                    merged_profile.stop_condition_unique_value_ratio = (
                        other.stop_condition_unique_value_ratio
                    )
                    merged_profile.max_sample_size_to_check_stop_condition = (
                        other.max_sample_size_to_check_stop_condition
                    )

                # Check merged profile w/ stop condition
                if merged_profile._check_stop_condition_is_met(
                    merged_profile.sample_size, merged_profile.unique_ratio
                ):
                    merged_profile._stopped_at_unique_ratio = (
                        merged_profile.unique_ratio
                    )
                    merged_profile._stopped_at_unique_count = (
                        merged_profile.unique_count
                    )
                    merged_profile._categories = {}
                    merged_profile._stop_condition_is_met = True

            else:
                if self.sample_size > other.sample_size:
                    merged_profile._stopped_at_unique_ratio = self.unique_ratio
                    merged_profile._stopped_at_unique_count = self.unique_count
                    merged_profile.sample_size = self.sample_size
                else:
                    merged_profile._stopped_at_unique_ratio = other.unique_ratio
                    merged_profile._stopped_at_unique_count = other.unique_count
                    merged_profile.sample_size = other.sample_size

                # If either profile has hit stop condition, remove categories dict
                merged_profile._categories = {}
                merged_profile._stop_condition_is_met = True

        else:
            raise Exception(
                "Unable to add two profiles: One is using count min sketch"
                "and the other is using full."
            )

        return merged_profile

    @property
    def gini_impurity(self) -> float | None:
        """
        Return Gini Impurity.

        Gini Impurity is a way to calculate
        likelihood of an incorrect classification of a new instance of
        a random variable.

        G = Σ(i=1; J): P(i) * (1 - P(i)), where i is the category classes.
        We are traversing through categories and calculating with the column

        :return: None or Gini Impurity probability
        """
        if self.sample_size == 0:
            return None
        gini_sum: float = 0
        for i in self._categories:
            gini_sum += (self._categories[i] / self.sample_size) * (
                1 - (self._categories[i] / self.sample_size)
            )
        return gini_sum

    @property
    def unalikeability(self) -> float | None:
        """
        Return Unlikeability.

        Unikeability checks for "how often observations differ from one another"
        Reference: Perry, M. and Kader, G. Variation as Unalikeability.
        Teaching Statistics, Vol. 27, No. 2 (2005), pp. 58-60.

        U = Σ(i=1,n)Σ(j=1,n): (Cij)/(n**2-n)
        Cij = 1 if i!=j, 0 if i=j

        :return: None or unlikeability probability
        """
        if self.sample_size == 0:
            return None
        elif self.sample_size == 1:
            return 0
        unalike_sum: int = 0
        for category in self._categories:
            unalike_sum += (
                self.sample_size - self._categories[category]
            ) * self._categories[category]
        unalike: float = unalike_sum / (self.sample_size**2 - self.sample_size)
        return unalike

    def diff(self, other_profile: CategoricalColumn, options: dict = None) -> dict:
        """
        Find the differences for CategoricalColumns.

        :param other_profile: profile to find the difference with
        :type other_profile: CategoricalColumn
        :return: the CategoricalColumn differences
        :rtype: dict
        """
        # Make sure other_profile's type matches this class
        differences: dict = super().diff(other_profile, options)

        differences["categorical"] = profiler_utils.find_diff_of_strings_and_bools(
            self.is_match, other_profile.is_match
        )

        differences["statistics"] = dict(
            [
                (
                    "unique_count",
                    profiler_utils.find_diff_of_numbers(
                        self.unique_count, other_profile.unique_count
                    ),
                ),
                (
                    "unique_ratio",
                    profiler_utils.find_diff_of_numbers(
                        self.unique_ratio, other_profile.unique_ratio
                    ),
                ),
            ]
        )

        # These stats are only diffed if both profiles are categorical
        if self.is_match and other_profile.is_match:
            differences["statistics"][
                "chi2-test"
            ] = profiler_utils.perform_chi_squared_test_for_homogeneity(
                self._categories,
                self.sample_size,
                other_profile._categories,
                other_profile.sample_size,
            )
            differences["statistics"][
                "categories"
            ] = profiler_utils.find_diff_of_lists_and_sets(
                self.categories, other_profile.categories
            )
            differences["statistics"][
                "gini_impurity"
            ] = profiler_utils.find_diff_of_numbers(
                self.gini_impurity, other_profile.gini_impurity
            )
            differences["statistics"][
                "unalikeability"
            ] = profiler_utils.find_diff_of_numbers(
                self.unalikeability, other_profile.unalikeability
            )
            cat_count1 = dict(
                sorted(self._categories.items(), key=itemgetter(1), reverse=True)
            )
            cat_count2 = dict(
                sorted(
                    other_profile._categories.items(), key=itemgetter(1), reverse=True
                )
            )
            (
                self_cat_count,
                other_cat_count,
            ) = self._preprocess_for_categorical_psi_calculation(
                self_cat_count=cat_count1,
                other_cat_count=cat_count2,
            )

            total_psi = 0.0
            for iter_key in self_cat_count.keys():
                percent_self = self_cat_count[iter_key] / self.sample_size
                percent_other = other_cat_count[iter_key] / other_profile.sample_size
                if (percent_other != 0) and (percent_self != 0):
                    total_psi += (percent_other - percent_self) * math.log(
                        percent_other / percent_self
                    )
                differences["statistics"]["psi"] = total_psi

            differences["statistics"][
                "categorical_count"
            ] = profiler_utils.find_diff_of_dicts(self_cat_count, other_cat_count)

        return differences

    def report(self, remove_disabled_flag: bool = False) -> dict:
        """
        Return report.

        This is a private abstract method.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        return self.profile

    @classmethod
    def load_from_dict(cls, data: dict, config: dict | None = None):
        """
        Parse attribute from json dictionary into self.

        :param data: dictionary with attributes and values.
        :type data: dict[string, Any]
        :param config: config for loading column profiler params from dictionary
        :type config: Dict | None

        :return: Profiler with attributes populated.
        :rtype: CategoricalColumn
        """
        value = data.pop("_categories")
        profile = super().load_from_dict(data)
        setattr(profile, "_categories", defaultdict(int, value))
        return profile

    @property
    def profile(self) -> dict:
        """
        Return the profile of the column.

        For categorical_count, it will display the top k categories most
        frequently occurred in descending order.
        """
        profile: dict = dict(
            categorical=self.is_match,
            statistics=dict(
                [
                    ("unique_count", self.unique_count),
                    ("unique_ratio", self.unique_ratio),
                ]
            ),
            times=self.times,
        )
        if self.is_match:
            profile["statistics"]["categories"] = self.categories
            profile["statistics"]["gini_impurity"] = self.gini_impurity
            profile["statistics"]["unalikeability"] = self.unalikeability
            profile["statistics"]["categorical_count"] = dict(
                sorted(self._categories.items(), key=itemgetter(1), reverse=True)[
                    : self._top_k_categories
                ]
            )
        return profile

    @property
    def categories(self) -> list[str]:
        """Return categories."""
        return list(self._categories.keys())

    @property
    def categorical_counts(self) -> dict[str, int]:
        """Return counts of each category."""
        return self._categories.copy()

    @property
    def unique_ratio(self) -> float:
        """Return ratio of unique categories to sample_size."""
        if self._stop_condition_is_met:
            return cast(float, self._stopped_at_unique_ratio)

        if self.sample_size:
            return len(self.categories) / self.sample_size
        return 0

    @property
    def unique_count(self) -> int:
        """Return ratio of unique categories to sample_size."""
        if self._stop_condition_is_met:
            return cast(int, self._stopped_at_unique_count)

        return len(self.categories)

    @property
    def is_match(self) -> bool:
        """Return true if column is categorical."""
        if self._stop_condition_is_met:
            return False

        is_match = False
        unique = len(self._categories)
        if unique <= self._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL:
            is_match = True
        elif (
            self.sample_size
            and self.unique_ratio <= self._CATEGORICAL_THRESHOLD_DEFAULT
        ):
            is_match = True
        return is_match

    def _preprocess_for_categorical_psi_calculation(
        self, self_cat_count, other_cat_count
    ):
        super_set_categories = set(self_cat_count.keys()) | set(other_cat_count.keys())
        if self_cat_count.keys() != other_cat_count.keys():
            logger.info(
                f"""PSI data pre-processing found that categories between
                    the profiles were not equal. Both profiles do not contain
                    the following categories {super_set_categories}."""
            )

        for iter_key in super_set_categories:
            for iter_dictionary in [self_cat_count, other_cat_count]:
                try:
                    iter_dictionary[iter_key] = iter_dictionary[iter_key]
                except KeyError:
                    iter_dictionary[iter_key] = 0
        return self_cat_count, other_cat_count

    def _check_stop_condition_is_met(self, sample_size: int, unqiue_ratio: float):
        """Return boolean given stop conditions.

        :param sample_size: Number of samples to check the stop condition
        :type sample_size: int
        :param unqiue_ratio: Ratio of unique values to full sample size to
            check stop condition
        :type unqiue_ratio: float
        :return: boolean for stop conditions
        """
        if (
            self.max_sample_size_to_check_stop_condition is not None
            and self.stop_condition_unique_value_ratio is not None
            and sample_size >= self.max_sample_size_to_check_stop_condition
            and unqiue_ratio >= self.stop_condition_unique_value_ratio
        ):
            return True
        return False

    def _update_stop_condition(self, data: DataFrame):
        """Return value stop_condition_is_met given stop conditions.

        :param data: Dataframe currently being processed by categorical profiler
        :type data: DataFrame
        :return: boolean for stop conditions
        """
        merged_unique_count = len(self._categories)
        merged_sample_size = self.sample_size + len(data)
        merged_unique_ratio = merged_unique_count / merged_sample_size

        self._stop_condition_is_met = self._check_stop_condition_is_met(
            merged_sample_size, merged_unique_ratio
        )
        if self._stop_condition_is_met:
            self._stopped_at_unique_ratio = merged_unique_ratio
            self._stopped_at_unique_count = merged_unique_count

    @BaseColumnProfiler._timeit(name="categories")
    def _get_categories_cms(self, df_series, len_df):
        """Return count min sketch and heavy hitters for both the batch and stream case.

        :param df_series: Series currently being processed by categorical profiler
        :type df_series: Series
        :param len_df: the total number of samples iin df_series
        :type len_df: int
        :return: cms, heavy_hitter_dict, missing_heavy_hitter_dict
        """
        cms = datasketches.count_min_sketch(self.cms_num_hashes, self.cms_num_buckets)
        heavy_hitter_dict = defaultdict(int)
        missing_heavy_hitter_dict = defaultdict(int)
        for i, value in enumerate(df_series):
            cms.update(value)
            i_count = cms.get_estimate(value)
            i_total_count = i_count + self.cms.get_estimate(value)
            # approximate heavy-hitters
            if i_count >= int(len_df / self._cms_max_num_heavy_hitters):
                heavy_hitter_dict[value] = i_count
                missing_heavy_hitter_dict.pop(value, None)
            elif i_total_count >= int(
                (self.sample_size + len_df) / self._cms_max_num_heavy_hitters
            ):
                missing_heavy_hitter_dict[value] = i_total_count
                heavy_hitter_dict.pop(value, None)

        return cms, heavy_hitter_dict, missing_heavy_hitter_dict

    @BaseColumnProfiler._timeit(name="categories")
    def _merge_categories_cms(
        self,
        cms1,
        heavy_hitter_dict1,
        len1,
        missing_heavy_hitter_dict,
        cms2,
        heavy_hitter_dict2,
        len2,
        max_num_heavy_hitters,
    ):
        """Return the aggregate count min sketch and approximate histogram (categories).

        :param cms1: count min sketch
        :type cms1: datasketches.countminsketch
        :param cms2: count min sketch
        :type cms2: datasketches.countminsketch
        :param heavy_hitter_dict1: Heavy Hitters category count
        :type heavy_hitter_dict1: Dict
        :param heavy_hitter_dict2: Heavy Hitters category count
        :type heavy_hitter_dict2: Dict
        :param missing_heavy_hitter_dict: Heavy Hitters category count
        considering two batches are two chunks of a data stream
        :type missing_heavy_hitter_dict: Dict
        :param len1: number of samples in batch 1
        :type len1: int
        :param len2: number of samples in batch 2
        :type len2: int
        :param max_num_heavy_hitters: value used to define
        the threshold for minimum frequency required by a category to be counted
        :type max_num_heavy_hitters: int
        :return: cms1, categories, max_num_heavy_hitters
        """
        try:
            cms3 = datasketches.count_min_sketch(
                self.cms_num_hashes, self.cms_num_buckets
            )
            cms3.merge(cms1)
            cms3.merge(cms2)
        except ValueError as err:
            raise err(
                """Incompatible sketch configuration. When merging two sketches,
                they must have the same number of buckets and hashes,
                which are defined by cms_confidence and cms_relative_error options,
                respectively."""
            )

        # re-collecting the estimates of non intersecting categories before
        # re-applying heavy-hitters to the aggregate profile.
        heavy_hitter_dict1 = heavy_hitter_dict1.copy()
        heavy_hitter_dict2 = heavy_hitter_dict2.copy()
        for k in (x for x in heavy_hitter_dict1 if x not in heavy_hitter_dict2):
            heavy_hitter_dict2[k] = cms2.get_estimate(k)
        for k in (x for x in heavy_hitter_dict2 if x not in heavy_hitter_dict1):
            heavy_hitter_dict1[k] = cms1.get_estimate(k)

        categories = profiler_utils.add_nested_dictionaries(
            heavy_hitter_dict2, heavy_hitter_dict1
        )

        # This is a catch all for edge cases where batch heavy hitters under estimates
        # frequencies compared to treated as a sequence of batches as part of
        # the same stream.
        categories.update(missing_heavy_hitter_dict)

        total_samples = len1 + len2
        for cat in list(categories):
            if categories[cat] < (total_samples / max_num_heavy_hitters):
                categories.pop(cat)
        return cms3, categories, max_num_heavy_hitters

    def _get_categories_full(self, df_series) -> dict:
        """Get the unique counts (categories) of a series.

        :param df_series: df series with nulls removed
        :type df_series: pandas.core.series.Series
        :return: dict of counts for each unique value
        :rtype: dict
        """
        category_count: dict = df_series.value_counts(dropna=False).to_dict()
        return category_count

    @BaseColumnProfiler._timeit(name="categories")
    def _update_categories(
        self,
        df_series: DataFrame,
        prev_dependent_properties: dict = None,
        subset_properties: dict = None,
    ) -> None:
        """
        Check whether column corresponds to category type.

        Adds category parameters if it is.

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
        if self.cms is not None:
            if self._cms_max_num_heavy_hitters is None:
                raise ValueError(
                    "when using CMS, cms_max_num_heavy_hitters must be an integer"
                )
            len_df = len(df_series)
            (
                cms,
                heavy_hitter_dict,
                missing_heavy_hitter_dict,
            ) = self._get_categories_cms(df_series, len_df)

            self.cms, self._categories, _ = self._merge_categories_cms(
                cms,
                heavy_hitter_dict,
                len_df,
                missing_heavy_hitter_dict,
                self.cms,
                self._categories,
                self.sample_size,
                self._cms_max_num_heavy_hitters,
            )
        else:
            category_count = self._get_categories_full(df_series)
            self._categories = profiler_utils.add_nested_dictionaries(
                self._categories, category_count
            )
            self._update_stop_condition(df_series)
            if self._stop_condition_is_met:
                self._categories = {}

    def _update_helper(self, df_series_clean: Series, profile: dict) -> None:
        """
        Update col profile properties with clean dataset and its known profile.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: categorical profile dictionary
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series: Series) -> CategoricalColumn:
        """
        Update the column profile.

        :param df_series: Data to profile.
        :type df_series: pandas.core.series.Series
        :return: updated CategoricalColumn
        :rtype: CategoricalColumn
        """
        # If condition for limiting profile calculations
        if len(df_series) == 0 or self._stop_condition_is_met:
            return self

        profile = dict(sample_size=len(df_series))
        CategoricalColumn._update_categories(self, df_series)
        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series,
            prev_dependent_properties={},
            subset_properties=profile,
        )

        self._update_helper(df_series, profile)

        return self
