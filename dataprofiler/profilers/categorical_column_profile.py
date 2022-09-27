"""Contains class for categorical column profiler."""
from __future__ import annotations

from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Optional

from pandas import DataFrame, Series

from . import BaseColumnProfiler, utils
from .profiler_options import CategoricalOptions


class CategoricalColumn(BaseColumnProfiler):
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

    def __init__(self, name: Optional[str], options: CategoricalOptions = None) -> None:
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
        super(CategoricalColumn, self).__init__(name)
        self._categories: Dict[str, int] = defaultdict(int)
        self.__calculations: Dict = {}
        self._filter_properties_w_options(self.__calculations, options)
        self._top_k_categories: Optional[int] = None
        if options:
            self._top_k_categories = options.top_k_categories

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
        merged_profile._categories = utils.add_nested_dictionaries(
            self._categories, other._categories
        )
        BaseColumnProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )
        return merged_profile

    def diff(self, other_profile: CategoricalColumn, options: Dict = None) -> Dict:
        """
        Find the differences for CategoricalColumns.

        :param other_profile: profile to find the difference with
        :type other_profile: CategoricalColumn
        :return: the CategoricalColumn differences
        :rtype: dict
        """
        differences: Dict = super().diff(other_profile, options)

        differences["categorical"] = utils.find_diff_of_strings_and_bools(
            self.is_match, other_profile.is_match
        )

        differences["statistics"] = dict(
            [
                (
                    "unique_count",
                    utils.find_diff_of_numbers(
                        len(self.categories), len(other_profile.categories)
                    ),
                ),
                (
                    "unique_ratio",
                    utils.find_diff_of_numbers(
                        self.unique_ratio, other_profile.unique_ratio
                    ),
                ),
            ]
        )

        # These stats are only diffed if both profiles are categorical
        if self.is_match and other_profile.is_match:
            differences["chi2-test"] = utils.perform_chi_squared_test_for_homogeneity(
                self._categories,
                self.sample_size,
                other_profile._categories,
                other_profile.sample_size,
            )
            differences["statistics"]["categories"] = utils.find_diff_of_lists_and_sets(
                self.categories, other_profile.categories
            )
            differences["statistics"]["gini_impurity"] = utils.find_diff_of_numbers(
                self.gini_impurity, other_profile.gini_impurity
            )
            differences["statistics"]["unalikeability"] = utils.find_diff_of_numbers(
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

            differences["statistics"]["categorical_count"] = utils.find_diff_of_dicts(
                cat_count1, cat_count2
            )

        return differences

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """
        Return report.

        This is a private abstract method.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        return self.profile

    @property
    def profile(self) -> Dict:
        """
        Return the profile of the column.

        For categorical_count, it will display the top k categories most
        frequently occurred in descending order.
        """
        profile: Dict = dict(
            categorical=self.is_match,
            statistics=dict(
                [
                    ("unique_count", len(self.categories)),
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
    def categories(self) -> List[str]:
        """Return categories."""
        return list(self._categories.keys())

    @property
    def categorical_counts(self) -> Dict[str, int]:
        """Return counts of each category."""
        return self._categories.copy()

    @property
    def unique_ratio(self) -> float:
        """Return ratio of unique categories to sample_size."""
        unique_ratio = 1.0
        if self.sample_size:
            unique_ratio = len(self.categories) / self.sample_size
        return unique_ratio

    @property
    def is_match(self) -> bool:
        """Return true if column is categorical."""
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

    @BaseColumnProfiler._timeit(name="categories")
    def _update_categories(
        self,
        df_series: DataFrame,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
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
        category_count = df_series.value_counts(dropna=False).to_dict()
        self._categories = utils.add_nested_dictionaries(
            self._categories, category_count
        )

    def _update_helper(self, df_series_clean: Series, profile: Dict) -> None:
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
        if len(df_series) == 0:
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

    @property
    def gini_impurity(self) -> Optional[float]:
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
    def unalikeability(self) -> Optional[float]:
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
