"""Index profile analysis for individual col within structured profiling."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, cast

from pandas import DataFrame, Series

from . import BaseColumnProfiler, utils
from .profiler_options import OrderOptions


class OrderColumn(BaseColumnProfiler):
    """
    Index column profile subclass of BaseColumnProfiler.

    Represents a column in the dataset which is an index column.
    """

    type = "order"

    def __init__(self, name: Optional[str], options: OrderOptions = None) -> None:
        """
        Initialize column base properties and self.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Order column
        :type options: OrderOptions
        """
        if options and not isinstance(options, OrderOptions):
            raise ValueError(
                "OrderColumn parameter 'options' must be of type" " OrderOptions."
            )
        self.order: Optional[str] = None
        self._last_value: Optional[int] = None
        self._first_value: Optional[int] = None
        self._piecewise: Optional[bool] = False
        self.__calculations: Dict = {}
        self._filter_properties_w_options(self.__calculations, options)
        super(OrderColumn, self).__init__(name)

    @staticmethod
    def _is_intersecting(
        first_value1: int, last_value1: int, first_value2: int, last_value2: int
    ) -> bool:
        """
        Check to see if the range of the datasets intersect.

        :param first_value1: beginning value of dataset 1
        :type first_value1: Integer
        :param last_value1: last value of dataset 1
        :type last_value1: Integer
        :param first_value2: beginning value of dataset 2
        :type first_value2: Integer
        :param last_value2: last value of dataset 2
        :type last_value2: Integer
        :return: Whether or not there is an intersection
        :rtype: Bool
        """
        # Some values may be in descending order so the true min/max must be
        # found
        low_value1 = min(first_value1, last_value1)
        high_value1 = max(first_value1, last_value1)
        low_value2 = min(first_value2, last_value2)
        high_value2 = max(first_value2, last_value2)
        is_intersecting = False

        if (
            (low_value2 < low_value1 < high_value2)
            or (low_value2 < high_value1 < high_value2)
            or (low_value1 < low_value2 < high_value1)
            or (low_value1 < high_value2 < high_value1)
            or (low_value1 == low_value2 and high_value1 == high_value2)
        ):
            is_intersecting = True
        return is_intersecting

    @staticmethod
    def _is_enveloping(
        first_value1: int, last_value1: int, first_value2: int, last_value2: int
    ) -> bool:
        """
        Check to see if the range of the dataset 1 envelopes dataset 2.

        :param first_value1: beginning value of dataset 1
        :type first_value1: Integer
        :param last_value1: last value of dataset 1
        :type last_value1: Integer
        :param first_value2: beginning value of dataset 2
        :type first_value2: Integer
        :param last_value2: last value of dataset 2
        :type last_value2: Integer
        :return: Whether or not there is an intersection
        :rtype: Bool
        """
        # Some values may be in descending order so the true min/max must be
        # found
        low_value1 = min(first_value1, last_value1)
        high_value1 = max(first_value1, last_value1)
        low_value2 = min(first_value2, last_value2)
        high_value2 = max(first_value2, last_value2)
        is_enveloping = False
        if low_value1 < low_value2 and high_value1 > high_value2:
            is_enveloping = True
        return is_enveloping

    @BaseColumnProfiler._timeit(name="order")
    def _merge_order(
        self,
        order1: str,
        first_value1: int,
        last_value1: int,
        piecewise1: bool,
        order2: str,
        first_value2: int,
        last_value2: int,
        piecewise2: bool,
    ) -> Tuple[str, int, int, bool]:
        """
        Add the order of two datasets together.

        :param order1: order of original dataset
        :param first_value1: beginning value of original dataset
        :param last_value1: last value of original dataset
        :param piecewise1: original dataset is piecewise or not
        :param order2: order of new dataset
        :param first_value2: beginning value of new dataset
        :param last_value2: last value of new dataset
        :param piecewise2: new dataset is piecewise or not
        :type order1: String
        :type first_value1: Integer
        :type last_value1: Integer
        :type piecewise1: Boolean
        :type order2: String
        :type first_value2: Integer
        :type last_value2: Integer
        :type piecewise2: Boolean
        :return: order, first_value, last_value, piecewise
        :rtype: String, Int, Int, Boolean
        """
        # Return either order if one is None
        if not order1:
            return order2, first_value2, last_value2, piecewise2
        elif not order2:
            return order1, first_value1, last_value1, piecewise1

        is_intersecting = self._is_intersecting(
            first_value1, last_value1, first_value2, last_value2
        )
        self_envelopes_other = self._is_enveloping(
            first_value1, last_value1, first_value2, last_value2
        )
        other_envelopes_self = self._is_enveloping(
            first_value2, last_value2, first_value1, last_value1
        )

        # Default initialization
        order = "random"
        first_value: Optional[int] = None
        last_value: Optional[int] = None

        if order1 == "random" or order2 == "random":
            order = "random"

        # Neither Random
        elif order1 == order2:
            if not is_intersecting or (piecewise1 and piecewise2):
                order = order1
            elif (
                piecewise1 and self_envelopes_other
            ):  # Intersect and not both piecewise
                order = order1
            elif piecewise2 and other_envelopes_self:
                order = order1
            elif order1 == "constant value":  # Constants that intersect
                order = order1
            else:  # Ascending/Descending that intersect but not both piecewise
                # and dont envelope one or the other
                order = "random"

        # Neither Random, nor the same order
        elif (order1 == "ascending" and order2 == "descending") or (
            order1 == "descending" and order2 == "ascending"
        ):
            order = "random"

        # One must be ascending/descending and one must be constant
        elif not is_intersecting:
            if order1 == "ascending" or order2 == "ascending":
                order = "ascending"
            else:  # order1 == 'descending' or order2 =='descending'
                order = "descending"

        # One must be ascending/descending and one must be constant and they
        # are intersecting
        else:
            if order1 == "constant value" and piecewise2:  # order2 ascend/descend
                order = order2
            elif order2 == "constant value" and piecewise1:  # order1 ascend/descend
                order = order1
            else:  # intersecting constant with non-piecewise ascend/descend
                order = "random"

        # Set variables
        if order == "ascending":
            first_value = min(first_value1, first_value2)
            last_value = max(last_value1, last_value2)
        elif order == "descending":
            first_value = max(first_value1, first_value2)
            last_value = min(last_value1, last_value2)
        elif order == "random" or order == "constant value":
            first_value = min(first_value1, first_value2, last_value1, last_value2)
            last_value = max(first_value1, first_value2, last_value1, last_value2)

        piecewise = True
        if (
            order == "constant value" and first_value == last_value
        ) or order == "random":
            piecewise = False

        return order, cast(int, first_value), cast(int, last_value), piecewise

    def __add__(self, other: OrderColumn) -> OrderColumn:
        """
        Merge the properties of two OrderColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: OrderColumn
        :type other: OrderColumn
        :return: Merged OrderColumn
        :rtype: OrderColumn
        """
        if not isinstance(other, OrderColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'OrderColumn' and '{}'".format(other.__class__.__name__)
            )

        merged_profile = OrderColumn(None)
        order, first_value, last_value, piecewise = self._merge_order(
            self.order,
            self._first_value,
            self._last_value,
            self._piecewise,
            other.order,
            other._first_value,
            other._last_value,
            other._piecewise,
        )

        merged_profile.order = order
        merged_profile._first_value = first_value
        merged_profile._last_value = last_value
        merged_profile._piecewise = piecewise

        BaseColumnProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )
        return merged_profile

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """
        Private abstract method for returning report.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        return self.profile

    @property
    def profile(self) -> Dict:
        """
        Property for profile. Returns the profile of the column.

        :return:
        """
        return dict(order=self.order, times=self.times)

    def diff(self, other_profile: OrderColumn, options: Dict = None) -> Dict:
        """
        Generate the differences between the orders of two OrderColumns.

        :return: Dict containing the differences between orders in their
        appropriate output formats
        :rtype: dict
        """
        super().diff(other_profile, options)

        differences = {
            "order": utils.find_diff_of_strings_and_bools(
                self.order, other_profile.order
            )
        }
        return differences

    @BaseColumnProfiler._timeit(name="order")
    def _get_data_order(self, df_series: Series) -> Tuple[str, float, float]:
        """
        Retrieve the order profile of a given data series.

        Return either: ascending, descending, constant value, or random.
        Additionally, return the first and last value of the series.

        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :return: order, first_value, last_value
        :rtype: String, Float, Float
        """
        try:
            df_series = df_series.astype(float)
        except ValueError:
            pass

        order = None
        last_value = df_series.iloc[0]
        first_value = df_series.iloc[0]

        for i in range(1, len(df_series)):
            value = df_series.iloc[i]
            if value < last_value and order == "ascending":
                order = "random"
                break
            elif value < last_value and order is None:
                order = "descending"
            elif value > last_value and order == "descending":
                order = "random"
                break
            elif value > last_value and order is None:
                order = "ascending"
            last_value = value
        if not order:
            order = "constant value"

        return order, first_value, last_value

    def _update_order(
        self,
        df_series: DataFrame,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """
        Update order profile with order info attained from new dataset.

        Do this in following two steps:
        1. Get order information from input column data.
        2. Merge information between existing profile and new column
           order information.

        :param df_series: Data to be profiled
        :type df_series: pandas.DataFrame
        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        if self.order == "random":
            return
        order, first_value, last_value = self._get_data_order(df_series)

        (
            self.order,
            self._first_value,
            self._last_value,
            self._piecewise,
        ) = self._merge_order(
            self.order,
            self._first_value,
            self._last_value,
            self._piecewise,
            order,
            first_value,
            last_value,
            piecewise2=False,
        )

    def _update_helper(self, df_series_clean: Series, profile: Dict) -> None:
        """
        Update col profile properties with clean dataset and its known null parameters.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: ordered profile
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series: Series) -> OrderColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: updated OrderColumn
        :rtype: OrderColumn
        """
        if len(df_series) == 0:
            return self

        profile = dict(sample_size=len(df_series))
        OrderColumn._update_order(self, df_series=df_series)
        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series,
            prev_dependent_properties={},
            subset_properties=profile,
        )
        self._update_helper(df_series, profile)

        return self
