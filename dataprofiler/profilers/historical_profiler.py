"""Build a collection of Profiles on same dataset over time."""
import copy
import datetime
import warnings
from queue import Queue
from typing import List

import numpy as np

import dataprofiler as dp

from .profile_builder import Profiler
from .utils import (
    find_diff_of_dates,
    find_diff_of_dicts_with_diff_keys,
    find_diff_of_lists_and_sets,
    find_diff_of_matrices,
    find_diff_of_numbers,
    find_diff_of_strings_and_bools,
)


class HistoricalProfiler:
    """
    HistoricalProfiler class.

    Stores several profiles that were generated on the same dataset
    taken over different points in time.
    """

    def __init__(
        self,
        profiles: List[Profiler],
        options=None,
    ):
        """
        Initialize a new HistoricalProfiler object from list of profiles & options.

        Assumes that the profiles provided in the list of profiles are stored in order
        with index 0 being the most recent profile and the last profile being the oldest
        """
        if options is None:
            options = self.historical_profiler_options()
        self.options = options

        if profiles is None:
            raise ValueError(
                "'profiles' is 'None', expected a list containing type `Profiler`"
            )
        if len(profiles) == 0:
            raise ValueError(
                "'profiles' is empty. At least one Profiler object is required"
            )
        for profile in profiles:
            # Only supports structured profiles for now
            if not isinstance(profile, dp.StructuredProfiler):
                raise ValueError(
                    "`profiles` has profile not of type `StructuredProfiler`."
                )

        profile_q = Queue(maxsize=0)
        for i in range(len(profiles) - 1, -1, -1):
            profile_q.put(profiles[i])

        historical_profile = {}
        oldest_profile = profile_q.get()
        oldest_report = oldest_profile.report(
            report_options={"output_format": self.options["output_format"]}
        )

        historical_profile["global_stats"] = self._wrap_dict_vals_in_list(
            oldest_report["global_stats"]
        )
        historical_profile["data_stats"] = []
        for column_stats in oldest_report["data_stats"]:
            historical_col = self._wrap_dict_vals_in_list(column_stats)
            historical_profile["data_stats"].append(historical_col)

        self.historical_profile = historical_profile
        self.length = 1

        while profile_q.qsize() != 0:
            current_profile = profile_q.get()
            self.append(current_profile)

    def _wrap_dict_vals_in_list(self, d: dict):
        """
        Copy value in dict to list.

        Makes a copy of the values stored in d, and puts them in a new dict
        under the same key but as the only element in a list. This method
        will act recursively on values that are of type `dict`.
        """
        wrapped_dict = {}
        for key, val in d.items():
            if key in self.options["exclude_keys"]:
                continue
            if type(val) == dict:
                wrapped_dict[key] = self._wrap_dict_vals_in_list(val)
            else:
                wrapped_dict[key] = [copy.deepcopy(val)]
        return wrapped_dict

    def _append_profile_values_to_dict(
        self, list_dict: dict, values_dict: dict, remove_oldest: bool
    ):
        """
        Prepends values in profile to front of list in dict.

        Takes the individual values stored in values_dict and inserts
        them into the front of the list values stored in list_dict

        This method will also remove the last value stored in the lists in list_dict
        in the case that adding the new value will result in the list becoming
        larger than `max_length` set in options.
        """
        for key, val in list_dict.items():
            if key in self.options["exclude_keys"]:
                continue
            if key not in values_dict:
                warnings.warn(
                    "'{}' key expected, but not contained in provided profile".format(
                        key
                    )
                )
                if isinstance(val, dict):
                    self._append_profile_values_to_dict(val, {}, remove_oldest)
                else:
                    val.insert(0, "NONE")
                    if remove_oldest:
                        val.pop()
                continue
            if isinstance(val, dict):
                self._append_profile_values_to_dict(
                    val, values_dict[key], remove_oldest
                )
            else:
                val.insert(0, copy.deepcopy(values_dict[key]))
                if remove_oldest:
                    val.pop()

    def append(self, profile):
        """
        Append the provided profile to the historical profile stored in this object.

        This method will assume that the provided profile is the most
        recent profile within this historical profile history.
        """
        if profile is None:
            raise ValueError(
                "`profile` is `None`. Expected object of type `dataprofiler.Profiler`"
            )
        historical_profile = self.historical_profile

        profile_report = profile.report(
            report_options={"output_format": self.options["output_format"]}
        )

        remove_oldest = False
        if self.length == self.options["max_length"]:
            remove_oldest = True

        hp_global_stats = historical_profile["global_stats"]
        profile_global_stats = profile_report["global_stats"]
        self._append_profile_values_to_dict(
            hp_global_stats, profile_global_stats, remove_oldest
        )

        hp_data_stats = historical_profile["data_stats"]
        profile_data_stats = profile_report["data_stats"]
        for hp_col, profile_col in zip(hp_data_stats, profile_data_stats):
            self._append_profile_values_to_dict(hp_col, profile_col, remove_oldest)
        if not remove_oldest:
            self.length += 1

    @classmethod
    def historical_profiler_options(self):
        """Return the default options for the HistoricalProfiler."""
        default_opts = {
            "max_length": None,
            "output_format": "serializable",
            "exclude_keys": [
                "categories",
                "gini_impurity",
                "unalikeability",
                "categorical_count",
            ],
        }
        return default_opts

    def report(self):
        """Return the historical profiler report."""
        return self.historical_profile

    def _get_value_from_index_in_dict_list(self, d, index):
        value_dict = {}
        for key, val in d.items():
            if isinstance(val, dict):
                value_dict[key] = self._get_value_from_index_in_dict_list(val, index)
            else:
                value_dict[key] = copy.deepcopy(val[index])
        return value_dict

    def get_profile_report_by_index(self, index: int):
        """Return the report of the profile stored at `index`."""
        if index is None:
            raise ValueError("`index` is `None`, expected `int` type")
        if index < 0 or index >= self.length:
            raise ValueError(
                "`index`: {} out of bounds within this historical profiler".format(
                    index
                )
            )

        hp = self.historical_profile

        profile_report = {}

        profile_report["global_stats"] = self._get_value_from_index_in_dict_list(
            hp["global_stats"], index
        )

        profile_report["data_stats"] = []
        for col in hp["data_stats"]:
            profile_report["data_stats"].append(
                self._get_value_from_index_in_dict_list(col, index)
            )

        return profile_report

    def get_most_recent_profile_report(self):
        """Return the most recent profile report stored in this object."""
        if self.historical_profile is None:
            raise ValueError("This Historical Profiler has not been initialized")

        return self.get_profile_report_by_index(0)

    def get_oldest_profile_report(self):
        """Return the oldest profile report stored in this object."""
        if self.historical_profile is None:
            raise ValueError("This Historical Profiler has not been initialized")

        return self.get_profile_report_by_index((self.length - 1))

    def _update_profile_values_in_dict_at_index(
        self, list_dict: dict, values_dict: dict, index: int
    ):
        """
        Insert individual values into list.

        Takes the individual values stored in values_dict and
        inserts them into the front of the list values stored in list_dict

        This method will also remove the last value stored in the lists in list_dict
        in the case that adding the new value will result in the list becoming
        larger than `max_length` set in options.
        """
        for key, val in list_dict.items():
            if key in self.options["exclude_keys"]:
                continue
            if key not in values_dict:
                warnings.warn(
                    "'{}' key expected but not contained in provided profile".format(
                        key
                    )
                )
                if isinstance(val, dict):
                    self._update_profile_values_in_dict_at_index(val, {}, index)
                else:
                    val[index] = "NONE"
                continue
            if isinstance(val, dict):
                self._update_profile_values_in_dict_at_index(
                    val, values_dict[key], index
                )
            else:
                val[index] = copy.deepcopy(values_dict[key])

    def update_profile_report_at_index(self, profile, index: int):
        """
        Update profile report at index.

        Updates the profile report stored at `index` within this historical profiler
        with the report generated from the profile provided
        """
        if index is None:
            raise ValueError("`index` is `None`, expected `int` type")
        if index < 0 or index >= self.length:
            raise ValueError(
                "`index`: {} out of bounds within this historical profiler".format(
                    index
                )
            )
        if profile is None:
            raise ValueError("`profile` is `None`, expected `Profiler` type")

        hp = self.historical_profile
        profile_report = profile.report(
            report_options={"output_format": self.options["output_format"]}
        )

        hp_global_stats = hp["global_stats"]
        pr_global_stats = profile_report["global_stats"]
        self._update_profile_values_in_dict_at_index(
            hp_global_stats, pr_global_stats, index
        )

        hp_data_stats = hp["data_stats"]
        pr_data_stats = profile_report["data_stats"]
        for hp_col, profile_col in zip(hp_data_stats, pr_data_stats):
            self._update_profile_values_in_dict_at_index(hp_col, profile_col, index)

    def _pop_value_from_index_in_dict_list(self, d, index):
        value_dict = {}
        for key, val in d.items():
            if isinstance(val, dict):
                value_dict[key] = self._pop_value_from_index_in_dict_list(val, index)
            else:
                value_dict[key] = val.pop(index)
        return value_dict

    def delete_profile_report_at_index(self, index):
        """
        Pop profile report from index.

        Removes and returns the profile report that is stored at the provided index.
        """
        if index is None:
            raise ValueError("`index` is `None`, expected `int` type")
        if index < 0 or index >= self.length:
            raise ValueError(
                "`index`: {} out of bounds within this historical profiler".format(
                    index
                )
            )

        hp = self.historical_profile
        removed_profile = {}

        removed_profile["global_stats"] = self._pop_value_from_index_in_dict_list(
            hp["global_stats"], index
        )

        removed_profile["data_stats"] = []
        for col in hp["data_stats"]:
            removed_profile["data_stats"].append(
                self._pop_value_from_index_in_dict_list(col, index)
            )

        self.length -= 1
        return removed_profile

    def _get_diff_function_type(self, val, val2):
        if type(val) != type(val2):
            return find_diff_of_strings_and_bools
        if isinstance(val, datetime.datetime):
            return find_diff_of_dates
        elif isinstance(val, dict):
            return find_diff_of_dicts_with_diff_keys
        elif isinstance(val, list) or isinstance(val, set):
            if len(val) > 0:
                if isinstance(val[0], list):
                    return find_diff_of_matrices
                else:
                    return find_diff_of_lists_and_sets
            else:
                return find_diff_of_lists_and_sets
        elif isinstance(val, int) or isinstance(val, float):
            return find_diff_of_numbers
        else:
            return find_diff_of_strings_and_bools

    def _get_dict_list_consecutive_deltas(self, d: dict):
        delta_dict = {}
        for key, val in d.items():
            if key == "column_name":
                delta_dict[key] = val
                continue
            if isinstance(val, dict):
                delta_dict[key] = self._get_dict_list_consecutive_deltas(val)
            else:
                deltas = []
                for i in range(0, len(val) - 1):
                    find_diff = self._get_diff_function_type(val[i], val[i + 1])
                    deltas.append(find_diff(val[i], val[i + 1]))
                delta_dict[key] = deltas
        return delta_dict

    def get_consecutive_diffs_report(self):
        """
        Return a report containing the consecutive deltas between historical reports.

        Output format:
            [profile_0-profile_1, profile_1-profile_2, ... ,profile_n-1-profile_n]
        """
        if self.length < 2:
            raise ValueError(
                "There must be at least two profiles stored in this object."
            )
        hp = self.historical_profile

        hp_diff_report = {}

        hp_diff_report["global_stats"] = self._get_dict_list_consecutive_deltas(
            hp["global_stats"]
        )

        hp_diff_data_stats = []
        hp_data_stats = hp["data_stats"]
        for data_stat in hp_data_stats:
            hp_diff_data_stats.append(self._get_dict_list_consecutive_deltas(data_stat))
        hp_diff_report["data_stats"] = hp_diff_data_stats

        return hp_diff_report

    def _convert_unchanged_in_list_to_zero(self, diff_list):
        """
        'unchanged' to '0' helper.

        Takes the list provided in `diff_list` and replaces the values of
        'unchanged' with '0'
        """
        for i in range(len(diff_list)):
            if isinstance(diff_list[i], dict):
                self.convert_consecutive_diffs_unchanged_to_zero(diff_list[i])
            elif isinstance(diff_list[i], list):
                self._convert_unchanged_in_list_to_zero(diff_list[i])
            elif type(diff_list[i]).__module__ != np.__name__:
                if diff_list[i] == "unchanged":
                    diff_list[i] = 0

    def convert_consecutive_diffs_unchanged_to_zero(self, d):
        """
        Convert 'unchanged' to '0'.

        Takes the diff report provided in `dict` and replaces the values of
        'unchanged' with 0.
        """
        for key, val in d.items():
            if isinstance(val, dict):
                self.convert_consecutive_diffs_unchanged_to_zero(val)
            elif isinstance(val, list):
                self._convert_unchanged_in_list_to_zero(val)
            elif val == "unchanged":
                d[key] = val
        return d

    def _get_min_and_max_from_dict_list(self, d):
        min_max_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                min_max_dict[key] = self._get_min_and_max_from_dict_list(value)
            else:
                if isinstance(value[0], int) or isinstance(value[0], float):
                    try:
                        min_max_dict[key] = (
                            np.min(value),
                            np.max(value),
                        )
                    except Exception as e:
                        min_max_dict[key] = e
                else:
                    try:
                        if value and all(value[0] == element for element in value):
                            min_max_dict[key] = value[0]
                        else:
                            min_max_dict[key] = value
                    except Exception:
                        pass
        return min_max_dict

    def get_diff_min_and_max_report(self):
        """
        Get a min/max report from this historical profiler.

        Returns a report containing the min and max values for each statistic for each
        field within a consecutive difference report. Output format:
        {
            {
                ...: (x, y) #Where x is the min, y is the max
            }
        }
        """
        hp = self.historical_profile

        if hp is None:
            raise ValueError("Historical Profiler has not been initialized")

        # consecutive diff report
        cdf = self.get_consecutive_diffs_report()
        cdf = self.convert_consecutive_diffs_unchanged_to_zero(cdf)
        min_max_report = {}

        min_max_report["global_stats"] = self._get_min_and_max_from_dict_list(
            cdf["global_stats"]
        )

        cdf_data_stats = cdf["data_stats"]
        min_max_data_stats = []
        for col in cdf_data_stats:
            min_max_data_stats.append(self._get_min_and_max_from_dict_list(col))
        min_max_report["data_stats"] = min_max_data_stats

        return min_max_report

    def __len__(self):
        """Return the number of profiles stored in this Object."""
        return self.length

    def get_statistics_limit_check_report_keys(
        self, report_dict: dict, keys: list, exclude_columns: list = []
    ):
        """
        Convert the report_dict into a limit_check_report_keys format.

        Args:
            - report_dict (dict): previously generated consecutive difference report
            - keys (list): the list of statistic keys to include in the output dict
            - exclude_columns (list): a list of columns to exclude from the output dict
        """
        if report_dict is None:
            raise ValueError("`report_dict` is None. Expected type `dict`")

        if "data_stats" not in report_dict:
            raise KeyError("'data_stats' missing from report dict")

        data_stats = report_dict["data_stats"]

        limit_check_report_keys = {}

        for col in data_stats:
            col_name = col["column_name"]
            if col_name in exclude_columns:
                continue
            limit_check_report_keys[col_name] = {}
            stats = col["statistics"]
            for stat, val in stats.items():
                if stat in keys:
                    limit_check_report_keys[col_name][stat] = {
                        "lower": val[0],
                        "upper": val[1],
                    }
        return limit_check_report_keys
