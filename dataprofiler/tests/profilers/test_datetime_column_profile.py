import datetime
import json
import unittest
import warnings
from collections import defaultdict
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataprofiler.profilers import DateTimeColumn
from dataprofiler.profilers.json_decoder import load_column_profile
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import DateTimeOptions

from . import utils as test_utils

# This is taken from: https://github.com/rlworkgroup/dowel/pull/36/files
# undo when cpython#4800 is merged.
unittest.case._AssertWarnsContext.__enter__ = test_utils.patched_assert_warns


class TestDateTimeColumnProfiler(unittest.TestCase):
    def setUp(self):
        test_utils.set_seed(seed=0)

    @staticmethod
    def _generate_datetime_data(date_format):

        gen_data = []
        for i in range(50):
            start_date = pd.Timestamp(1950, 7, 14)
            end_date = pd.Timestamp(2020, 7, 14)

            date_sample = test_utils.generate_random_date_sample(
                start_date, end_date, [date_format]
            )
            gen_data.append(date_sample)

        return pd.Series(gen_data)

    def _test_datetime_detection_helper(self, date_formats):

        for date_format in date_formats:
            # generate a few samples for each date format
            gen_data = self._generate_datetime_data(date_format)

            # Test to see if the format and col type is detected correctly.
            datetime_profile = DateTimeColumn(gen_data.name)
            datetime_profile.update(gen_data)

            self.assertEqual(date_format, datetime_profile.date_formats[0])

    def test_base_case(self):
        data = pd.Series([], dtype=object)
        profiler = DateTimeColumn(data.name)
        profiler.update(data)
        profiler.update(data)  # intentional to validate no changes if empty

        self.assertEqual(profiler.match_count, 0)
        self.assertIsNone(profiler.min)
        self.assertIsNone(profiler.max)
        self.assertListEqual([], profiler.date_formats)
        self.assertIsNone(profiler.data_type_ratio)

    def test_profiled_date_time_formats(self):
        """
        Checks whether the profiler properly determines all datetime formats.
        :return:
        """
        date_formats_1 = [
            "%Y-%m-%d %H:%M:%S",  # 2013-03-5 15:43:30
            "%Y-%m-%dT%H:%M:%S",  # 2013-03-6T15:43:30
            "%Y-%m-%dT%H:%M:%S.%fZ",  # 2013-03-6T15:43:30.123456Z
            "%m/%d/%y %H:%M",  # 03/10/13 15:43
            "%m/%d/%Y %H:%M",  # 3/8/2013 15:43
            "%Y%m%dT%H%M%S",  # 2013036T154330
            "%H:%M:%S.%f",  # 05:46:30.258509
        ]
        df_1 = pd.Series([], dtype=object)
        for date_format in date_formats_1:
            # generate a few samples for each date format
            df_1 = pd.concat([df_1, self._generate_datetime_data(date_format)])

        date_formats_2 = [
            "%Y-%m-%d",  # 2013-03-7
            "%m/%d/%Y",  # 3/8/2013
            "%m/%d/%y",  # 03/10/13
            "%B %d, %Y",  # March 9, 2013
            "%b %d, %Y",  # Mar 11, 2013
            "%d%b%y",  # 12Mar13
            "%b-%d-%y",  # Mar-13-13
            "%m%d%Y",  # 03142013
        ]
        df_2 = pd.Series([], dtype=object)
        for date_format in date_formats_2:
            # generate a few samples for each date format
            df_2 = pd.concat([df_2, self._generate_datetime_data(date_format)])

        date_formats_all = date_formats_1 + date_formats_2
        df_all = pd.concat([df_1, df_2])
        datetime_profile = DateTimeColumn(df_all.name)
        datetime_profile.update(df_all)

        self.assertCountEqual(date_formats_all, set(datetime_profile.date_formats))

        # Test chunks
        datetime_profile = DateTimeColumn(df_1.name)
        datetime_profile.update(df_1)

        self.assertCountEqual(date_formats_1, set(datetime_profile.date_formats))

        datetime_profile.update(df_2)
        self.assertCountEqual(date_formats_all, datetime_profile.date_formats)

    def test_profiled_min(self):
        def date_linspace(start, end, steps):
            delta = (end - start) / steps
            increments = list(range(0, steps)) * np.array([delta] * steps)
            return start + increments

        df = pd.core.series.Series(
            date_linspace(datetime.datetime.min, datetime.datetime.max, 11)
        )
        df = df.apply(
            lambda x: x - datetime.timedelta(microseconds=x.microsecond)
        ).apply(str)

        datetime_profile = DateTimeColumn(df[1:].name)
        datetime_profile.update(df[1:])

        self.assertEqual(datetime_profile.min, df.iloc[1])

        datetime_profile.update(df)
        self.assertEqual(datetime_profile.min, df.iloc[0])

        datetime_profile.update(pd.Series([np.nan, df.iloc[3]]))
        self.assertEqual(datetime_profile.min, df.iloc[0])

        datetime_profile.update(df[1:2])  # only way to keep as df
        self.assertEqual(datetime_profile.min, df.iloc[0])

    def test_profiled_max(self):
        def date_linspace(start, end, steps):
            delta = (end - start) / steps
            increments = list(range(0, steps)) * np.array([delta] * steps)
            return start + increments

        df = pd.core.series.Series(
            date_linspace(datetime.datetime.min, datetime.datetime.max, 11)
        )
        df = df.apply(
            lambda x: x - datetime.timedelta(microseconds=x.microsecond)
        ).apply(str)

        datetime_profile = DateTimeColumn(df[:-1].name)
        datetime_profile.update(df[:-1])

        self.assertEqual(datetime_profile.max, df.iloc[-2])

        datetime_profile.update(df)
        self.assertEqual(datetime_profile.max, df.iloc[-1])

        datetime_profile.update(pd.Series([np.nan, df.iloc[3]]))
        self.assertEqual(datetime_profile.max, df.iloc[-1])

        datetime_profile.update(df[1:2])  # only way to keep as df
        self.assertEqual(datetime_profile.max, df.iloc[-1])

    def test_date_time_detection(self):
        """
        Tests if get_datetime_params is able to detect the date time cols
        correctly
        :return:
        """
        date_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2013-03-5 15:43:30
            "%Y-%m-%dT%H:%M:%S",  # 2013-03-6T15:43:30
            "%Y-%m-%dT%H:%M:%S.%fZ",  # 2013-03-6T15:43:30.123456Z
            "%m/%d/%y %H:%M",  # 03/10/13 15:43
            "%m/%d/%Y %H:%M",  # 3/8/2013 15:43
            "%Y%m%dT%H%M%S",  # 2013036T154330
            "%H:%M:%S.%f",  # 05:46:30.258509
        ]

        self._test_datetime_detection_helper(date_formats)

    def test_date_time_detection_without_time(self):
        """
        Tests if get_datetime_params is able to detect the date cols correctly
        :return:
        """
        date_formats = [
            "%Y-%m-%d",  # 2013-03-7
            "%m/%d/%Y",  # 3/8/2013
            "%m/%d/%y",  # 03/10/13
            "%B %d, %Y",  # March 9, 2013
            "%b %d, %Y",  # Mar 11, 2013
            "%d%b%y",  # 12Mar13
            "%b-%d-%y",  # Mar-13-13
            "%m%d%Y",  # 03142013
        ]

        self._test_datetime_detection_helper(date_formats)

    def test_data_ratio(self):
        data = [2.5, 12.5, "2013-03-5 15:43:30", 5, "03/10/13 15:43", "Mar 11, 2013"]
        df = pd.Series(data).apply(str)

        profiler = DateTimeColumn(df.name)
        self.assertEqual(profiler.data_type_ratio, None)

        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 0.5)

        profiler.update(pd.Series([None, "10/20/13", "nan"]))
        self.assertEqual(profiler.data_type_ratio, 4 / 9.0)

    def test_profile(self):
        data = [2.5, 12.5, "2013-03-10 15:43:30", 5, "03/10/13 15:43", "Mar 11, 2013"]
        df = pd.Series(data).apply(str)
        profiler = DateTimeColumn(df.name)
        expected_profile = dict(
            min="03/10/13 15:43",
            max="Mar 11, 2013",
            histogram=None,
            format=[
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%y %H:%M",
                "%b %d, %Y",
            ],
            times=defaultdict(float, {"datetime": 1.0}),
        )
        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile["times"])

            # Validate the time in the datetime class has the expected time.
            profiler.update(df)
            expected = defaultdict(float, {"datetime": 1.0})
            self.assertEqual(expected, profiler.profile["times"])
            profile = profiler.profile
            self.assertCountEqual(expected_profile, profile)

            # Validate time in datetime class has expected time after second
            # update
            profiler.update(df)
            expected = defaultdict(float, {"datetime": 2.0})
            self.assertEqual(expected, profiler.profile["times"])
            self.assertEqual(expected_profile.pop("max"), profiler.profile["max"])

    def test_report(self):
        data = [2.5, 12.5, "2013-03-10 15:43:30", 5, "03/10/13 15:43", "Mar 11, 2013"]
        df = pd.Series(data).apply(str)
        profile = DateTimeColumn(df.name)

        report1 = profile.profile
        report2 = profile.report(remove_disabled_flag=False)
        report3 = profile.report(remove_disabled_flag=True)
        self.assertDictEqual(report1, report2)
        self.assertDictEqual(report1, report3)

    def test_warning_for_bad_dates(self):

        df = pd.Series(["03/10/2013 15:43"])

        profiler = DateTimeColumn(df.name)
        with warnings.catch_warnings(record=True) as w:
            profiler.update(df)
        self.assertEqual(len(w), 0)

        df = pd.Series(["03/10/13 15:43"])
        with self.assertWarns(RuntimeWarning) as r_warning:
            profiler.update(df)
        self.assertEqual(
            str(r_warning.warning),
            "Years provided were in two digit format. As a result, "
            "datetime assumes dates < 69 are for 2000s and above "
            "are for the 1990s. "
            "https://stackoverflow.com/questions/37766353/"
            "pandas-to-datetime-parsing-wrong-year",
        )

    def test_add(self):
        # unique format for the first profile
        data1 = [
            "2013-03-5 15:43:30",
            "2013-03-6T15:43:30",
            "2013-03-6T15:43:30.123456Z",
            "03/10/2013 15:43",
            "3/8/2013 15:43",
            "%2013036T154330",
            "05:46:30.258509",
        ]
        df = pd.Series(data1).apply(str)
        profile1 = DateTimeColumn(df.name)
        profile1.update(df)

        # unique format for second profile
        data2 = [
            2.5,
            12.5,
            "2013-03-10 15:23:20",
            5,
            "03/10/2013 15:23",
            "Mar 12, 2013",
        ]
        df = pd.Series(data2).apply(str)
        profile2 = DateTimeColumn(df.name)
        profile2.update(df)

        merged_profile = profile1 + profile2

        # checks for _dt_objs
        min_dt_obj = datetime.datetime.strptime("05:46:30.258509", "%H:%M:%S.%f")
        max_dt_obj = datetime.datetime.strptime("2013-03-12", "%Y-%m-%d")
        self.assertEqual(min_dt_obj, merged_profile._dt_obj_min)
        self.assertEqual(max_dt_obj, merged_profile._dt_obj_max)

        # checks for the proper max and min to be merged
        self.assertEqual("05:46:30.258509", merged_profile.min)
        self.assertEqual("Mar 12, 2013", merged_profile.max)

        # checks for date format merge
        self.assertCountEqual(
            [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%m/%d/%Y %H:%M",
                "%H:%M:%S.%f",
                "%b %d, %Y",
            ],
            merged_profile.date_formats,
        )

        # Checks for DateTimeColumn type for argument
        with self.assertRaises(TypeError) as exc:
            profile2 = "example_string"
            profile1 + profile2

        self.assertEqual(
            str(exc.exception),
            "Unsupported operand type(s) for +: "
            "'DateTimeColumn' and '{}'".format(profile2.__class__.__name__),
        )

    def test_null_add(self):

        # initialize the profiles
        dates = [None, "2014-12-18", "2015-07-21"]
        df = pd.Series(dates)
        df_nulls = df[:1]
        df_dates = df[1:]

        profile1 = DateTimeColumn(name="date")
        profile2 = DateTimeColumn(name="date")
        profile1.update(df_nulls)
        profile2.update(df_dates)

        # test when first profile has the nulls
        merged_profile = profile1 + profile2
        self.assertEqual("2014-12-18", merged_profile.min)
        self.assertEqual("2015-07-21", merged_profile.max)

        # test when second profile has the nulls
        merged_profile = profile2 + profile1
        self.assertEqual("2014-12-18", merged_profile.min)
        self.assertEqual("2015-07-21", merged_profile.max)

    def test_datetime_column_with_wrong_options(self):
        with self.assertRaisesRegex(
            ValueError,
            "DateTimeColumn parameter 'options' must be" " of type DateTimeOptions.",
        ):
            profiler = DateTimeColumn("Datetime", options="wrong_data_type")

    def test_day_suffixes(self):
        """
        Tests datetime examples with daytime suffixes.
        :return:
        """
        data = ["Mar 1st, 2020", "Feb 22nd, 2019", "October 23rd, 2018", "12thMar13"]
        df = pd.Series(data).apply(str)
        profiler = DateTimeColumn(df.name)
        profiler.update(df)
        self.assertEqual("Mar 1st, 2020", profiler.max)
        self.assertEqual("12thMar13", profiler.min)
        self.assertEqual(4, profiler.match_count)

    def test_diff(self):
        data1 = [None, "Mar 12, 2013", "2013-05-18", "2014-03-01"]
        df1 = pd.Series(data1).apply(str)
        profiler1 = DateTimeColumn(df1.name)
        profiler1.update(df1)

        data2 = [2.5, 12.5, "2013-03-10 15:43:30", 5, "03/10/14 15:43", "Mar 11, 2013"]
        df2 = pd.Series(data2).apply(str)
        profiler2 = DateTimeColumn(df2.name)
        profiler2.update(df2)

        expected_diff = {
            "min": "+1 days 08:16:30",
            "max": "-9 days 15:43:00",
            "format": [
                ["%Y-%m-%d"],
                ["%b %d, %Y"],
                ["%Y-%m-%d %H:%M:%S", "%m/%d/%y %H:%M"],
            ],
        }
        expected_format = expected_diff.pop("format")
        expected_unique1 = expected_format[0]
        expected_shared = expected_format[1]
        expected_unique2 = expected_format[2]

        diff = profiler1.diff(profiler2)
        format = diff.pop("format")
        unique1 = format[0]
        shared = format[1]
        unique2 = format[2]
        self.assertDictEqual(expected_diff, diff)
        self.assertEqual(set(expected_unique1), set(unique1))
        self.assertEqual(set(expected_shared), set(shared))
        self.assertEqual(set(expected_unique2), set(unique2))

        # Assert type error is properly called
        with self.assertRaises(TypeError) as exc:
            profiler1.diff("Inproper input")
        self.assertEqual(
            str(exc.exception),
            "Unsupported operand type(s) for diff: " "'DateTimeColumn' and 'str'",
        )

    def test_json_encode(self):
        profile = DateTimeColumn("0")

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "DateTimeColumn",
                "data": {
                    "name": "0",
                    "col_index": np.nan,
                    "sample_size": 0,
                    "metadata": dict(),
                    "times": defaultdict(),
                    "thread_safe": True,
                    "match_count": 0,
                    "date_formats": [],
                    "min": None,
                    "max": None,
                    "_dt_obj_min": None,
                    "_dt_obj_max": None,
                    "_DateTimeColumn__calculations": dict(),
                },
            }
        )

        self.assertEqual(serialized, expected)

    def test_json_encode_after_update(self):
        data = [2.5, 12.5, "2013-03-10 15:43:30", 5, "03/10/13 15:43", "Mar 11, 2013"]
        df = pd.Series(data).apply(str)
        profiler = DateTimeColumn("0")

        expected_date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%b %d, %Y",
            "%m/%d/%y %H:%M",
        ]
        with patch.object(
            profiler, "_combine_unique_sets", return_value=expected_date_formats
        ):
            with patch("time.time", return_value=0.0):
                profiler.update(df)

        serialized = json.dumps(profiler, cls=ProfileEncoder)

        expected = json.dumps(
            {
                "class": "DateTimeColumn",
                "data": {
                    "name": "0",
                    "col_index": np.nan,
                    "sample_size": 6,
                    "metadata": dict(),
                    "times": defaultdict(float, {"datetime": 0.0}),
                    "thread_safe": True,
                    "match_count": 3,
                    "date_formats": expected_date_formats,
                    "min": "03/10/13 15:43",
                    "max": "Mar 11, 2013",
                    "_dt_obj_min": "2013-03-10T15:43:00",
                    "_dt_obj_max": "2013-03-11T00:00:00",
                    "_DateTimeColumn__calculations": dict(),
                },
            }
        )

        self.assertEqual(serialized, expected)

    def test_json_encode_datetime(self):
        data = ["1209214"]
        df = pd.Series(data)
        profiler = DateTimeColumn("0")

        expected_date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%b %d, %Y",
            "%m/%d/%y %H:%M",
        ]
        with patch.object(
            profiler, "_combine_unique_sets", return_value=expected_date_formats
        ):
            with patch("time.time", return_value=0.0):
                profiler.update(df)

        serialized = json.dumps(profiler, cls=ProfileEncoder)

        expected = json.dumps(
            {
                "class": "DateTimeColumn",
                "data": {
                    "name": "0",
                    "col_index": np.nan,
                    "sample_size": 1,
                    "metadata": {},
                    "times": defaultdict(float, {"datetime": 0.0}),
                    "thread_safe": True,
                    "match_count": 1,
                    "date_formats": expected_date_formats,
                    "min": "1209214",
                    "max": "1209214",
                    "_dt_obj_min": "9214-01-20T00:00:00",
                    "_dt_obj_max": "9214-01-20T00:00:00",
                    "_DateTimeColumn__calculations": dict(),
                },
            }
        )

        self.assertEqual(serialized, expected)

    def test_json_decode(self):
        fake_profile_name = None
        expected_profile = DateTimeColumn(fake_profile_name)

        serialized = json.dumps(expected_profile, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_profile)

    def test_json_decode_after_update(self):
        fake_profile_name = "Fake profile name"

        data = [2.5, 12.5, "2013-03-10 15:43:30", 5, "03/10/13 15:43", "Mar 11, 2013"]
        df = pd.Series(data)

        expected_profile = DateTimeColumn(fake_profile_name)
        expected_profile.update(df)

        serialized = json.dumps(expected_profile, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_profile)

        expected_formats = [
            "%m/%d/%y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%B %d, %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y%m%dT%H%M%S",
            "%b %d, %Y",
        ]

        data_new = ["2012-02-10T15:43:30", "20120210T154300", "March 12, 2014"]
        df_new = pd.Series(data_new)

        # validating update after deserialization
        deserialized.update(df_new)

        assert deserialized._dt_obj_min == pd.Timestamp("2012-02-10 15:43:00")
        assert deserialized._dt_obj_max == pd.Timestamp("2014-03-12 00:00:00")

        assert set(deserialized.date_formats) == set(expected_formats)
