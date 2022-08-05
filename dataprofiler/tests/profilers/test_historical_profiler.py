from __future__ import print_function

import os
import unittest
from io import BytesIO
from unittest import mock

import pandas as pd

import dataprofiler as dp
from dataprofiler.profilers.profiler_options import ProfilerOptions

from . import utils as test_utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def setup_save_mock_open(mock_open):
    mock_file = BytesIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


class TestHistoricalProfiler(unittest.TestCase):
    @classmethod
    def setUp(cls):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        test_utils.set_seed(seed=0)

        cls.input_file_path = os.path.join(test_root_path, "data", "csv/ny_climate.csv")
        ny_climate_df = pd.read_csv(cls.input_file_path)
        ny_climate_df.sort_values(by="YEAR", axis=0, inplace=True)
        years = ny_climate_df["YEAR"].unique().tolist()
        years.reverse()
        individual_dfs = []
        for year in years:
            current_year_df = ny_climate_df.loc[ny_climate_df["YEAR"] == year]
            current_year_df = current_year_df.drop("YEAR", axis=1)
            individual_dfs.append(current_year_df)
        cls.ny_climate_datasets = individual_dfs
        profiler_options = ProfilerOptions()
        profiler_options.set({"data_labeler.is_enabled": False})
        with test_utils.mock_timeit():
            ny_climate_profiles = []
            for df in cls.ny_climate_datasets:
                ny_climate_profiles.append(
                    dp.StructuredProfiler(df, len(df), options=profiler_options)
                )
            cls.ny_climate_profiles = ny_climate_profiles

    def test_init_fail(self, *mocks):
        with self.assertRaisesRegex(
            ValueError,
            "'profiles' is 'None', expected a list containing type `Profiler`",
        ):
            hp = dp.HistoricalProfiler(None)
        emptyProfileObjs = []
        with self.assertRaisesRegex(
            ValueError, "'profiles' is empty. At least one Profiler object is required"
        ):
            hp = dp.HistoricalProfiler(emptyProfileObjs)

        with self.assertRaisesRegex(
            ValueError, "`profiles` has profile not of type `StructuredProfiler`."
        ):
            hp = dp.HistoricalProfiler(["Not a Profile"])

    def test_init_list_profiles(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)

        self.assertEqual(8, hp.length)
        self.assertListEqual(
            [5, 12, 12, 10, 12, 12, 12, 12],
            hp.historical_profile["global_stats"]["samples_used"],
        )

    def test_length(self, *mocks):
        with test_utils.mock_timeit():
            hp1 = dp.HistoricalProfiler(self.ny_climate_profiles)
            hp2 = dp.HistoricalProfiler(self.ny_climate_profiles[2:])

        self.assertEqual(len(self.ny_climate_profiles), len(hp1))
        self.assertEqual((len(self.ny_climate_profiles) - 2), len(hp2))

    def test_append_profile(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles[1:])

        self.assertEqual(7, len(hp))

        hp.append(self.ny_climate_profiles[0])

        self.assertEqual(8, len(hp))
        self.assertEqual(5, hp.historical_profile["global_stats"]["samples_used"][0])

    def test_getting_reports(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)

        most_recent = hp.get_most_recent_profile_report()
        self.assertEqual(
            hp.historical_profile["global_stats"]["samples_used"][0],
            most_recent["global_stats"]["samples_used"],
        )
        self.assertListEqual(
            hp.historical_profile["data_stats"][0]["samples"][0],
            most_recent["data_stats"][0]["samples"],
        )

        oldest = hp.get_oldest_profile_report()
        self.assertEqual(
            hp.historical_profile["global_stats"]["samples_used"][len(hp) - 1],
            oldest["global_stats"]["samples_used"],
        )
        self.assertListEqual(
            hp.historical_profile["data_stats"][0]["samples"][len(hp) - 1],
            oldest["data_stats"][0]["samples"],
        )

        index_1 = hp.get_profile_report_by_index(1)
        self.assertEqual(
            hp.historical_profile["global_stats"]["samples_used"][1],
            index_1["global_stats"]["samples_used"],
        )
        self.assertListEqual(
            hp.historical_profile["data_stats"][0]["samples"][1],
            index_1["data_stats"][0]["samples"],
        )

    def test_get_full_report(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)

        self.assertDictEqual(hp.historical_profile, hp.report())

    def test_update_report_by_index(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)

        self.assertListEqual(
            [5, 12, 12, 10, 12, 12, 12, 12],
            hp.historical_profile["global_stats"]["samples_used"],
        )

        hp.update_profile_report_at_index(self.ny_climate_profiles[1], 0)

        self.assertListEqual(
            [12, 12, 12, 10, 12, 12, 12, 12],
            hp.historical_profile["global_stats"]["samples_used"],
        )

    def test_delete_report_by_index(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)

        self.assertEqual(8, len(hp))
        self.assertListEqual(
            [5, 12, 12, 10, 12, 12, 12, 12],
            hp.historical_profile["global_stats"]["samples_used"],
        )

        hp.delete_profile_report_at_index(0)

        self.assertEqual(7, len(hp))
        self.assertListEqual(
            [12, 12, 10, 12, 12, 12, 12],
            hp.historical_profile["global_stats"]["samples_used"],
        )

    def test_consecutive_diffs_report(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)
            consecutive_diff_report = hp.get_consecutive_diffs_report()

        self.assertEqual(
            [-7, "unchanged", 2, -2, "unchanged", "unchanged", "unchanged"],
            consecutive_diff_report["global_stats"]["samples_used"],
        )

    def test_diff_min_max_report(self, *mocks):
        with test_utils.mock_timeit():
            hp = dp.HistoricalProfiler(self.ny_climate_profiles)
            diff_min_max_report = hp.get_diff_min_and_max_report()

        self.assertEqual((-7, 2), diff_min_max_report["global_stats"]["samples_used"])
        self.assertEqual(
            hp.historical_profile["data_stats"][0]["column_name"][0],
            diff_min_max_report["data_stats"][0]["column_name"],
        )

    def test_appending_beyond_max_length(self, *mocks):
        with test_utils.mock_timeit():
            opts = dp.HistoricalProfiler.historical_profiler_options()
            opts["max_length"] = 7
            hp = dp.HistoricalProfiler(self.ny_climate_profiles[1:], options=opts)

        self.assertEqual(7, len(hp))

        hp.append(self.ny_climate_profiles[0])

        self.assertEqual(7, len(hp))
        self.assertEqual(5, hp.historical_profile["global_stats"]["samples_used"][0])


if __name__ == "__main__":
    unittest.main()
