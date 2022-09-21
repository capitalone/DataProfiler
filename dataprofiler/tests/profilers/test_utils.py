import os
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd

import dataprofiler as dp
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler
from dataprofiler.profilers import utils


class TestShuffleInChunks(unittest.TestCase):
    """
    Validates utils.shuffle_in_chunks is properly working.
    """

    def test_full_sample(self):
        """
        Check if can shuffle full sample.
        """
        sample = next(utils.shuffle_in_chunks(data_length=10, chunk_size=10))
        self.assertCountEqual(sample, list(range(10)))

    def test_even_chunk_sample(self):
        """
        Check if can shuffle sample where chunk size is evenly divisible.
        """
        sample_gen = utils.shuffle_in_chunks(data_length=12, chunk_size=3)

        all_values = set()
        num_chunks = 0
        for sample in sample_gen:
            self.assertFalse(all_values & set(sample))
            all_values = all_values | set(sample)
            num_chunks += 1
        self.assertEqual(num_chunks, 4)
        self.assertCountEqual(all_values, list(range(12)))

    def test_uneven_chunk_sample(self):
        """
        Check if can shuffle sample where chunk size is not evenly divisible.
        """
        sample_gen = utils.shuffle_in_chunks(data_length=100, chunk_size=7)

        all_values = set()
        num_chunks = 0
        for sample in sample_gen:
            self.assertFalse(all_values & set(sample))
            all_values = all_values | set(sample)
            num_chunks += 1
        self.assertEqual(num_chunks, 100 // 7 + 1)
        self.assertCountEqual(all_values, list(range(100)))

    def test_find_diff(self):
        """
        Checks to see if the find difference function is operating
        appropriately.
        """

        # Ensure lists and sets are handled appropriately
        self.assertEqual(
            [[], [3, 2], [2]], utils.find_diff_of_lists_and_sets([3, 2], [2, 3, 2])
        )
        self.assertEqual(
            [[1], [2, 3], [4]], utils.find_diff_of_lists_and_sets([1, 2, 3], [2, 3, 4])
        )
        self.assertEqual("unchanged", utils.find_diff_of_lists_and_sets({3, 2}, {2, 3}))
        self.assertEqual(
            [[1], [2, 3], [4]], utils.find_diff_of_lists_and_sets({1, 2, 3}, {2, 3, 4})
        )
        self.assertEqual("unchanged", utils.find_diff_of_lists_and_sets({2, 3}, [2, 3]))
        self.assertEqual(
            [[1], [2, 3], [4]], utils.find_diff_of_lists_and_sets([1, 2, 3], {2, 3, 4})
        )
        self.assertEqual(
            [None, {1, 2}], utils.find_diff_of_lists_and_sets(None, {1, 2})
        )
        self.assertEqual("unchanged", utils.find_diff_of_lists_and_sets(None, None))

        # Ensure ints and floats are handled appropriately
        self.assertEqual(1, utils.find_diff_of_numbers(5, 4))
        self.assertEqual(1.0, utils.find_diff_of_numbers(5.0, 4.0))
        self.assertEqual(1.0, utils.find_diff_of_numbers(5.0, 4))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(5.0, 5.0))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(5, 5.0))
        self.assertEqual([4, None], utils.find_diff_of_numbers(4, None))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(None, None))

        # Ensure strings are handled appropriately
        self.assertEqual(
            "unchanged", utils.find_diff_of_strings_and_bools("Hello", "Hello")
        )
        self.assertEqual(
            ["Hello", "team"], utils.find_diff_of_strings_and_bools("Hello", "team")
        )
        self.assertEqual("unchanged", utils.find_diff_of_strings_and_bools(None, None))

        # Ensure dates are handled appropriately
        a = datetime(2021, 6, 28)
        b = datetime(2021, 6, 27, 1)
        self.assertEqual("unchanged", utils.find_diff_of_dates(a, a))
        self.assertEqual("+23:00:00", utils.find_diff_of_dates(a, b))
        self.assertEqual("-23:00:00", utils.find_diff_of_dates(b, a))
        self.assertEqual(["06/28/21 00:00:00", None], utils.find_diff_of_dates(a, None))
        self.assertEqual("unchanged", utils.find_diff_of_dates(None, None))

        # Ensure that differencing dictionaries is handled appropriately
        dict1 = {
            "a": 0.25,
            "b": 0.0,
            "c": [1, 2],
            "d": datetime(2021, 6, 28),
            "e": "hi",
            "f": "hi2",
        }
        dict2 = {
            "a": 0.25,
            "b": 0.01,
            "c": [2, 3],
            "d": datetime(2021, 6, 27, 1),
            "e": "hihi",
            "g": 15,
        }
        expected_diff = {
            "a": "unchanged",
            "b": -0.01,
            "c": [[1], [2], [3]],
            "d": "+23:00:00",
            "e": ["hi", "hihi"],
            "f": ["hi2", None],
            "g": [None, 15],
        }
        self.assertDictEqual(expected_diff, utils.find_diff_of_dicts(dict1, dict2))

        dict1 = {
            "nested_key_one": {"fruit": ["apple", "banana", "orange"], "yes_no": False},
            "key_one": True,
            "attributes": {"value": 5, "id": 10},
            "last_nested_key": {"color": "blue", "weight": 35},
        }

        dict2 = {
            "nested_key_two": {"fruit": ["apple", "banana", "orange"], "yes_no": True},
            "key_one": True,
            "attributes": {"value": None, "id": 10},
            "last_nested_key": {"weight": 35, "height": "500"},
            "additional_key": "random_string",
        }

        expected_diff = {
            "nested_key_one": [
                {"fruit": ["apple", "banana", "orange"], "yes_no": False},
                None,
            ],
            "key_one": "unchanged",
            "attributes": {"value": [5, None], "id": "unchanged"},
            "last_nested_key": {
                "color": ["blue", None],
                "weight": "unchanged",
                "height": [None, "500"],
            },
            "nested_key_two": [
                None,
                {"fruit": ["apple", "banana", "orange"], "yes_no": True},
            ],
            "additional_key": [None, "random_string"],
        }

        self.assertDictEqual(expected_diff, utils.find_diff_of_dicts(dict1, dict2))

    def test_diff_of_dicts_with_diff_keys(self):
        dict1 = {"unique1": 1, "shared1": 2, "shared2": 3}
        dict2 = {"unique2": 5, "shared1": 2, "shared2": 6}

        expected = [
            {"unique1": 1},
            {"shared1": "unchanged", "shared2": -3},
            {"unique2": 5},
        ]

        # Assert difference is appropriate
        self.assertListEqual(
            expected, utils.find_diff_of_dicts_with_diff_keys(dict1, dict2)
        )

        # Assert empty dicts are unchanged
        self.assertEqual("unchanged", utils.find_diff_of_dicts_with_diff_keys({}, {}))

        # Assert all edge cases work
        a = datetime(2021, 6, 28)
        b = datetime(2021, 6, 27, 1)
        dict1 = {
            "unique1": 1,
            "shared1": "Hello",
            "shared2": a,
            "shared3": ["entry1"],
            "shared4": False,
        }
        dict2 = {
            "unique2": 5,
            "shared1": "Hi",
            "shared2": b,
            "shared3": ["entry1", "entry2", 3],
            "shared4": True,
        }
        expected = [
            {"unique1": 1},
            {
                "shared1": ["Hello", "Hi"],
                "shared2": "+23:00:00",
                "shared3": [[], ["entry1"], ["entry2", 3]],
                "shared4": [False, True],
            },
            {"unique2": 5},
        ]
        self.assertListEqual(
            expected, utils.find_diff_of_dicts_with_diff_keys(dict1, dict2)
        )

        dict1 = {
            "nested_key_one": {"fruit": ["apple", "banana", "orange"], "yes_no": False},
            "key_one": True,
            "attributes": {"value": 5, "id": 10},
            "last_nested_key": {"color": "blue", "weight": 35},
        }

        dict2 = {
            "nested_key_two": {"fruit": ["apple", "banana", "orange"], "yes_no": True},
            "key_one": True,
            "attributes": {"value": None, "id": 10},
            "last_nested_key": {"weight": 35, "height": "500"},
            "additional_key": "random_string",
        }

        expected = [
            {
                "nested_key_one": {
                    "fruit": ["apple", "banana", "orange"],
                    "yes_no": False,
                }
            },
            {
                "key_one": "unchanged",
                "attributes": [{}, {"value": [5, None], "id": "unchanged"}, {}],
                "last_nested_key": [
                    {"color": "blue"},
                    {"weight": "unchanged"},
                    {"height": "500"},
                ],
            },
            {
                "nested_key_two": {
                    "fruit": ["apple", "banana", "orange"],
                    "yes_no": True,
                },
                "additional_key": "random_string",
            },
        ]

        self.assertListEqual(
            expected, utils.find_diff_of_dicts_with_diff_keys(dict1, dict2)
        )

    def test_list_diff_with_nan(self):
        # when lists are same length
        list_1 = [np.nan, 1.5, 6.7]
        list_2 = [np.nan, 1.5, np.nan]
        diff_1 = utils.find_diff_of_lists_and_sets(list_1, list_2)
        expected_diff_1 = [[6.7], [np.nan, 1.5], [np.nan]]

        for x, y in zip(diff_1, expected_diff_1):
            comparison_1 = ((x == y) | (np.isnan(x) & np.isnan(y))).all()
            self.assertEqual(True, comparison_1)

        # when lists aren't the same length
        list_3 = [np.nan, 1.5, 6.7, np.nan, np.nan, np.nan]
        list_4 = [4.2, 1.5, np.nan]
        diff_2 = utils.find_diff_of_lists_and_sets(list_3, list_4)
        expected_diff_2 = [[6.7, np.nan, np.nan, np.nan], [np.nan, 1.5], [4.2]]

        for x, y in zip(diff_2, expected_diff_2):
            comparison_2 = ((x == y) | (np.isnan(x) & np.isnan(y))).all()
            self.assertEqual(True, comparison_2)

        list_5 = [np.nan, np.nan]
        list_6 = [np.nan]
        diff_3 = utils.find_diff_of_lists_and_sets(list_5, list_6)
        expected_diff_3 = [[np.nan], [np.nan], []]

        for x, y in zip(diff_3, expected_diff_3):
            comparison_3 = ((x == y) | (np.isnan(x) & np.isnan(y))).all()
            self.assertEqual(True, comparison_3)

        list_7 = [np.nan, 3]
        list_8 = [np.nan, 3]
        diff_4 = utils.find_diff_of_lists_and_sets(list_7, list_8)
        expected_diff_4 = "unchanged"

        self.assertEqual(diff_4, expected_diff_4)

    def test_find_diff_of_matrices(self):
        import numpy as np

        matrix1 = [[1, None, 3], [4, 5, 6], [7, 8, 9]]
        matrix2 = [[11, np.nan, 0], [1, 5, 2], [np.nan, 20, 1]]

        # Check matrix subtraction of same size matrices
        expected_matrix = [[-10.0, np.nan, 3.0], [3.0, 0.0, 4.0], [np.nan, -12.0, 8.0]]
        diff_matrix = utils.find_diff_of_matrices(matrix1, matrix2)
        comparison = (
            (expected_matrix == diff_matrix)
            | (np.isnan(expected_matrix) & np.isnan(diff_matrix))
        ).all()
        self.assertEqual(True, comparison)

        # Check matrix subtraction of same exact matrices
        self.assertEqual("unchanged", utils.find_diff_of_matrices(matrix1, matrix1))
        # Check matrix subtraction with different sized matrices
        matrix1 = [[1, 2], [1, 2]]
        self.assertIsNone(utils.find_diff_of_matrices(matrix1, matrix2))

        # Check matrix with none
        self.assertIsNone(utils.find_diff_of_matrices(matrix1, None))

    def test_get_memory_size(self):
        """
        Checks to see if the get memory size function is operating appropriately.
        """
        # wrong unit input
        with self.assertRaisesRegex(
            ValueError,
            "Currently only supports the memory size unit " "in \['B', 'K', 'M', 'G'\]",
        ):
            utils.get_memory_size([], unit="wrong_unit")

        # test with different data sizes
        self.assertEqual(0, utils.get_memory_size([]))
        self.assertEqual(
            33 / 1024**2, utils.get_memory_size(["This is test, a Test sentence.!!!"])
        )
        self.assertEqual(
            33 / 1024**2,
            utils.get_memory_size(["This is test,", " a Test sentence.!!!"]),
        )
        self.assertEqual(
            33 / 1024**3,
            utils.get_memory_size(["This is test, a Test sentence.!!!"], unit="G"),
        )


@mock.patch("dataprofiler.profilers.profile_builder.DataLabeler", spec=BaseDataLabeler)
class TestProfileDistributedMerge(unittest.TestCase):
    """
    Validates utils.merge_profile_list is properly working.
    """

    @staticmethod
    def _setup_data_labeler_mock(mock_instance):
        mock_DataLabeler = mock_instance.return_value
        mock_DataLabeler.label_mapping = {"a": 0, "b": 1}
        mock_DataLabeler.reverse_label_mapping = {0: "a", 1: "b"}
        mock_DataLabeler.model.num_labels = 2
        mock_DataLabeler.model.requires_zero_mapping = False

        def mock_predict(data, *args, **kwargs):
            len_data = len(data)
            output = [[1, 0], [0, 1]] * (len_data // 2)
            if len_data % 2:
                output += [[1, 0]]
            conf = np.array(output)
            if mock_DataLabeler.model.requires_zero_mapping:
                conf = np.concatenate([[[0]] * len_data, conf], axis=1)
            pred = np.argmax(conf, axis=1)
            return {"pred": pred, "conf": conf}

        mock_DataLabeler.predict.side_effect = mock_predict

    def test_merge_profile_list(self, mock_data_labeler, *mocks):
        """
        A top-level function which takes in a list of profile objects, merges
            all the profiles together into one profile, and returns the single
            merged profile as the return value.

            The labeler object is removed prior to merge and added back to the
            single profile object.
        """
        self._setup_data_labeler_mock(mock_data_labeler)

        data = pd.DataFrame([1, 2, 3, 4, 5, 60, 1])
        profile_one = dp.Profiler(data[:2])
        profile_two = dp.Profiler(data[2:])

        list_of_profiles = [profile_one, profile_two]
        single_profile = utils.merge_profile_list(list_of_profiles=list_of_profiles)
        single_report = single_profile.report()

        self.assertEqual(1, len(single_report["data_stats"]))
        self.assertEqual(1, single_report["global_stats"]["column_count"])
        self.assertEqual(7, single_report["global_stats"]["row_count"])

        self.assertEqual("int", single_report["data_stats"][0]["data_type"])

        self.assertEqual(1, single_report["data_stats"][0]["statistics"]["min"])
        self.assertEqual(60.0, single_report["data_stats"][0]["statistics"]["max"])
        self.assertEqual(
            2.9764999999999997, single_report["data_stats"][0]["statistics"]["median"]
        )
        self.assertEqual(
            10.857142857142858, single_report["data_stats"][0]["statistics"]["mean"]
        )

    def test_odd_merge_profile_list(self, mock_data_labeler, *mocks):
        """
        A top-level function which takes in a list of profile objects, merges
            all the profiles together into one profile, and returns the single
            merged profile as the return value.

            The labeler object is removed prior to merge and added back to the
            single profile object.
        """
        self._setup_data_labeler_mock(mock_data_labeler)

        data = pd.DataFrame([1, 2, 3, 4, 5, 60, 1])
        profile_one = dp.Profiler(data[:2])
        profile_two = dp.Profiler(data[2:])
        profile_three = dp.Profiler(data[2:])

        list_of_profiles = [profile_one, profile_two, profile_three]
        single_profile = utils.merge_profile_list(list_of_profiles=list_of_profiles)
        single_report = single_profile.report()

        self.assertEqual(1, len(single_report["data_stats"]))
        self.assertEqual(1, single_report["global_stats"]["column_count"])
        self.assertEqual(12, single_report["global_stats"]["row_count"])

        self.assertEqual("int", single_report["data_stats"][0]["data_type"])

        self.assertEqual(1, single_report["data_stats"][0]["statistics"]["min"])
        self.assertEqual(60.0, single_report["data_stats"][0]["statistics"]["max"])
