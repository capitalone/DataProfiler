from __future__ import print_function

import os
import six
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from dataprofiler.profilers import utils
from dataprofiler.tests.profilers import utils as test_utils
from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler, \
    BaseColumnPrimitiveTypeProfiler


class TestBaseColumnProfileClass(unittest.TestCase):
    def test_add_helper(self):
        """
        Tests the _add_helper function, specifically for merging two
        BaseColumnProfiler objects
        """
        with patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            profile1 = BaseColumnProfiler(name=0)
        profile1.sample_size = 2

        with patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            profile2 = BaseColumnProfiler(name=0)
        profile2.sample_size = 3

        with patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            merged_profile = BaseColumnProfiler(name=0)

        @BaseColumnProfiler._timeit
        def test_time(self):
            pass

        @BaseColumnProfiler._timeit
        def test_time2(self):
            pass

        # Dictionary starts empty
        self.assertDictEqual({}, profile1.times)

        # Array is popped twice per _timeit call start_time and end_time respectively
        time_array = [float(i) for i in range(10, -1, -1)]
        with patch('time.time', side_effect=lambda: time_array.pop()):
            # add one entry to profile1.times
            test_time(profile1)

            # add one entry that is the same to profile2.times
            test_time(profile2)
            # add unique entry to profile2.times
            test_time2(profile2)

            merged_profile._add_helper(profile1, profile2)

            # Ensure merge of times dict is done correctly
            expected3 = {"test_time": 2.0, "test_time2": 1.0}
            self.assertDictEqual(expected3, merged_profile.times)

        self.assertEqual(merged_profile.sample_size, 5)

        # Check for name alignment
        self.assertEqual(merged_profile.name, profile2.name)

        # Check for np.nan column index values for the merged profile
        self.assertTrue(np.isnan(merged_profile.col_index))

        # Check for same column index values
        profile1.col_index = 0
        profile2.col_index = 0
        merged_profile._add_helper(profile1, profile2)
        self.assertEqual(merged_profile.col_index, profile2.col_index)

        # Check for different column index but same column name
        with self.assertRaises(ValueError) as exc:
            profile1.col_index = 1
            profile1.name = "test"
            profile2.col_index = 2
            profile2.name = "test"
            merged_profile._add_helper(profile1, profile2)

        self.assertEqual(str(exc.exception),
                         "Column indexes unmatched: {} != {}"
                         .format(profile1.col_index, profile2.col_index))

        # Check for different column name but same column index
        with self.assertRaises(ValueError) as exc:
            profile1.col_index = 0
            profile1.name = "test1"
            profile2.col_index = 0
            profile2.name = "test2"
            merged_profile._add_helper(profile1, profile2)

        self.assertEqual(str(exc.exception),
                         "Column names unmatched: {} != {}"
                         .format(profile1.name, profile2.name))

    def test_time_it(self):
        with patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            profile1 = BaseColumnProfiler(name=0)

        @BaseColumnProfiler._timeit
        def test_time(self):
            pass

        @BaseColumnProfiler._timeit
        def test_time2(self):
            pass

        @BaseColumnProfiler._timeit(name="SetName")
        def test_time3(self):
            pass

        # Dictionary starts empty
        self.assertDictEqual({}, profile1.times)

        # Array is popped twice per _timeit call start_time and end_time respectively
        time_array = [12.0, 10.0, 9.0, 6.0, 5.0, 3.0, 2.0, 1.0]
        with patch('time.time', side_effect=lambda: time_array.pop()):
            # key and value populated correctly
            test_time(profile1)
            expected = {"test_time": 1.0}
            self.assertDictEqual(expected, profile1.times)

            # Different function times recorded correctly
            test_time2(profile1)
            expected["test_time2"] = 2.0
            self.assertDictEqual(expected, profile1.times)

            # Summation of first test_time and second test_time call
            test_time(profile1)
            expected["test_time"] = 4.0
            self.assertDictEqual(expected, profile1.times)

            # Can set name of key
            test_time3(profile1)
            self.assertTrue("SetName" in profile1.times)


class TestBaseColumnPrimitiveTypeProfileClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch.multiple(BaseColumnPrimitiveTypeProfiler, __abstractmethods__=set()):
            cls.b_profile = BaseColumnPrimitiveTypeProfiler(name=0)

    def test_cannot_instantiate(self):
        """showing we normally can't instantiate an abstract class"""
        with self.assertRaises(TypeError) as e:
            BaseColumnPrimitiveTypeProfiler()
        self.assertEqual(
            "Can't instantiate abstract class BaseColumnPrimitiveTypeProfiler "
            "with abstract methods _update_helper, profile, update",
            str(e.exception)
        )

    def test_combine_unqiue_sets(self):
        a = [1, 2, 3]
        b = [3, 1, 4, -1]
        c = utils._combine_unique_sets(a, b)
        six.assertCountEqual(self, [1, 2, 3, 4, -1], c)

    def test__init__(self):
        self.assertEqual(0, self.b_profile.name)
        self.assertTrue(np.nan is self.b_profile.col_index)
        self.assertEqual(0, self.b_profile.sample_size)
        self.assertEqual(0, self.b_profile.match_count)
        self.assertDictEqual(dict(), self.b_profile.metadata)

    def test_update_column_base_properties(self):
        extradata = dict(test1=1, test2=2)
        metadata = dict(sample_size=3, match_count=3)
        metadata.update(extradata)
        self.b_profile._update_column_base_properties(metadata)
        self.assertEqual(3, self.b_profile.sample_size)
        self.assertEqual(3, self.b_profile.match_count)
        self.assertDictEqual(metadata, self.b_profile.metadata)

    def test_update_match_are_abstract(self):
        six.assertCountEqual(
            self,
            {'_update_helper', 'update', 'profile'},
            BaseColumnPrimitiveTypeProfiler.__abstractmethods__
        )

    def test_add_helper(self):
        with patch.multiple(BaseColumnPrimitiveTypeProfiler, __abstractmethods__=set()):
            profile1 = BaseColumnPrimitiveTypeProfiler(name=0)

        with patch.multiple(BaseColumnPrimitiveTypeProfiler, __abstractmethods__=set()):
            profile2 = BaseColumnPrimitiveTypeProfiler(name=0)

        with patch.multiple(BaseColumnPrimitiveTypeProfiler, __abstractmethods__=set()):
            merged_profile = BaseColumnPrimitiveTypeProfiler(name=0)

        profile1.match_count = 2
        profile2.match_count = 3

        profile1.sample_size = 2
        profile2.sample_size = 3

        # Check if match_count was merged
        merged_profile._add_helper(profile1, profile2)
        self.assertEqual(merged_profile.match_count, 5)

        # Check if sample size merged
        self.assertEqual(merged_profile.sample_size, 5)

        # Check for np.nan column index values for the merged profile
        self.assertTrue(np.isnan(merged_profile.col_index))

        # Check for name alignment
        self.assertEqual(merged_profile.name, profile2.name)


if __name__ == '__main__':
    unittest.main()
test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_object_path(obj):
    return '.'.join(['data_profiler.profilers.profile_builder', obj.__name__])


class AbstractTestColumnProfiler(object):

    column_profiler = None

    def setUp(self):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        cls.input_file_path = os.path.join(
            test_root_path, 'data', 'csv/aws_honeypot_marx_geo.csv'
        )
        cls.aws_dataset = next(pd.read_csv(cls.input_file_path, chunksize=100))
        dataset = cls.aws_dataset["datetime"].dropna()
        cls.column_profile = cls.column_profiler(dataset)
        cls.profilers = cls.column_profile._profilers

    def test_profile_not_empty(self):
        self.assertIsNotNone(self.column_profile._profiles)

    def test_profilers_are_not_empty(self):
        self.assertIsNotNone(self.column_profile._profilers)

    def _create_profiler_mocks(self):
        profiler_mocks = []
        for profiler in self.profilers:
            mock_object = MagicMock(spec=profiler)
            mock_object.update.return_value = dict(test=1)
            mock_object.col_type = profiler.col_type
            mock_object.return_value.profile = {
                "data type": profiler.col_type,
                "statistics": dict()
            }
            mock_object.return_value.data_type_ratio = 1.0
            profiler_mocks.append(mock_object)
        self.column_profiler._profilers = profiler_mocks

        return profiler_mocks

    def _delete_profiler_mocks(self):
        self.column_profiler._profilers = self.profilers

    def test_created_profile(self):
        profiler_mocks = self._create_profiler_mocks()
        self.column_profiler(self.aws_dataset["datetime"])
        for profile, profiler_mock in \
                zip(self.column_profile._profiles.values(), profiler_mocks):
            self.assertIn(profile.__class__, self.profilers)
            self.assertEqual(1, profiler_mock.call_count)
        self._delete_profiler_mocks()

    def test_updated_profile(self):
        profiler_mocks = self._create_profiler_mocks()
        profile = self.column_profiler(self.aws_dataset["datetime"])
        profile.update_profile(self.aws_dataset["datetime"])

        for profiler_mock in profiler_mocks:
            self.assertEqual(1, profiler_mock.call_count)
        self._delete_profiler_mocks()

    def test_profile(self):
        self._create_profiler_mocks()
        profile = self.column_profiler(self.aws_dataset["datetime"])
        report = profile.profile

        six.assertCountEqual(
            self,
            self.profile_keys,
            report.keys()
        )
        self._delete_profiler_mocks()
