import unittest
from unittest import mock
from collections import defaultdict

import pandas as pd

from .. import test_utils

from dataprofiler.profilers import OrderColumn

# This is taken from: https://github.com/rlworkgroup/dowel/pull/36/files
# undo when cpython#4800 is merged.
unittest.case._AssertWarnsContext.__enter__ = test_utils.patched_assert_warns


class TestOrderColumn(unittest.TestCase):

    @staticmethod
    def _update_order(data):
        df = pd.Series(data).apply(str)

        profiler = OrderColumn(df.name)
        profiler.update(df)

        return profiler.order

    def test_base_case(self):
        data = pd.Series([], dtype=object)
        profiler = OrderColumn(data.name)
        profiler.update(data)

        self.assertEqual(profiler.sample_size, 0)
        self.assertIsNone(profiler.order)

    def test_descending(self):
        data = ['za', 'z', 'c', 'a']
        order = self._update_order(data)
        self.assertEqual(order, 'descending')

        data = [5, 3, 2]
        order = self._update_order(data)
        self.assertEqual(order, 'descending')

    def test_ascending(self):
        data = ['a', 'b', 'z', 'za']
        order = self._update_order(data)
        self.assertEqual(order, 'ascending')

        data = [2, 3, 11]
        order = self._update_order(data)
        self.assertEqual(order, 'ascending')

    def test_constant_value(self):
        data = ['a']
        order = self._update_order(data)
        self.assertEqual(order, 'constant value')

        data = ['a', 'a', 'a', 'a', 'a']
        order = self._update_order(data)
        self.assertEqual(order, 'constant value')

    def test_random(self):
        data = ['a', 'b', 'ab']
        order = self._update_order(data)
        self.assertEqual(order, 'random')

        data = [1, 11, 4]
        order = self._update_order(data)
        self.assertEqual(order, 'random')

    def test_batch_updates(self):
        data = ['a', 'a', 'a']
        df = pd.Series(data)
        profiler = OrderColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.order, 'constant value')

        data = ['a', 'b', 'c']
        df = pd.Series(data)
        profiler.update(df)
        self.assertEqual(profiler.order, 'ascending')

        # previous was ascending, should stay ascending bc now receiving const
        data = ['c', 'c', 'c']
        df = pd.Series(data)
        profiler.update(df)
        self.assertEqual(profiler.order, 'ascending')

        # previous was ascending, should be random now receiving descending
        data = ['c', 'b', 'a']
        df = pd.Series(data)
        profiler.update(df)
        self.assertEqual(profiler.order, 'random')

    def test_profile(self):
        data = [1]
        df = pd.Series(data).apply(str)

        profiler = OrderColumn(df.name)

        expected_profile = dict(
            order='constant value',
            times={'order' : 2.0}
        )
        time_array = [float(x) for x in range(4, 0, -1)]
        with mock.patch('time.time', side_effect = lambda: time_array.pop()):
            profiler.update(df)
            profile = profiler.profile

            # key and value populated correctly
            self.assertDictEqual(expected_profile, profile)

    def test_profile_merge(self):
        data = [1, 2, 3, 4, 5, 6]
        df = pd.Series(data).apply(str)
        profiler = OrderColumn("placeholder_name")
        profiler.update(df)

        data2 = [7, 8, 9, 10]
        df2 = pd.Series(data2).apply(str)
        profiler2 = OrderColumn("placeholder_name")
        profiler2.update(df2)

        data3 = [2, 3, 4]
        df3 = pd.Series(data3).apply(str)
        profiler3 = OrderColumn("placeholder_name")
        profiler3.update(df3)

        data4 = [3, 3, 3, 3]
        df4 = pd.Series(data4).apply(str)
        profiler4 = OrderColumn("placeholder_name")
        profiler4.update(df4)

        data5 = [4, 2, 3, 1, 5]
        df5 = pd.Series(data5).apply(str)
        profiler5 = OrderColumn("placeholder_name")
        profiler5.update(df5)

        data6 = [10, 9, 8, 7]
        df6 = pd.Series(data6).apply(str)
        profiler6 = OrderColumn("placeholder_name")
        profiler6.update(df6)

        data7 = [3, 3, 3]
        df7 = pd.Series(data7).apply(str)
        profiler7 = OrderColumn("placeholder_name")
        profiler7.update(df7)

        data8 = [7, 7, 7, 7, 7, 7, 7]
        df8 = pd.Series(data8).apply(str)
        profiler8 = OrderColumn("placeholder_name")
        profiler8.update(df8)

        data9 = [7, 6, 5, 4, 3]
        df9 = pd.Series(data9).apply(str)
        profiler9 = OrderColumn("placeholder_name")
        profiler9.update(df9)

        data10 = [1, 5, 6]
        df10 = pd.Series(data10).apply(str)
        profiler10 = OrderColumn("placeholder_name")
        profiler10.update(df10)
        profiler10._piecewise = True

        data11 = pd.Series([], dtype=object)
        df11 = pd.Series(data11).apply(str)
        profiler11 = OrderColumn("placeholder_name")
        profiler11.update(df11)

        #Ascending + Ascending, non-intersecting, non-piecewise
        profiler_merged = profiler + profiler2
        self.assertEqual(profiler_merged.order, "ascending")
        self.assertEqual(profiler_merged._last_value, 10)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 1)

        #Ascending + Ascending, intersecting, non-piecewise
        profiler_merged = profiler + profiler3
        self.assertEqual(profiler_merged.order, "random")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 1)

        #Ascending + Ascending, intersecting, both piecewise
        profiler_merged = profiler + profiler2
        profiler_merged2 = profiler + profiler2
        profiler_merged = profiler_merged + profiler_merged2
        self.assertEqual(profiler_merged.order, "ascending")
        self.assertEqual(profiler_merged._last_value, 10)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Ascending, intersecting, Enveloping piecewise
        profiler_merged = profiler3 + profiler10
        self.assertEqual(profiler_merged.order, "ascending")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Ascending, non-intersecting, both piecewise
        profiler_test1 = profiler
        profiler_test1._piecewise = True
        profiler_test2 = profiler2
        profiler_test2._piecewise = True
        profiler_merged = profiler_test1 + profiler2
        self.assertEqual(profiler_merged.order, "ascending")
        self.assertEqual(profiler_merged._last_value, 10)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Ascending, intersecting (on top of each other), Not piecewise
        profiler._piecewise = False
        profiler_merged = profiler + profiler
        self.assertEqual(profiler_merged.order, "random")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Constant, intersecting, Not piecewise
        profiler_merged = profiler + profiler4
        self.assertEqual(profiler_merged.order, "random")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Constant, intersecting, Ascending is piecewise
        profiler._piecewise = True
        profiler_merged = profiler + profiler4
        self.assertEqual(profiler_merged.order, "ascending")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Random
        profiler._piecewise = False
        profiler_merged = profiler + profiler5
        self.assertEqual(profiler_merged.order, "random")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 1)

        # Ascending + Descending
        profiler._piecewise = False
        profiler_merged = profiler + profiler6
        self.assertEqual(profiler_merged.order, "random")
        self.assertEqual(profiler_merged._last_value, 10)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 1)

        # Constant + Constant (same constant value)
        profiler_merged = profiler4 + profiler7
        self.assertEqual(profiler_merged.order, "constant value")
        self.assertEqual(profiler_merged._last_value, 3)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 3)

        # Constant + Constant (different constant value)
        profiler_merged = profiler4 + profiler8
        self.assertEqual(profiler_merged.order, "constant value")
        self.assertEqual(profiler_merged._last_value, 7)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 3)

        # Descending + Descending, non intersecting (except on edge), non piecewise
        profiler_merged = profiler6 + profiler9
        self.assertEqual(profiler_merged.order, "descending")
        self.assertEqual(profiler_merged._last_value, 3)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 10)

        # Descending + Constant, non intersecting (except on edge), non piecewise
        profiler_merged = profiler8 + profiler9
        self.assertEqual(profiler_merged.order, "descending")
        self.assertEqual(profiler_merged._last_value, 3)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 7)

        # Descending + Constant, non intersecting (except on edge), non piecewise
        profiler_merged = profiler6 + profiler8 + profiler9
        self.assertEqual(profiler_merged.order, "descending")
        self.assertEqual(profiler_merged._last_value, 3)
        self.assertEqual(profiler_merged._piecewise, True)
        self.assertEqual(profiler_merged._first_value, 10)

        # Ascending + Empty = Ascending
        profiler_merged = profiler + profiler11
        self.assertEqual(profiler_merged.order, "ascending")
        self.assertEqual(profiler_merged._last_value, 6)
        self.assertEqual(profiler_merged._piecewise, False)
        self.assertEqual(profiler_merged._first_value, 1)

    def test_merge_timing(self):
        profiler1 = OrderColumn("placeholder_name")
        profiler2 = OrderColumn("placeholder_name")
            
        profiler1.times = dict(order=2.0)
        profiler2.times = dict(order=3.0)

        time_array = [float(i) for i in range(2, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            profiler3 = profiler1 + profiler2

            # __add__() call adds 1 so expected is 6
            expected_times = defaultdict(float, {'order': 6.0})
            self.assertDictEqual(expected_times, profiler3.profile['times'])

    @mock.patch('dataprofiler.profilers.OrderColumn._get_data_order')
    def test_random_order_prevents_update_from_occuring(self,
                                                        mock_get_data_order):
        mock_get_data_order.return_value = ["random", 1, 2]
        data = ['a', 'b', 'ab']
        df = pd.Series(data).apply(str)

        # Assert the order is random
        profiler = OrderColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.order, 'random')

        # Assert that the update wasn't called again
        profiler.update(data)
        mock_get_data_order.assert_called_once()
        
    def test_order_column_with_wrong_options(self):
        with self.assertRaisesRegex(ValueError,
                                   "OrderColumn parameter 'options' must be of"
                                   " type OrderOptions."):
            profiler = OrderColumn("Order", options="wrong_data_type")
