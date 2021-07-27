import unittest

from datetime import datetime
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
        self.assertEqual("unchanged",
                         utils.find_diff_of_lists_and_sets([3, 2], [2, 3, 2]))
        self.assertEqual([[1], [2, 3], [4]],
                         utils.find_diff_of_lists_and_sets([1, 2, 3], [2, 3, 4]))
        self.assertEqual("unchanged",
                         utils.find_diff_of_lists_and_sets({3, 2}, {2, 3}))
        self.assertEqual([[1], [2, 3], [4]],
                         utils.find_diff_of_lists_and_sets({1, 2, 3}, {2, 3, 4}))
        self.assertEqual("unchanged",
                         utils.find_diff_of_lists_and_sets({2, 3}, [2, 3]))
        self.assertEqual([[1], [2, 3], [4]],
                         utils.find_diff_of_lists_and_sets([1, 2, 3], {2, 3, 4}))
        self.assertEqual([None, {1, 2}],
                         utils.find_diff_of_lists_and_sets(None, {1, 2}))
        self.assertEqual("unchanged",
                         utils.find_diff_of_lists_and_sets(None, None))

        # Ensure ints and floats are handled appropriately
        self.assertEqual(1, utils.find_diff_of_numbers(5, 4))
        self.assertEqual(1.0, utils.find_diff_of_numbers(5.0, 4.0))
        self.assertEqual(1.0, utils.find_diff_of_numbers(5.0, 4))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(5.0, 5.0))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(5, 5.0))
        self.assertEqual([4, None],
                         utils.find_diff_of_numbers(4, None))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(None, None))

        # Ensure strings are handled appropriately
        self.assertEqual("unchanged",
                         utils.find_diff_of_strings_and_bools("Hello", "Hello"))
        self.assertEqual(["Hello", "team"],
                         utils.find_diff_of_strings_and_bools("Hello", "team"))
        self.assertEqual("unchanged",
                         utils.find_diff_of_strings_and_bools(None, None))

        # Ensure dates are handled appropriately
        a = datetime(2021, 6, 28)
        b = datetime(2021, 6, 27, 1)
        self.assertEqual("unchanged", utils.find_diff_of_dates(a, a))
        self.assertEqual("+23:00:00", utils.find_diff_of_dates(a, b))
        self.assertEqual("-23:00:00", utils.find_diff_of_dates(b, a))
        self.assertEqual(["06/28/21 00:00:00", None], utils.find_diff_of_dates(a, None))
        self.assertEqual("unchanged", utils.find_diff_of_numbers(None, None))

        # Ensure that differencing dictionaries is handled appropriately
        dict1 = {
            "a": 0.25,
            "b": 0.0,
            "c": [1, 2],
            "d": datetime(2021, 6, 28),
            "e": "hi",
            "f": "hi2"
        }
        dict2 = {
            "a": 0.25,
            "b": 0.01,
            "c": [2, 3],
            "d": datetime(2021, 6, 27, 1),
            "e": "hihi",
            "g": 15
        }
        expected_diff = {
            "a": "unchanged",
            "b": -0.01,
            "c": [[1], [2], [3]],
            "d": "+23:00:00",
            "e": ["hi", "hihi"],
            "f": ["hi2", None],
            "g": [None, 15]
        }
        self.assertDictEqual(expected_diff, utils.find_diff_of_dicts(dict1, dict2))

    def test_diff_of_dicts_with_diff_keys(self):
        dict1 = {"unique1": 1, "shared1": 2, "shared2": 3}
        dict2 = {"unique2": 5, "shared1": 2, "shared2": 6}

        expected = [{'unique1': 1}, 
                    {'shared1': 'unchanged', 'shared2': -3}, 
                    {'unique2': 5}]
        
        # Assert difference is appropriate
        self.assertListEqual(expected,
                         utils.find_diff_of_dicts_with_diff_keys(dict1, dict2))

        # Assert empty dicts are unchanged
        self.assertEqual("unchanged",
                         utils.find_diff_of_dicts_with_diff_keys({},{}))
        
        # Assert all edge cases work
        a = datetime(2021, 6, 28)
        b = datetime(2021, 6, 27, 1)
        dict1 = {"unique1": 1, 
                 "shared1": "Hello", 
                 "shared2": a, 
                 "shared3": ["entry1"], 
                 "shared4": False}
        dict2 = {"unique2": 5, 
                 "shared1": "Hi", 
                 "shared2": b, 
                 "shared3": ["entry1", "entry2", 3], 
                 "shared4": True}
        expected = [{'unique1': 1}, 
                    {'shared1': ['Hello', 'Hi'], 
                     'shared2': '+23:00:00', 
                     'shared3': [[], ['entry1'], ['entry2', 3]], 
                     'shared4': [False, True]}, 
                    {'unique2': 5}]
        self.assertListEqual(expected, 
                             utils.find_diff_of_dicts_with_diff_keys(dict1, 
                                                                     dict2))
       
    def test_find_diff_of_matrices(self):
        import numpy as np
        matrix1 = [[1, None, 3],
                   [4, 5, 6],
                   [7, 8, 9]]
        matrix2 = [[11, np.nan, 0],
                   [1, 5, 2],
                   [np.nan, 20, 1]]
        
        # Check matrix subtraction of same size matrices
        expected_matrix = [[-10., np.nan, 3.],
                           [ 3., 0., 4.],
                           [np.nan, -12., 8.]]
        diff_matrix = utils.find_diff_of_matrices(matrix1, matrix2)
        comparison = ((expected_matrix == diff_matrix) | 
                      (np.isnan(expected_matrix) & np.isnan(diff_matrix))).all()
        self.assertEqual(True, comparison)

        # Check matrix subtraction of same exact matrices
        self.assertEqual("unchanged", utils.find_diff_of_matrices(matrix1, 
                                                                  matrix1))
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
        with self.assertRaisesRegex(ValueError,
                                    "Currently only supports the memory size unit "
                                    "in \['B', 'K', 'M', 'G'\]"):
            utils.get_memory_size([], unit="wrong_unit")

        # test with different data sizes
        self.assertEqual(0, utils.get_memory_size([]))
        self.assertEqual(33 / 1024 ** 2,
            utils.get_memory_size(["This is test, a Test sentence.!!!"]))
        self.assertEqual(33 / 1024 ** 2,
            utils.get_memory_size(["This is test,", " a Test sentence.!!!"]))
        self.assertEqual(33 / 1024 ** 3,
            utils.get_memory_size(["This is test, a Test sentence.!!!"], unit='G'))
