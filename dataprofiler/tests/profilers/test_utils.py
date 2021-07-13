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
