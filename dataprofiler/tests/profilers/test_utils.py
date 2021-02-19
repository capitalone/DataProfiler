import unittest

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
