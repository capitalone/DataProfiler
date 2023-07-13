"""Validates that generator intakes DATAPROFILER_SEED properly."""
import unittest

import numpy as np

import dataprofiler.generator as generator


class TestGetRandomNumberGenerator(unittest.TestCase):
    """Validates get_random_number_generator() is properly working."""

    def setUp(self):
        """Declare variables to be used in the rest of the test suite."""
        self.rng = generator.get_random_number_generator()
        self.lower_bound_list = np.array([1, 2, 3, 4, 5])
        self.data_length = 10

    def test_rng_integer(self):
        """Check that the rng generates the expected single number."""
        sample_value = self.rng.integers(0, 100, 1)[0]
        self.assertEqual(sample_value, 85)

    def test_rng_integer_series(self):
        """Check that rng generates the expected series of integers."""
        sample_series = self.rng.integers(
            self.lower_bound_list, self.data_length
        ).tolist()
        self.assertListEqual(sample_series, [8, 7, 6, 5, 6])

    def test_rng_choice(self):
        """Check that the random number generator generates the expected series."""
        sample_series = self.rng.choice(
            list(self.lower_bound_list),
            (min(len(self.lower_bound_list), 5),),
            replace=False,
        ).tolist()
        self.assertListEqual(sample_series, [5, 3, 4, 1, 2])
