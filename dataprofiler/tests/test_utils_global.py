"""Validates that generator intakes DATAPROFILER_SEED properly."""
import os
import unittest

import numpy as np
from numpy.random import PCG64

import dataprofiler.utils_global as utils_global


class TestOriginalFunction(unittest.TestCase):
    """Validates get_random_number_generator() is properly working."""

    @classmethod
    def setUp(self):
        """Declare variables to be used in the rest of the test suite."""
        self.rng = utils_global.get_random_number_generator()
        self.lower_bound_list = np.array([1, 2, 3, 4, 5])
        self.data_length = 10

    @unittest.mock.patch("os.environ")
    def test_return_random_value(self, *mocks):
        """If DATAPROFILER_SEED not set, test that rng returns different random value."""
        rng = utils_global.get_random_number_generator()
        sample_value = rng.integers(0, 1e6, 1)[0]
        self.assertNotEqual(sample_value, 850624)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_return_default_rng_with_settings_seed(self):
        """If DATAPROFILER_SEED not set, test that rng uses seed = settings._seed."""
        rng = utils_global.get_random_number_generator()
        actual_value = rng.integers(0, 100, 1)[0]
        expected_value_generator = np.random.default_rng(123)
        expected_value = expected_value_generator.integers(0, 100, 1)[0]
        self.assertEqual(actual_value, expected_value)

    @unittest.mock.patch.dict(os.environ, {"DATAPROFILER_SEED": "0"}, clear=True)
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_dataprofiler_seed_true_settings_seed_false(self):
        """Verify that we get the expected result when DATAPROFILER_SEED in os.environ and settings._seed!=None."""
        rng = utils_global.get_random_number_generator()
        actual_value = rng.integers(0, 100, 1)[0]
        expected_value_generator = np.random.default_rng(123)
        expected_value = expected_value_generator.integers(0, 100, 1)[0]
        self.assertEqual(actual_value, expected_value)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    def test_dataprofiler_seed_false_settings_seed_true(self):
        """Verify that we get the expected result when DATAPROFILER_SEED not in os.environ and settings._seed==None."""
        with unittest.mock.patch.dict("os.environ"):
            del os.environ["DATAPROFILER_SEED"]
            rng = utils_global.get_random_number_generator()
            actual_value = rng.integers(0, 1e6, 1)[0]
        self.assertNotEqual(actual_value, 850624)


class TestCorrectOutputs(unittest.TestCase):
    """Validates get_random_number_generator() is properly working."""

    def setUp(self):
        """Declare variables to be used in the rest of the test suite."""
        self.rng = utils_global.get_random_number_generator()
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
        """Check that the rng generates the expected choices from a series."""
        sample_series = self.rng.choice(
            list(self.lower_bound_list),
            (min(len(self.lower_bound_list), 5),),
            replace=False,
        ).tolist()
        self.assertListEqual(sample_series, [5, 3, 4, 1, 2])
