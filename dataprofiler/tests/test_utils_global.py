"""Validates that generator intakes DATAPROFILER_SEED properly."""
import os
import unittest
import unittest.mock

import numpy as np
from numpy.random import PCG64

from .. import utils_global


class TestGetRandomNumberGenerator(unittest.TestCase):
    """Validates get_random_number_generator() is properly working."""

    @unittest.mock.patch.dict(os.environ, {"DATAPROFILER_SEED": "0"}, clear=True)
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_dataprofiler_seed_true_settings_seed_false(self):
        """Test for DATAPROFILER_SEED in os.environ and settings._seed!=None."""
        with unittest.mock.patch("numpy.random.default_rng") as mock_np_generator:
            rng = utils_global.get_random_number_generator()
            self.assertEqual(mock_np_generator.call_count, 1)
            mock_np_generator.assert_called_with(123)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    @unittest.mock.patch.dict("os.environ", clear=True)
    def test_dataprofiler_seed_false_settings_seed_true(self):
        """Test for DATAPROFILER_SEED not in os.environ and settings._seed==None."""
        with unittest.mock.patch("numpy.random.default_rng") as mock_np_generator:
            rng = utils_global.get_random_number_generator()
            self.assertEqual(mock_np_generator.call_count, 1)
            mock_np_generator.assert_called_with(None)

    @unittest.mock.patch.dict(os.environ, {"DATAPROFILER_SEED": "123"}, clear=True)
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    def test_dataprofiler_seed_true_settings_seed_true(self):
        """Test for DATAPROFILER_SEED in os.environ and settings._seed==None."""
        with unittest.mock.patch("numpy.random.default_rng") as mock_np_generator:
            rng = utils_global.get_random_number_generator()
            self.assertEqual(mock_np_generator.call_count, 2)
            mock_np_generator.assert_called_with(123)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    @unittest.mock.patch.dict("os.environ", clear=True)
    def test_dataprofiler_seed_false_settings_seed_false(self):
        """Test for DATAPROFILER_SEED not in os.environ and settings._seed!=None."""
        with unittest.mock.patch("numpy.random.default_rng") as mock_np_generator:
            rng = utils_global.get_random_number_generator()
            self.assertEqual(mock_np_generator.call_count, 1)
            mock_np_generator.assert_called_with(123)

    @unittest.mock.patch.dict(
        os.environ, {"DATAPROFILER_SEED": "George Washington"}, clear=True
    )
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    def test_warning_raised(self):
        """Test that warning raises if seed is not an integer."""
        with self.assertWarnsRegex(RuntimeWarning, "Seed should be an integer"):
            rng = utils_global.get_random_number_generator()
        with unittest.mock.patch("numpy.random.default_rng") as mock_np_generator:
            rng = utils_global.get_random_number_generator()
            self.assertEqual(mock_np_generator.call_count, 1)
            mock_np_generator.assert_called_with(None)
