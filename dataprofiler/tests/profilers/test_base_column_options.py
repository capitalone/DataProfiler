import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import BaseColumnOptions
from dataprofiler.tests.profilers.test_boolean_option import TestBooleanOption


class TestBaseColumnOptions(TestBooleanOption):
    
    @classmethod
    def setUpClass(cls):
        cls.option_class = BaseColumnOptions

    @classmethod
    def get_options(self, *args, **params):
        self.validate_option_class()
        options = self.option_class()
        options.set(params)
        return options
            
    def test_init(self, *mocks):
        super().test_init(*mocks)
    
    def test_set_helper(self, *mocks):
        super().test_set_helper(*mocks)

    def test_set(self, *mocks):
        super().test_set(*mocks)
    
    def test_validate_helper(self, *mocks):
        super().test_validate_helper(*mocks)
    
    def test_validate(self, *mocks):
        super().test_validate(*mocks)

    def test_is_prop_enabled(self, *mocks):
        options = self.get_options()
        optpth = self.get_options_path()

        # Check is prop enabled for valid property
        options.set({"is_enabled": True})
        self.assertTrue(options.is_prop_enabled("is_enabled"))
        options.set({"is_enabled": False})
        self.assertFalse(options.is_prop_enabled("is_enabled"))
    
        # Check is prop enabled for invalid property    
        expected_error = 'Property "Hello World" does not exist in {}.'.format(optpth)
        with self.assertRaisesRegex(AttributeError, expected_error):
            options.is_prop_enabled("Hello World")

