import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import BaseOption
from dataprofiler.tests.profilers.abstract_test_options import AbstractTestOptions

class TestBaseOption(AbstractTestOptions, unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.option_class = BaseOption

    def test_init(self, *mocks):
        options = self.get_options()
        self.assertDictEqual({}, options.properties) 

    def test_set_helper(self, *mocks):
        options = self.get_options()

        # Options Is Not A Dictionary
        expected_error = "The options must be a dictionary."
        with self.assertRaisesRegex(ValueError, expected_error):
            options._set_helper("notadictionary", "")
        with self.assertRaisesRegex(ValueError, expected_error):
            options._set_helper(["not", "a", "dictionary"], "")

        # Variable Path Is Not A String
        expected_error = "The variable path must be a string."
        with self.assertRaisesRegex(ValueError, expected_error):
            options._set_helper({"hello": "world"}, 1)
        with self.assertRaisesRegex(ValueError, expected_error):
            options._set_helper({}, 1)

    def test_set(self, *mocks):
        options = self.get_options()

        # Options Is Not A Dictionary
        expected_error = "The options must be a dictionary."
        with self.assertRaisesRegex(ValueError, expected_error):
            options.set("notadictionary")
        with self.assertRaisesRegex(ValueError, expected_error):
            options.set(["not", "a", "dictionary"])
    
    def test_validate_helper(self, *mocks):
        options = self.get_options()

        with self.assertRaises(NotImplementedError):
            options._validate_helper()
    
    def test_validate(self, *mocks):
        options = self.get_options()

        with self.assertRaises(NotImplementedError):
            options.validate()
