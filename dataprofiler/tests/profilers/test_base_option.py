import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import BaseOption


@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestBaseOption(unittest.TestCase):
	
	@classmethod
	def setUpClass(cls):
		cls.data = Data(data=pd.DataFrame([1, 2]), data_type='csv')
	
	def test_init(self, *mocks):
		options = BaseOption()
		self.assertDictEqual({}, options.properties) 

	def test_set_helper(self, *mocks):
		options = BaseOption()

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
		options = BaseOption()

		# Options Is Not A Dictionary
		expected_error = "The options must be a dictionary."
		with self.assertRaisesRegex(ValueError, expected_error):
			options.set("notadictionary")
		with self.assertRaisesRegex(ValueError, expected_error):
			options.set(["not", "a", "dictionary"])
	
	def test_validate_helper(self, *mocks):
		options = BaseOption()

		with self.assertRaises(NotImplementedError):
			options._validate_helper()
	
	def test_validate(self, *mocks):
		options = BaseOption()

		with self.assertRaises(NotImplementedError):
			options.validate()
