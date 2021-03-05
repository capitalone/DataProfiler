import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import BaseColumnOptions
from dataprofiler.tests.profilers.test_boolean_option import TestBooleanOption


@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestBaseColumnOptions(TestBooleanOption):
	
	@classmethod
	def setUpClass(cls):
		cls.data = Data(data=pd.DataFrame([1, 2]), data_type='csv')

	@classmethod
	def getOptions(self, **params):
		options = BaseColumnOptions()
		options.set(params)
		return options
			
	@classmethod
	def getOptionsPath(self, **params):
		return "BaseColumnOptions"
		
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
		options = self.getOptions()
		optpth = self.getOptionsPath()

		self.assertTrue(options.is_prop_enabled("is_enabled"))
		options.set({"is_enabled": False})
		self.assertFalse(options.is_prop_enabled("is_enabled"))
		
		expected_error = 'Property "Hello World" does not exist in {}.'.format(optpth)
		with self.assertRaisesRegex(AttributeError, expected_error):
			options.is_prop_enabled("Hello World")

