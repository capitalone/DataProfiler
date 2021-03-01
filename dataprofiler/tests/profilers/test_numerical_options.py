import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import BooleanOption, NumericalOptions
from dataprofiler.tests.profilers.test_base_column_options import TestBaseColumnOptions


@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestNumericalOptions(TestBaseColumnOptions):
	
	@classmethod
	def setUpClass(cls):
		cls.keys = ["min", "max", "sum", "variance", "histogram_and_quantiles"]
	
	def sanity(self, *mocks):
		options = NumericalOptions()
		for key in keys:
			self.assertTrue(key in options.properties)
	
	def test_set_helper(self, *mocks):
		super().test_set_helper(*mocks)

		#Enable and Disable Options
		options = NumericalOptions()
		for key in self.keys:
			skey = '{}.is_enabled'.format(key)
			for enabled in [True, False]:
				options._set_helper({skey:enabled}, '') 
				self.assertEqual(options.properties[key].is_enabled, enabled)		

	def test_set(self, *mocks):
		super().test_set(*mocks)
		
		#Enable and Disable Options
		options = NumericalOptions()
		for key in self.keys:
			skey = '{}.is_enabled'.format(key)
			for enabled in [True, False]:
				options.set({skey:enabled}) 
				self.assertEqual(options.properties[key].is_enabled, enabled)		
	
	def test_validate_helper(self, *mocks):
		super().test_validate_helper(*mocks)
	
		options = NumericalOptions()
		for key in self.keys:
			skey = '{}.is_enabled'.format(key)
			expected_error = "NumericalOptions.{}.is_enabled must be a Boolean.".format(key)
			
			options.set({skey: "Hello World"})
			self.assertEqual(options._validate_helper(), [expected_error])
			options.set({skey: True})

		options.set({"sum.is_enabled": False, "variance.is_enabled": True})
		expected_error = "NumericalOptions: The numeric stats must toggle on the sum if the variance is toggled on."
		self.assertEqual(options._validate_helper(), [expected_error]) 
	
	def test_validate(self, *mocks):
		super().test_validate(*mocks)

		options = NumericalOptions()
		for key in self.keys:
			skey = '{}.is_enabled'.format(key)
			expected_error = "NumericalOptions.{}.is_enabled must be a Boolean.".format(key)
			
			options.set({skey: "Hello World"})
			with self.assertRaisesRegex(ValueError, expected_error):
				options.validate(raise_error=True)	
			self.assertEqual(options.validate(raise_error=False), [expected_error])
			options.set({skey: True})

		options.set({"sum.is_enabled": False, "variance.is_enabled": True})
		expected_error = "NumericalOptions: The numeric stats must toggle on the sum if the variance is toggled on."
		with self.assertRaisesRegex(ValueError, expected_error):
			options.validate(raise_error=True)	
		self.assertEqual(options.validate(raise_error=False), [expected_error]) 
		
