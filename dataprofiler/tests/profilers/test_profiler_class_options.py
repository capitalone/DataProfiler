import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import ProfilerOptions
from dataprofiler.tests.profilers.test_base_option import TestBaseOption

@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestProfilerOptions(TestBaseOption):
	
	@classmethod
	def setUpClass(cls):
		cls.keys = ["structured_options"]
	
	@classmethod
	def getOptions(self, **params):
		options = ProfilerOptions()
		options.set(params)
		return options
	
	@classmethod
	def getOptionsPath(self):
		return "ProfilerOptions"
	
	def test_init(self, *mocks):
		options = self.getOptions()
		for key in self.keys:
			self.assertTrue(key in options.properties)

	def test_set_helper(self, *mocks):
		super().test_set_helper(*mocks)
		option = self.getOptions()
		optpth = self.getOptionsPath()
		
		# Enable and Disable Option
		for key in option.structured_options.properties:
			option._set_helper({'structured_options.{}.is_enabled'.format(key): False}, '')
			self.assertFalse(option.structured_options.properties[key].is_enabled)		
			option._set_helper({'structured_options.{}.is_enabled'.format(key): True}, '')
			self.assertTrue(option.structured_options.properties[key].is_enabled)		

		# Treat is_enabled as a BooleanOption
		for key in option.structured_options.properties:
			expected_error = "type object 'structured_options.{}.is_enabled' has no attribute 'is_enabled'".format(key)
			with self.assertRaisesRegex(AttributeError, expected_error):
				option._set_helper({'structured_options.{}.is_enabled.is_enabled'.format(key): True}, '')	
		
	def test_set(self, *mocks):
		super().test_set(*mocks)
		option = self.getOptions()
		optpth = self.getOptionsPath()

		# Enable and Disable Options		
		for key in option.structured_options.properties:
			option.set({'structured_options.{}.is_enabled'.format(key): False})
			self.assertFalse(option.structured_options.properties[key].is_enabled)		
			option.set({'structured_options.{}.is_enabled'.format(key): True})
			self.assertTrue(option.structured_options.properties[key].is_enabled)		
	
		# Treat is_enabled as a BooleanOption
		for key in option.structured_options.properties:
			expected_error = "type object 'structured_options.{}.is_enabled' has no attribute 'is_enabled'".format(key)
			with self.assertRaisesRegex(AttributeError, expected_error):
				option.set({'structured_options.{}.is_enabled.is_enabled'.format(key): True})
	
	def test_validate_helper(self, *mocks):
		option = self.getOptions()
		optpth = self.getOptionsPath()

		# Default Configuration Is Valid
		self.assertEqual([], option._validate_helper())
		
		# Variable Path Is Not A String
		expected_error = "The variable path must be a string."
		with self.assertRaisesRegex(ValueError, expected_error):
			option._validate_helper(1)
		
		# Option is_enabled is not a boolean
		for key in option.structured_options.properties:
			option.set({'structured_options.{}.is_enabled'.format(key): "Hello World"}) 

		expected_error = ['{}.structured_options.{}.is_enabled must be a Boolean.'.format(optpth, key) 
			for key in option.structured_options.properties]
		expected_error = set(expected_error)
		self.assertSetEqual(expected_error, expected_error.intersection(set(option._validate_helper())))

		# Wrong Class Type
		option = self.getOptions()
		option.structured_options = ProfilerOptions()

		expected_error = set()
		expected_error.add('{}.structured_options must be a StructuredOptions.'.format(optpth,))

		self.assertSetEqual(expected_error, set(option._validate_helper()))
			
	def test_validate(self, *mocks):
		option = self.getOptions()
		optpth = self.getOptionsPath()
	
		# Default Configuration Is Valid
		self.assertEqual([], option.validate())
		
		# Option is_enabled is not a boolean
		for key in option.structured_options.properties:
			option.set({'structured_options.{}.is_enabled'.format(key): "Hello World"}) 

		expected_error = ['{}.structured_options.{}.is_enabled must be a Boolean.'.format(optpth, key) 
			for key in option.structured_options.properties]
		expected_error = set(expected_error)
		self.assertSetEqual(expected_error, expected_error.intersection(set(option.validate(raise_error=False))))

		# Wrong Class Type
		option = self.getOptions()
		option.structured_options = ProfilerOptions()

		expected_error = set()
		expected_error.add('{}.structured_options must be a StructuredOptions.'.format(optpth,))
		self.assertSetEqual(expected_error, set(option.validate(raise_error=False)))
