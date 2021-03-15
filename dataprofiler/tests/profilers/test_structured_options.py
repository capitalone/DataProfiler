import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import StructuredOptions
from dataprofiler.tests.profilers.test_base_option import TestBaseOption

@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestStructuredOptions(TestBaseOption):
	
	@classmethod
	def setUpClass(cls):
		cls.keys = ["int", "float", "datetime", "text", "order", "category", "data_labeler"]
	
	@classmethod
	def getOptions(self, **params):
		options = StructuredOptions()
		options.set(params)
		return options
	
	@classmethod
	def getOptionsPath(self):
		return "StructuredOptions"
	
	def test_init(self, *mocks):
		options = self.getOptions()
		for key in self.keys:
			self.assertTrue(key in options.properties)

	def test_set_helper(self, *mocks):
		super().test_set_helper(*mocks)
		option = self.getOptions()
		optpth = self.getOptionsPath()
		
		# Enable and Disable Option
		for key in self.keys:
			option._set_helper({'{}.is_enabled'.format(key): False}, '')
			self.assertFalse(option.properties[key].is_enabled)		
			option._set_helper({'{}.is_enabled'.format(key): True}, '')
			self.assertTrue(option.properties[key].is_enabled)		

		# Treat is_enabled as a BooleanOption
		for key in self.keys:
			expected_error = "type object '{}.is_enabled' has no attribute 'is_enabled'".format(key)
			with self.assertRaisesRegex(AttributeError, expected_error):
				option._set_helper({'{}.is_enabled.is_enabled'.format(key): True}, '')	
		
	def test_set(self, *mocks):
		super().test_set(*mocks)
		option = self.getOptions()
		optpth = self.getOptionsPath()

		# Enable and Disable Options		
		for key in self.keys:
			option.set({'{}.is_enabled'.format(key): False})
			self.assertFalse(option.properties[key].is_enabled)		
			option.set({'{}.is_enabled'.format(key): True})
			self.assertTrue(option.properties[key].is_enabled)		
	
		# Treat is_enabled as a BooleanOption
		for key in self.keys:
			expected_error = "type object '{}.is_enabled' has no attribute 'is_enabled'".format(key)
			with self.assertRaisesRegex(AttributeError, expected_error):
				option.set({'{}.is_enabled.is_enabled'.format(key): True})
	
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
		for key in self.keys:
			option.set({'{}.is_enabled'.format(key): "Hello World"}) 
		expected_error = ['{}.{}.is_enabled must be a Boolean.'.format(optpth, key) 
			for key in self.keys]
		expected_error = set(expected_error)
		self.assertSetEqual(expected_error, expected_error.intersection(set(option._validate_helper())))

		# Wrong Class Type
		option = self.getOptions()
		option.int = StructuredOptions()
		option.float = StructuredOptions()
		option.datetime = StructuredOptions()
		option.text = StructuredOptions()
		option.order = StructuredOptions()
		option.category = StructuredOptions()
		option.data_labeler = StructuredOptions()

		expected_error = set()
		for key in self.keys:
			ckey = key.capitalize()
			if key == "data_labeler": ckey = "DataLabeler"
			elif key == "category": ckey = "Categorical"
			elif key == "datetime": ckey = "DateTime"
			expected_error.add('{}.{} must be a {}Options.'.format(optpth, key, ckey))
		expected_error = set(expected_error)
		self.assertSetEqual(expected_error, set(option._validate_helper()))
			
	def test_validate(self, *mocks):
		option = self.getOptions()
		optpth = self.getOptionsPath()
	
		# Default Configuration Is Valid
		self.assertEqual([], option.validate())
		
		# Option is_enabled is not a boolean
		for key in self.keys:
			option.set({'{}.is_enabled'.format(key): "Hello World"}) 
		
		expected_error = ["{}.{}.is_enabled must be a Boolean.".format(optpth, key) 
			for key in self.keys]
		expected_error = set(expected_error)
		self.assertSetEqual(expected_error, expected_error.intersection(set(option.validate(raise_error=False))))

		# Wrong Class Type
		option = self.getOptions()
		option.int = StructuredOptions()
		option.float = StructuredOptions()
		option.datetime = StructuredOptions()
		option.text = StructuredOptions()
		option.order = StructuredOptions()
		option.category = StructuredOptions()
		option.data_labeler = StructuredOptions()

		expected_error = set()
		for key in self.keys:
			ckey = key.capitalize()
			if key == "data_labeler": ckey = "DataLabeler"
			elif key == "category": ckey = "Categorical"
			elif key == "datetime": ckey = "DateTime"
			expected_error.add('{}.{} must be a {}Options.'.format(optpth, key, ckey))
		expected_error = set(expected_error)
		self.assertSetEqual(expected_error, set(option.validate(raise_error=False)))
			
	def test_enabled_columns(self, *mocks):
		options = self.getOptions()
		
		# All Columns Enabled
		for key in self.keys: 
			options.set({'{}.is_enabled'.format(key): True})
		self.assertSetEqual(set(self.keys), set(options.enabled_columns))

		# No Columns Enabled		
		for key in self.keys: 
			options.set({'{}.is_enabled'.format(key): False})
		self.assertEqual([], options.enabled_columns)

		# One Column Enabled
		for key in self.keys:
			options.set({'{}.is_enabled'.format(key): True})
			self.assertSetEqual(set([key]), set(options.enabled_columns))
			options.set({'{}.is_enabled'.format(key): False})

