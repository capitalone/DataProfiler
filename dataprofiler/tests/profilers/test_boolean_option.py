import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import BooleanOption
from dataprofiler.tests.profilers.test_base_option import TestBaseOption


@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestBooleanOption(TestBaseOption):
	
	@classmethod
	def setUpClass(cls):
		cls.data = Data(data=pd.DataFrame([1, 2]), data_type='csv')
	
	@classmethod
	def getOptions(self, **params):
		return BooleanOption(**params)
	
	@classmethod
	def getOptionsPath(self, **params):
		return "BooleanOption"
		
	def test_init(self, *mocks):
		option = self.getOptions()
		self.assertEqual(option.properties, {"is_enabled": True})
	
	def test_set_helper(self, *mocks):
		super().test_set_helper(*mocks)
		
		#Enable and Disable Option
		option = self.getOptions(is_enabled=False)
		self.assertEqual(option.properties['is_enabled'], False)		
		option._set_helper({'is_enabled':True}, '') 
		self.assertEqual(option.properties['is_enabled'], True)		

		#Treat is_enabled as a BooleanOption
		expected_error = "type object 'is_enabled' has no attribute 'error'"
		with self.assertRaisesRegex(AttributeError, expected_error):
			option._set_helper({'is_enabled.error': True}, '')	
		
	def test_set(self, *mocks):
		super().test_set(*mocks)

		#Enable and Disable Options		
		option = self.getOptions(is_enabled=False)
		self.assertEqual(option.properties['is_enabled'], False)		
		option.set({'is_enabled':True}) 
		self.assertEqual(option.properties['is_enabled'], True)		
	
		#Treat is_enabled as a BooleanOption
		expected_error = "type object 'is_enabled' has no attribute 'error'"
		with self.assertRaisesRegex(AttributeError, expected_error):
			option.set({'is_enabled.error': True})	

	def test_validate_helper(self, *mocks):
		option = self.getOptions(is_enabled=True)
		optpth = self.getOptionsPath()

		#Default Configuration Is Valid
		self.assertEqual(option._validate_helper(), [])
		
		#Variable Path Is Not A String
		expected_error = "The variable path must be a string."
		with self.assertRaisesRegex(ValueError, expected_error):
			option._validate_helper(1)
		
		#Option is_enabled is not a boolean
		option = self.getOptions(is_enabled="Hello World")
		expected_error = "{}.is_enabled must be a Boolean.".format(optpth)
		self.assertEqual(option._validate_helper(), [expected_error])
	
	def test_validate(self, *mocks):
		option = self.getOptions(is_enabled=True)
		optpth = self.getOptionsPath()
	
		#Default Configuration Is Valid
		self.assertEqual(option.validate(), [])
		
		#Option is_enabled is not a boolean
		option = self.getOptions(is_enabled="Hello World")
		expected_error = "{}.is_enabled must be a Boolean.".format(optpth)
		with self.assertRaisesRegex(ValueError, expected_error):
			option.validate(raise_error=True)
		self.assertEqual(option.validate(raise_error=False), [expected_error])
