import os
import unittest
from unittest import mock

import pandas as pd

from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import IntOptions
from dataprofiler.tests.profilers.test_numerical_options import TestNumericalOptions


@mock.patch('dataprofiler.profilers.data_labeler_column_profile.'
            'DataLabelerColumn.update', return_value=None)
@mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
class TestIntOptions(TestNumericalOptions):
	
	@classmethod
	def setUpClass(cls):
		super().setUpClass()
	
	@classmethod
	def getOptions(self, **params):
		options = IntOptions()
		options.set(params)
		return options
			
	@classmethod
	def getOptionsPath(self, **params):
		return "IntOptions"
		
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
	
	def test_is_numeric_stats_enabled(self, *mocks):
		super().test_is_numeric_stats_enabled(*mocks)
