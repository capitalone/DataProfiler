from unittest import mock

from dataprofiler.profilers.profiler_options import DataLabelerOptions
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options \
     import TestBaseInspectorOptions
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler


class TestDataLabelerOptions(TestBaseInspectorOptions):
    
    option_class = DataLabelerOptions

    def test_init(self):
        options = self.get_options()
        expected_val = {'data_labeler_dirpath': None, 
            'max_sample_size': None, 
            'is_enabled': True,
            'data_labeler_object': None}

        self.assertDictEqual(expected_val, options.properties)    
    
    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
    
    def test_validate_helper(self):
        # Valid cases should return [] while invalid case
        # should return a list of errors
        super().test_validate_helper()
        optpth = self.get_options_path() 

        # Test valid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': ''})
        self.assertEqual([], options._validate_helper())

        # Test valid sample size
        options = self.get_options()
        options.set({'max_sample_size': 1})
        self.assertEqual([], options._validate_helper()) 

        # Test valid data labeler object
        options = self.get_options()
        options.set({'data_labeler_object': BaseDataLabeler()})
        self.assertEqual([], options._validate_helper())

        # Test invalid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': 0})
        expected_error = "{}.data_labeler_dirpath must be a string." \
                         .format(optpth)
        self.assertEqual([expected_error], options._validate_helper())    

        # Test invalid sample size
        options = self.get_options()
        options.set({'max_sample_size': ''})
        expected_error = "{}.max_sample_size must be an integer." \
                         .format(optpth)
        self.assertEqual([expected_error], options._validate_helper())    
        
        # Test max sample size less than or equal to 0
        options = self.get_options()
        expected_error = "{}.max_sample_size must be greater than 0." \
                         .format(optpth)
        options.set({'max_sample_size': 0})
        self.assertEqual([expected_error], options._validate_helper())    
        options.set({'max_sample_size': -1})
        self.assertEqual([expected_error], options._validate_helper())    
    
        # Test invalid data labeler object
        options = self.get_options()
        expected_error = 'DataLabelerOptions.data_labeler_object must be a '\
                         'BaseDataLabeler object.'
        options.set({'data_labeler_object': 0})
        self.assertEqual([expected_error], options._validate_helper())

    def test_validate(self):
        # Valid cases should return None while invalid cases 
        # should return or throw a list of errors
        super().test_validate()
        optpth = self.get_options_path() 

        # Test valid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': ''})
        self.assertEqual(None, options.validate())

        # Test valid sample size
        options = self.get_options()
        options.set({'max_sample_size': 1})
        self.assertEqual(None, options.validate()) 

        # Test valid data labeler object
        options = self.get_options()
        options.set({'data_labeler_object': BaseDataLabeler()})
        self.assertEqual(None, options.validate())

        # Test invalid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': 0})
        expected_error = "{}.data_labeler_dirpath must be a string." \
                         .format(optpth)
        self.assertEqual([expected_error], 
                         options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)

        # Test invalid sample size
        options = self.get_options()
        options.set({'max_sample_size': ''})
        expected_error = "{}.max_sample_size must be an integer." \
                         .format(optpth)
        self.assertEqual([expected_error], 
                         options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)

        # Test max sample size less than or equal to 0
        options = self.get_options()
        expected_error = "{}.max_sample_size must be greater than 0." \
                         .format(optpth)
        options.set({'max_sample_size': 0})
        self.assertEqual([expected_error], 
                         options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)

        options.set({'max_sample_size': -1})
        self.assertEqual([expected_error], 
                         options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)
    
        # Test invalid data labeler object
        options = self.get_options()
        expected_error = 'DataLabelerOptions.data_labeler_object must be a ' \
                         'BaseDataLabeler object.'
        options.set({'data_labeler_object': 0})
        self.assertEqual([expected_error], 
                         options.validate(raise_error=False))
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)

    def test_is_prop_enabled(self):
        super().test_is_prop_enabled()

    @mock.patch('dataprofiler.labelers.base_data_labeler.BaseDataLabeler.'
                '_load_data_labeler')
    def test_eq(self, *mocks):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.data_labeler_dirpath = "hello"
        self.assertNotEqual(options, options2)
        options2.data_labeler_dirpath = "hello there"
        self.assertNotEqual(options, options2)
        options2.data_labeler_dirpath = "hello"
        self.assertEqual(options, options2)

        # Labeler equality is determined by processor and model equality
        # the model is just set to different ints to ensure it is being
        # looked at by the options __eq__
        options.data_labeler_object = BaseDataLabeler()
        options.data_labeler_object._model = 7
        self.assertNotEqual(options, options2)
        options2.data_labeler_object = BaseDataLabeler()
        options2.data_labeler_object._model = 8
        self.assertNotEqual(options, options2)
        options2.data_labeler_object._model = 7
        self.assertEqual(options, options2)
