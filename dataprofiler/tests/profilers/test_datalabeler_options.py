from dataprofiler.profilers.profiler_options import DataLabelerOptions
from dataprofiler.tests.profilers.test_base_column_options import TestBaseColumnOptions


class TestDataLabelerOptions(TestBaseColumnOptions):
    
    option_class = DataLabelerOptions

    def test_init(self, *mocks):
        options = self.get_options()
        expected_val = {'data_labeler_dirpath': None, 
            'max_sample_size': None, 
            'is_enabled': True}
        self.assertDictEqual(expected_val, options.properties)    
    
    def test_set_helper(self, *mocks):
        super().test_set_helper(*mocks)

    def test_set(self, *mocks):
        super().test_set(*mocks)
    
    def test_validate_helper(self, *mocks):
        super().test_validate_helper(*mocks)
        optpth = self.get_options_path() 

        # Test valid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': ''})
        self.assertEqual([], options._validate_helper())

        # Test valid sample size
        options = self.get_options()
        options.set({'max_sample_size': 1})
        self.assertEqual([], options._validate_helper()) 

        # Test invalid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': 0})
        expected_error = "{}.data_labeler_dirpath must be a string.".format(optpth)
        self.assertEqual([expected_error], options._validate_helper())    

        # Test invalid sample size
        options = self.get_options()
        options.set({'max_sample_size': ''})
        expected_error = "{}.max_sample_size must be an integer.".format(optpth)
        self.assertEqual([expected_error], options._validate_helper())    
        
        # Test max sample size less than or equal to 0
        options = self.get_options()
        expected_error = "{}.max_sample_size must be greater than 0.".format(optpth)
        options.set({'max_sample_size': 0})
        self.assertEqual([expected_error], options._validate_helper())    
        options.set({'max_sample_size': -1})
        self.assertEqual([expected_error], options._validate_helper())    
    
    def test_validate(self, *mocks):
        super().test_validate(*mocks)
        optpth = self.get_options_path() 

        # Test valid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': ''})
        self.assertEqual(None, options.validate())

        # Test valid sample size
        options = self.get_options()
        options.set({'max_sample_size': 1})
        self.assertEqual(None, options.validate()) 

        # Test invalid dirpath
        options = self.get_options()
        options.set({'data_labeler_dirpath': 0})
        expected_error = "{}.data_labeler_dirpath must be a string.".format(optpth)
        self.assertEqual([expected_error], options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)

        # Test invalid sample size
        options = self.get_options()
        options.set({'max_sample_size': ''})
        expected_error = "{}.max_sample_size must be an integer.".format(optpth)
        self.assertEqual([expected_error], options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)

        # Test max sample size less than or equal to 0
        options = self.get_options()
        expected_error = "{}.max_sample_size must be greater than 0.".format(optpth)
        options.set({'max_sample_size': 0})
        self.assertEqual([expected_error], options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)
        options.set({'max_sample_size': -1})
        self.assertEqual([expected_error], options.validate(raise_error=False))    
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)
    
    def test_is_prop_enabled(self, *mocks):
        super().test_is_prop_enabled(*mocks)
