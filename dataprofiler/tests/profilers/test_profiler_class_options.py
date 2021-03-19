from dataprofiler.profilers.profiler_options import ProfilerOptions
from dataprofiler.tests.profilers.test_base_option import TestBaseOption

class TestProfilerOptions(TestBaseOption):
    
    option_class = ProfilerOptions
    keys = ["structured_options"]
    
    @classmethod
    def get_options(self, **params):
        options = ProfilerOptions()
        options.set(params)
        return options
    
    def test_init(self, *mocks):
        options = self.get_options()
        for key in self.keys:
            self.assertTrue(key in options.properties)

    def test_set_helper(self, *mocks):
        super().test_set_helper(*mocks)
        option = self.get_options()
        optpth = self.get_options_path()
        
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
        option = self.get_options()
        optpth = self.get_options_path()

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
        option = self.get_options()
        optpth = self.get_options_path()

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
        option = self.get_options()
        option.structured_options = ProfilerOptions()

        expected_error = set()
        expected_error.add('{}.structured_options must be a StructuredOptions.'.format(optpth,))

        self.assertSetEqual(expected_error, set(option._validate_helper()))
            
    def test_validate(self, *mocks):
        option = self.get_options()
        optpth = self.get_options_path()
    
        # Default Configuration Is Valid
        self.assertEqual(None, option.validate())
        
        # Option is_enabled is not a boolean
        for key in option.structured_options.properties:
            option.set({'structured_options.{}.is_enabled'.format(key): "Hello World"}) 

        expected_error = ['{}.structured_options.{}.is_enabled must be a Boolean.'.format(optpth, key) 
            for key in option.structured_options.properties]
        expected_error = set(expected_error)
        self.assertSetEqual(expected_error, expected_error.intersection(set(option.validate(raise_error=False))))

        # Wrong Class Type
        option = self.get_options()
        option.structured_options = ProfilerOptions()

        expected_error = set()
        expected_error.add('{}.structured_options must be a StructuredOptions.'.format(optpth,))
        self.assertSetEqual(expected_error, set(option.validate(raise_error=False)))
