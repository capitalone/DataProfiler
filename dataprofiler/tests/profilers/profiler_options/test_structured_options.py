from dataprofiler.profilers.profiler_options import StructuredOptions
from dataprofiler.tests.profilers.profiler_options.test_base_option \
     import TestBaseOption


class TestStructuredOptions(TestBaseOption):
    
    option_class = StructuredOptions
    keys = ["int", "float", "datetime", "text", "order", "category",
            "data_labeler", "multiprocess"]

    @classmethod
    def get_options(self, **params):
        options = StructuredOptions()
        options.set(params)
        return options
    
    def test_init(self):
        options = self.get_options()
        for key in self.keys:
            self.assertIn(key, options.properties)

    def test_set_helper(self):
        super().test_set_helper()
        option = self.get_options()
        optpth = self.get_options_path()
        
        # Enable and Disable Option
        for key in self.keys:
            option._set_helper({'{}.is_enabled'.format(key): False}, '')
            self.assertFalse(option.properties[key].is_enabled)        
            option._set_helper({'{}.is_enabled'.format(key): True}, '')
            self.assertTrue(option.properties[key].is_enabled)        

        # Treat is_enabled as a BooleanOption
        for key in self.keys:
            expected_error = "type object '{}.is_enabled' has no attribute " \
                             "'is_enabled'".format(key)
            with self.assertRaisesRegex(AttributeError, expected_error):
                option._set_helper({'{}.is_enabled.is_enabled' \
                                   .format(key): True}, '')
        
    def test_set(self):
        super().test_set()
        option = self.get_options()
        optpth = self.get_options_path()

        # Enable and Disable Options        
        for key in self.keys:
            option.set({'{}.is_enabled'.format(key): False})
            self.assertFalse(option.properties[key].is_enabled)        
            option.set({'{}.is_enabled'.format(key): True})
            self.assertTrue(option.properties[key].is_enabled)        
    
        # Treat is_enabled as a BooleanOption
        for key in self.keys:
            expected_error = "type object '{}.is_enabled' has no attribute " \
                             "'is_enabled'".format(key)
            with self.assertRaisesRegex(AttributeError, expected_error):
                option.set({'{}.is_enabled.is_enabled'.format(key): True})
    
    def test_validate_helper(self):
        # Valid cases should return [] while invalid cases
        # should return a list of errors
        option = self.get_options()
        optpth = self.get_options_path()

        # Default Configuration Is Valid
        self.assertEqual([], option._validate_helper())
        
        # Variable Path Is Not A String
        expected_error = "The variable path must be a string."
        with self.assertRaisesRegex(ValueError, expected_error):
            option._validate_helper(1)
        
        # Option is_enabled is not a boolean
        for key in self.keys:
            option.set({'{}.is_enabled'.format(key): "Hello World"}) 
        expected_error = ['{}.{}.is_enabled must be a Boolean.' \
                          .format(optpth, key) for key in self.keys]
        expected_error = set(expected_error)
        # Verify expected errors are a subset of all errors
        self.assertSetEqual(expected_error,
                            expected_error \
                            .intersection(set(option._validate_helper())))

        # Wrong Class Type
        option = self.get_options()
        option.int = StructuredOptions()
        option.float = StructuredOptions()
        option.datetime = StructuredOptions()
        option.text = StructuredOptions()
        option.order = StructuredOptions()
        option.category = StructuredOptions()
        option.data_labeler = StructuredOptions()
        option.multiprocess = StructuredOptions()

        expected_error = set()
        for key in self.keys:
            ckey = key.capitalize()
            if key == "data_labeler": ckey = "DataLabeler"
            elif key == "category": ckey = "Categorical"
            elif key == "datetime": ckey = "DateTime"
            if key == "multiprocess":
                expected_error.add('{}.{} must be a(n) BooleanOption.' \
                                   .format(optpth, key, ckey))
            else:
                expected_error.add('{}.{} must be a(n) {}Options.' \
                                   .format(optpth, key, ckey))
        self.assertSetEqual(expected_error, set(option._validate_helper()))
            
    def test_validate(self):
        # Valid cases should return None while invalid cases
        # should return or throw a list of errors
        option = self.get_options()
        optpth = self.get_options_path()
    
        # Default Configuration Is Valid
        self.assertEqual(None, option.validate())
        
        # Option is_enabled is not a boolean
        for key in self.keys:
            option.set({'{}.is_enabled'.format(key): "Hello World"}) 
        
        expected_error = ["{}.{}.is_enabled must be a Boolean." \
                              .format(optpth, key)
                          for key in self.keys]
        expected_error = set(expected_error)
        # Verify expected errors are a subset of all errors
        with self.assertRaises(ValueError) as cm:
            option.validate(raise_error=True)
        raised_error = set(str(cm.exception).split("\n"))
        self.assertEqual(expected_error,
                         expected_error.intersection(raised_error))
        self.assertSetEqual(expected_error,
                            expected_error. \
                            intersection(set(option \
                                             .validate(raise_error=False))))

        # Wrong Class Type
        option = self.get_options()
        option.int = StructuredOptions()
        option.float = StructuredOptions()
        option.datetime = StructuredOptions()
        option.text = StructuredOptions()
        option.order = StructuredOptions()
        option.category = StructuredOptions()
        option.data_labeler = StructuredOptions()
        option.multiprocess = StructuredOptions()

        expected_error = set()
        for key in self.keys:
            ckey = key.capitalize()
            if key == "data_labeler": ckey = "DataLabeler"
            elif key == "category": ckey = "Categorical"
            elif key == "datetime": ckey = "DateTime"
            if key == "multiprocess":
                expected_error.add('{}.{} must be a(n) BooleanOption.' \
                                   .format(optpth, key, ckey))
            else:
                expected_error.add('{}.{} must be a(n) {}Options.' \
                                   .format(optpth, key, ckey))
        # Verify expected errors are a subset of all errors
        self.assertSetEqual(expected_error,
                            set(option.validate(raise_error=False)))
        with self.assertRaises(ValueError) as cm:
            option.validate(raise_error=True)
        raised_error = set(str(cm.exception).split("\n"))
        self.assertEqual(expected_error, raised_error)
            
    def test_enabled_columns(self):
        options = self.get_options()
        
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

