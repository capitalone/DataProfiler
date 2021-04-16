from dataprofiler.profilers.profiler_options \
     import BooleanOption, NumericalOptions
from dataprofiler.tests.profilers.profiler_options.test_base_column_options \
     import TestBaseColumnOptions


class TestNumericalOptions(TestBaseColumnOptions):
    
    option_class = NumericalOptions
    keys = ["min", "max", "sum", "variance", "histogram_and_quantiles"]

    def test_init(self):
        options = self.get_options()
        for key in self.keys + ['is_numeric_stats_enabled', 'is_enabled']:
            self.assertIn(key, options.properties)
    
    def test_set_helper(self):
        super().test_set_helper()
        options = self.get_options()

        # Enable and Disable Options
        for key in self.keys:
            skey = '{}.is_enabled'.format(key)
            for enabled in [True, False]:
                options._set_helper({skey:enabled}, '') 
                self.assertEqual(enabled, options.properties[key].is_enabled) 

    def test_set(self):
        super().test_set()
        options = self.get_options()
        
        # Enable and Disable Options
        for key in self.keys:
            skey = '{}.is_enabled'.format(key)
            for enabled in [True, False]:
                options.set({skey:enabled}) 
                self.assertEqual(enabled, options.properties[key].is_enabled)
    
    def test_validate_helper(self):
        super().test_validate_helper()
        options = self.get_options()
        optpth = self.get_options_path()
    
        # Set BooleanOptions' is_enabled to a non-boolean value
        for key in self.keys:
            skey = '{}.is_enabled'.format(key)
            expected_error = "{}.{}.is_enabled must be a Boolean." \
                             .format(optpth, key)
            
            options.set({skey: "Hello World"})
            self.assertEqual([expected_error], options._validate_helper())
            options.set({skey: True})

        # Disable Sum and Enable Variance
        options.set({"sum.is_enabled": False, "variance.is_enabled": True})
        expected_error = "{}: The numeric stats must toggle on the sum if " \
                         "the variance is toggled on.".format(optpth)
        self.assertEqual([expected_error], options._validate_helper()) 
    
    def test_validate(self):
        super().test_validate()
        options = self.get_options()
        optpth = self.get_options_path()
        
        # Set BooleanOptions' is_enabled to a non-boolean value
        for key in self.keys:
            skey = '{}.is_enabled'.format(key)
            expected_error = "{}.{}.is_enabled must be a Boolean."\
                .format(optpth, key)
            
            options.set({skey: "Hello World"})
            with self.assertRaisesRegex(ValueError, expected_error):
                options.validate(raise_error=True)    
            self.assertEqual([expected_error], 
                             options.validate(raise_error=False))
            options.set({skey: True})

        # Disable Sum and Enable Variance
        options.set({"sum.is_enabled": False, "variance.is_enabled": True})
        expected_error = "{}: The numeric stats must toggle on the sum if " \
                         "the variance is toggled on.".format(optpth)
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate(raise_error=True)    
        self.assertEqual([expected_error], options.validate(raise_error=False))
    
    def test_is_numeric_stats_enabled(self):
        options = self.get_options()
        numeric_keys = ["min", "max", "sum", "variance", 
                        "histogram_and_quantiles"] 

        # Disable All Numeric Stats
        options.set({'{}.is_enabled'.format(key):False 
                     for key in numeric_keys})
        self.assertFalse(options.is_numeric_stats_enabled)
        
        # Enable Only One Numeric Stat
        for key in numeric_keys:
            skey = '{}.is_enabled'.format(key)
            options.set({skey: True})
            self.assertTrue(options.is_numeric_stats_enabled)
            options.set({skey: False})

        # Enable All Numeric Stats
        options.is_numeric_stats_enabled = True
        for key in numeric_keys:            
            self.assertTrue(options.is_numeric_stats_enabled)

        # Disable All Numeric Stats
        options.is_numeric_stats_enabled = False
        for key in numeric_keys:            
            self.assertFalse(options.is_numeric_stats_enabled)

