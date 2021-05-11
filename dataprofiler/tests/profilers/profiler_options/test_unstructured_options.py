from dataprofiler.profilers.profiler_options \
    import UnstructuredOptions, BooleanOption
from dataprofiler.tests.profilers.profiler_options.test_base_option \
     import TestBaseOption


class TestUnstructuredOptions(TestBaseOption):

    option_class = UnstructuredOptions
    keys = ["text", "data_labeler"]

    @classmethod
    def get_options(self, **params):
        options = UnstructuredOptions()
        options.set(params)
        return options

    def test_init(self):
        options = self.get_options()
        for key in self.keys:
            self.assertIn(key, options.properties)

    def test_set_helper(self):
        super().test_set_helper()
        option = self.get_options()

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
        option = self.get_options()

        # Default configuration
        self.assertEqual([], option._validate_helper())

        # Wrong text option type
        option.text = BooleanOption()
        self.assertEqual(option._validate_helper(),
                         ["UnstructuredOptions.text must be a(n) "
                          "TextProfilerOptions."])

        # Wrong labeler option type
        option = self.get_options()
        option.data_labeler = BooleanOption()
        self.assertEqual(option._validate_helper(),
                         ["UnstructuredOptions.data_labeler must be a(n) "
                          "DataLabelerOptions."])

        # Both incorrect
        option.text = BooleanOption()
        self.assertCountEqual(option._validate_helper(),
                              ["UnstructuredOptions.text must be a(n) "
                               "TextProfilerOptions.",
                               "UnstructuredOptions.data_labeler must be a(n) "
                               "DataLabelerOptions."])

    def test_validate(self):
        option = self.get_options()

        # Default configuration
        option.validate()

        # Wrong text option type
        option.text = BooleanOption()
        with self.assertRaisesRegex(ValueError,
                                    r"UnstructuredOptions\.text must be"
                                    r" a\(n\) TextProfilerOptions\."):
            option.validate()

        # Wrong labeler option type
        option = self.get_options()
        option.data_labeler = BooleanOption()
        with self.assertRaisesRegex(ValueError,
                                    r"UnstructuredOptions\.data_labeler "
                                    r"must be a\(n\) DataLabelerOptions\."):
            option.validate()

        # Both incorrect
        option.text = BooleanOption()
        self.assertCountEqual(option.validate(raise_error=False),
                              ["UnstructuredOptions.text must be a(n) "
                               "TextProfilerOptions.",
                               "UnstructuredOptions.data_labeler must be a(n) "
                               "DataLabelerOptions."])

    def test_enabled_profilers(self):
        # Default
        option = self.get_options()
        self.assertCountEqual(['text', 'data_labeler'], option.enabled_profiles)

        # Disable via set
        option.set({'text.is_enabled': False})
        self.assertEqual(['data_labeler'], option.enabled_profiles)

        # Disable directly
        option.data_labeler.is_enabled = False
        self.assertEqual([], option.enabled_profiles)
