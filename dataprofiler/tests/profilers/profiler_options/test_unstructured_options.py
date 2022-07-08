from dataprofiler.profilers.profiler_options import BooleanOption, UnstructuredOptions
from dataprofiler.tests.profilers.profiler_options.test_base_option import (
    TestBaseOption,
)


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

        self.assertTrue(options.text.is_enabled)
        self.assertTrue(options.text.is_case_sensitive)
        self.assertIsNone(options.text.stop_words)
        self.assertTrue(options.text.vocab.is_enabled)
        self.assertTrue(options.text.words.is_enabled)
        self.assertTrue(options.data_labeler.is_enabled)
        self.assertIsNone(options.data_labeler.data_labeler_dirpath)
        self.assertIsNone(options.data_labeler.data_labeler_object)
        self.assertIsNone(options.data_labeler.max_sample_size)

    def test_set_helper(self):
        super().test_set_helper()
        option = self.get_options()

        # validate, variable path being passed
        expected_error = (
            "type object 'test.text.is_enabled' has no attribute " "'other_props'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({"text.is_enabled.other_props": True}, "test")

        expected_error = (
            "type object 'test.data_labeler.is_enabled' has no attribute "
            "'other_props'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({"data_labeler.is_enabled.other_props": True}, "test")

    def test_set(self):
        super().test_set()
        option = self.get_options()

        # Enable and Disable Options
        for key in self.keys:
            option.set({"{}.is_enabled".format(key): False})
            self.assertFalse(option.properties[key].is_enabled)
            option.set({"{}.is_enabled".format(key): True})
            self.assertTrue(option.properties[key].is_enabled)

        # Set text options
        option.set({"text.is_case_sensitive": False})
        self.assertFalse(option.text.is_case_sensitive)
        option.set({"text.stop_words": ["hello", "there"]})
        self.assertEqual(["hello", "there"], option.text.stop_words)
        option.set({"text.vocab.is_enabled": False})
        self.assertFalse(option.text.vocab.is_enabled)
        option.set({"text.words.is_enabled": False})
        self.assertFalse(option.text.words.is_enabled)

        # Set data labeler options
        option.set({"data_labeler.data_labeler_dirpath": "hi"})
        self.assertEqual("hi", option.data_labeler.data_labeler_dirpath)
        option.set({"data_labeler.max_sample_size": 12})
        self.assertEqual(12, option.data_labeler.max_sample_size)

    def test_validate_helper(self):
        option = self.get_options()

        # validate, variable path being passed
        expected_errors = ["test.text must be a(n) TextProfilerOptions."]
        option.text = 7
        self.assertEqual(expected_errors, option._validate_helper("test"))

        expected_errors = ["test.data_labeler must be a(n) DataLabelerOptions."]
        option = self.get_options()
        option.data_labeler = 7
        self.assertEqual(expected_errors, option._validate_helper("test"))

    def test_validate(self):
        option = self.get_options()

        # Default configuration
        self.assertIsNone(option.validate(raise_error=False))

        # Wrong text option type
        option.text = BooleanOption()
        self.assertEqual(
            option.validate(raise_error=False),
            ["UnstructuredOptions.text must be a(n) " "TextProfilerOptions."],
        )

        # Wrong labeler option type
        option = self.get_options()
        option.data_labeler = BooleanOption()
        self.assertEqual(
            option.validate(raise_error=False),
            ["UnstructuredOptions.data_labeler must be a(n) " "DataLabelerOptions."],
        )

        # Both incorrect
        option.text = BooleanOption()
        self.assertCountEqual(
            option.validate(raise_error=False),
            [
                "UnstructuredOptions.text must be a(n) " "TextProfilerOptions.",
                "UnstructuredOptions.data_labeler must be a(n) " "DataLabelerOptions.",
            ],
        )

    def test_enabled_profilers(self):
        # Default
        option = self.get_options()
        self.assertCountEqual(["text", "data_labeler"], option.enabled_profiles)

        # Disable via set
        option.set({"text.is_enabled": False})
        self.assertEqual(["data_labeler"], option.enabled_profiles)

        # Disable directly
        option.data_labeler.is_enabled = False
        self.assertEqual([], option.enabled_profiles)

    def test_eq(self):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.data_labeler.is_enabled = False
        self.assertNotEqual(options, options2)
        options2.data_labeler.is_enabled = False
        self.assertEqual(options, options2)

        options.text.stop_words = ["woah", "stop", "right", "there"]
        self.assertNotEqual(options, options2)
        options2.text.stop_words = ["those", "don't", "match"]
        self.assertNotEqual(options, options2)
        options2.text.stop_words = ["woah", "stop", "right", "there"]
        self.assertEqual(options, options2)
