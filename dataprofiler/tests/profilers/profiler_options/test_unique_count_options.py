from dataprofiler.profilers.profiler_options import UniqueCountOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBooleanOption,
)


class TestUniqueCountOptions(TestBooleanOption):

    option_class = UniqueCountOptions
    keys = ["hll_hashing", "full_hashing"]

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.properties["is_enabled"])
        option = self.get_options(is_enabled=False)
        self.assertFalse(option.properties["is_enabled"])

        options = self.get_options()
        for key in self.keys:
            self.assertIn(key, options.properties)

    def test_set_helper(self):
        super().test_set_helper()
        option = self.get_options()
        for key in self.keys:
            # Enable and Disable Option
            option._set_helper({f"{key}.is_enabled": False}, "")
            self.assertFalse(option.properties[key].is_enabled)
            option._set_helper({f"{key}.is_enabled": True}, "")
            self.assertTrue(option.properties[key].is_enabled)

            # Treat is_enabled as a BooleanOption
            expected_error = (
                "type object '{}.is_enabled' has no attribute "
                "'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option._set_helper({f"{key}.is_enabled.is_enabled": True}, "")

    def test_set(self):
        super().test_set_helper()
        option = self.get_options()
        optpth = self.get_options_path()
        for key in self.keys:
            # Enable and Disable Option
            option._set_helper({f"{key}.is_enabled": False}, "")
            self.assertFalse(option.properties[key].is_enabled)
            option._set_helper({f"{key}.is_enabled": True}, "")
            self.assertTrue(option.properties[key].is_enabled)

            # Treat is_enabled as a BooleanOption
            expected_error = (
                "type object '{}.is_enabled' has no attribute "
                "'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option._set_helper({f"{key}.is_enabled.is_enabled": True}, "")

    def test_validate_helper(self):
        super().test_validate_helper()
        optpth = self.get_options_path()

        # Both options, full_hashing and hll_hashing, cannot be true
        option = self.get_options(full_hashing=True, hll_hashing=True)
        expected_error = [
            f"Both {optpth}.full_hashing and {optpth}.hll_hashing cannot be enabled "
            f"simultaneously."
        ]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Either option, full_hashing and hll_hashing, must be true
        option = self.get_options(full_hashing=False, hll_hashing=False)
        expected_error = [
            f"Either {optpth}.full_hashing and {optpth}.hll_hashing must be enabled."
        ]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

    def test_validate(self):
        super().test_validate()

        optpth = self.get_options_path()

        # Both options, full_hashing and hll_hashing, cannot be true
        option = self.get_options(full_hashing=True, hll_hashing=True)
        expected_error = "Both UniqueCountOptions.full_hashing and UniqueCountOptions.hll_hashing cannot be enabled simultaneously."
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Either option, full_hashing and hll_hashing, must be true
        option = self.get_options(full_hashing=False, hll_hashing=False)
        expected_error = "Either UniqueCountOptions.full_hashing and UniqueCountOptions.hll_hashing must be enabled."
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

    def test_eq(self):
        options = self.get_options()
        options2 = self.get_options()
        options.is_enabled = False
        self.assertNotEqual(options, options2)
        options2.is_enabled = False
        self.assertEqual(options, options2)
