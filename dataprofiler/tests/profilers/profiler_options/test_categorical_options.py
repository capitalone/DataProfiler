import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import CategoricalOptions
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options import (
    TestBaseInspectorOptions,
)


class TestCategoricalOptions(TestBaseInspectorOptions):

    option_class = CategoricalOptions

    def test_init(self):
        option = self.get_options()
        self.assertDictEqual(
            {"is_enabled": True, "top_k_categories": None}, option.properties
        )
        option = self.get_options(is_enabled=False)
        self.assertDictEqual(
            {"is_enabled": False, "top_k_categories": None}, option.properties
        )
        option = self.get_options(top_k_categories=2)
        self.assertDictEqual(
            {"is_enabled": True, "top_k_categories": 2}, option.properties
        )

    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
        option = self.get_options()

        params_to_check = [
            dict(prop="is_enabled", value_list=[False, True]),
            dict(prop="top_k_categories", value_list=[None, 3]),
        ]

        # this code can be abstracted to limit code everywhere else
        # AKA, params_to_check would be the only needed code plus raise errors
        def _assert_set_helper(prop, value):
            option.set({prop: value})
            self.assertEqual(value, getattr(option, prop), msg=prop)

        for params in params_to_check:
            prop, value_list = params["prop"], params["value_list"]
            for value in value_list:
                _assert_set_helper(prop, value)

        # Treat is_case_sensitive and stop_words as BooleanOption
        expected_error = (
            "type object 'top_k_categories' has no attribute " "'is_enabled'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({"top_k_categories.is_enabled": True})

    def test_validate_helper(self):
        super().test_validate_helper()
        optpth = self.get_options_path()
        # These ones should throw an error
        expected_error = (
            "{}.top_k_categories must be either None"
            " or a positive integer".format(optpth)
        )
        options = self.get_options(top_k_categories=-2)
        self.assertEqual([expected_error], options._validate_helper())
        options = self.get_options(top_k_categories="E")
        self.assertEqual([expected_error], options._validate_helper())
        options = self.get_options(top_k_categories=2.0)
        self.assertEqual([expected_error], options._validate_helper())
        # These ones should not
        expected_error = ""
        options = self.get_options(top_k_categories=6)
        self.assertEqual([], options._validate_helper())
        options = self.get_options(top_k_categories=None)
        self.assertEqual([], options._validate_helper())

    def test_validate(self):
        super().test_validate()

    def test_is_prop_enabled(self):
        super().test_is_prop_enabled()

    def test_eq(self):
        super().test_eq()

    def test_json_encode(self):
        option = CategoricalOptions(is_enabled=False, top_k_categories=5)

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected_class = "CategoricalOptions"
        expected_options_attributes = {"is_enabled", "top_k_categories"}
        expected_is_enabled = option.is_enabled
        expected_top_k_categories = option.top_k_categories

        actual_option_json = json.loads(serialized)

        self.assertIn("class", actual_option_json)
        self.assertEqual(expected_class, actual_option_json["class"])
        self.assertIn("data", actual_option_json)
        self.assertEqual(
            expected_options_attributes, set(actual_option_json["data"].keys())
        )
        self.assertIn("is_enabled", actual_option_json["data"])
        self.assertEqual(expected_is_enabled, actual_option_json["data"]["is_enabled"])
        self.assertIn("top_k_categories", actual_option_json["data"])
        self.assertEqual(
            expected_top_k_categories,
            actual_option_json["data"]["top_k_categories"],
        )
