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
            {
                "is_enabled": True,
                "top_k_categories": None,
                "max_sample_size_to_check_stop_condition": None,
                "stop_condition_unique_value_ratio": None,
                "cms": False,
                "cms_confidence": 0.95,
                "cms_relative_error": 0.01,
                "cms_max_num_heavy_hitters": 5000,
            },
            option.properties,
        )
        option = self.get_options(is_enabled=False)
        self.assertDictEqual(
            {
                "is_enabled": False,
                "top_k_categories": None,
                "max_sample_size_to_check_stop_condition": None,
                "stop_condition_unique_value_ratio": None,
                "cms": False,
                "cms_confidence": 0.95,
                "cms_relative_error": 0.01,
                "cms_max_num_heavy_hitters": 5000,
            },
            option.properties,
        )
        option = self.get_options(top_k_categories=2)
        self.assertDictEqual(
            {
                "is_enabled": True,
                "top_k_categories": 2,
                "max_sample_size_to_check_stop_condition": None,
                "stop_condition_unique_value_ratio": None,
                "cms": False,
                "cms_confidence": 0.95,
                "cms_relative_error": 0.01,
                "cms_max_num_heavy_hitters": 5000,
            },
            option.properties,
        )
        option = self.get_options(max_sample_size_to_check_stop_condition=20)
        self.assertDictEqual(
            {
                "is_enabled": True,
                "top_k_categories": None,
                "max_sample_size_to_check_stop_condition": 20,
                "stop_condition_unique_value_ratio": None,
                "cms": False,
                "cms_confidence": 0.95,
                "cms_relative_error": 0.01,
                "cms_max_num_heavy_hitters": 5000,
            },
            option.properties,
        )
        option = self.get_options(stop_condition_unique_value_ratio=2)
        self.assertDictEqual(
            {
                "is_enabled": True,
                "top_k_categories": None,
                "max_sample_size_to_check_stop_condition": None,
                "stop_condition_unique_value_ratio": 2,
                "cms": False,
                "cms_confidence": 0.95,
                "cms_relative_error": 0.01,
                "cms_max_num_heavy_hitters": 5000,
            },
            option.properties,
        )
        option = self.get_options(
            max_sample_size_to_check_stop_condition=20,
            stop_condition_unique_value_ratio=2,
        )
        self.assertDictEqual(
            {
                "is_enabled": True,
                "top_k_categories": None,
                "max_sample_size_to_check_stop_condition": 20,
                "stop_condition_unique_value_ratio": 2,
                "cms": False,
                "cms_confidence": 0.95,
                "cms_relative_error": 0.01,
                "cms_max_num_heavy_hitters": 5000,
            },
            option.properties,
        )
        option = self.get_options(
            cms=True,
            cms_confidence=0.98,
            cms_relative_error=0.1,
            cms_max_num_heavy_hitters=5,
        )
        self.assertDictEqual(
            {
                "is_enabled": True,
                "top_k_categories": None,
                "max_sample_size_to_check_stop_condition": None,
                "stop_condition_unique_value_ratio": None,
                "cms": True,
                "cms_confidence": 0.98,
                "cms_relative_error": 0.1,
                "cms_max_num_heavy_hitters": 5,
            },
            option.properties,
        )

    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
        option = self.get_options()

        params_to_check = [
            dict(prop="is_enabled", value_list=[False, True]),
            dict(prop="top_k_categories", value_list=[None, 3]),
            dict(prop="max_sample_size_to_check_stop_condition", value_list=[None, 20]),
            dict(prop="stop_condition_unique_value_ratio", value_list=[None, 0.7]),
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

        # These ones should throw a top_k_error
        expected_top_k_error = (
            "{}.top_k_categories must be either None"
            " or a positive integer".format(optpth)
        )

        options = self.get_options(top_k_categories=-2)
        self.assertEqual([expected_top_k_error], options._validate_helper())
        options = self.get_options(top_k_categories="E")
        self.assertEqual([expected_top_k_error], options._validate_helper())
        options = self.get_options(top_k_categories=2.0)
        self.assertEqual([expected_top_k_error], options._validate_helper())

        # These ones should throw a max_sample_size_type_error
        expected_max_sample_size_type_error = (
            "{}.max_sample_size_to_check_stop_condition must be either None"
            " or a non-negative integer".format(optpth)
        )

        options = self.get_options(
            max_sample_size_to_check_stop_condition=-2,
            stop_condition_unique_value_ratio=1.0,
        )
        self.assertEqual(
            [expected_max_sample_size_type_error], options._validate_helper()
        )
        options = self.get_options(
            max_sample_size_to_check_stop_condition="E",
            stop_condition_unique_value_ratio=1.0,
        )
        self.assertEqual(
            [expected_max_sample_size_type_error], options._validate_helper()
        )
        options = self.get_options(
            max_sample_size_to_check_stop_condition=2.0,
            stop_condition_unique_value_ratio=1.0,
        )
        self.assertEqual(
            [expected_max_sample_size_type_error], options._validate_helper()
        )

        # These ones should throw a unique_ratio_type_error
        expected_unique_value_ratio_type_error = (
            "{}.stop_condition_unique_value_ratio must be either None"
            " or a float between 0 and 1".format(optpth)
        )

        options = self.get_options(
            stop_condition_unique_value_ratio=2.7,
            max_sample_size_to_check_stop_condition=20,
        )
        self.assertEqual(
            [expected_unique_value_ratio_type_error], options._validate_helper()
        )
        options = self.get_options(
            stop_condition_unique_value_ratio="E",
            max_sample_size_to_check_stop_condition=20,
        )
        self.assertEqual(
            [expected_unique_value_ratio_type_error], options._validate_helper()
        )
        options = self.get_options(
            stop_condition_unique_value_ratio=-0.7,
            max_sample_size_to_check_stop_condition=20,
        )
        self.assertEqual(
            [expected_unique_value_ratio_type_error], options._validate_helper()
        )
        options = self.get_options(
            stop_condition_unique_value_ratio=2,
            max_sample_size_to_check_stop_condition=20,
        )
        self.assertEqual(
            [expected_unique_value_ratio_type_error], options._validate_helper()
        )

        # These ones should throw a stop_condition_error
        expected_stop_condition_error = (
            "Both, {}.max_sample_size_to_check_stop_condition and "
            "{}.stop_condition_unique_value_ratio, options either need to be "
            "set or not set.".format(optpth, optpth)
        )

        options = self.get_options(max_sample_size_to_check_stop_condition=20)
        self.assertEqual([expected_stop_condition_error], options._validate_helper())
        options = self.get_options(stop_condition_unique_value_ratio=1.0)
        self.assertEqual([expected_stop_condition_error], options._validate_helper())
        options = self.get_options(
            max_sample_size_to_check_stop_condition=20,
            stop_condition_unique_value_ratio=None,
        )
        self.assertEqual([expected_stop_condition_error], options._validate_helper())

        # These ones should not
        options = self.get_options(top_k_categories=6)
        self.assertEqual([], options._validate_helper())
        options = self.get_options(top_k_categories=None)
        self.assertEqual([], options._validate_helper())

        options = self.get_options(
            max_sample_size_to_check_stop_condition=20,
            stop_condition_unique_value_ratio=1.0,
        )
        self.assertEqual([], options._validate_helper())
        options = self.get_options(
            max_sample_size_to_check_stop_condition=20,
            stop_condition_unique_value_ratio=0.0,
        )
        self.assertEqual([], options._validate_helper())
        options = self.get_options(
            max_sample_size_to_check_stop_condition=0,
            stop_condition_unique_value_ratio=0.3,
        )
        self.assertEqual([], options._validate_helper())
        options = self.get_options(
            max_sample_size_to_check_stop_condition=None,
            stop_condition_unique_value_ratio=None,
        )
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

        expected = {
            "class": "CategoricalOptions",
            "data": {
                "cms": False,
                "cms_confidence": 0.95,
                "cms_max_num_heavy_hitters": 5000,
                "cms_relative_error": 0.01,
                "is_enabled": False,
                "max_sample_size_to_check_stop_condition": None,
                "stop_condition_unique_value_ratio": None,
                "top_k_categories": 5,
            },
        }

        self.assertDictEqual(expected, json.loads(serialized))
