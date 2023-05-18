import unittest
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler import Data, Profiler, ProfilerOptions
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler
from dataprofiler.profilers.profiler_options import FloatOptions, IntOptions


@mock.patch(
    "dataprofiler.profilers.data_labeler_column_profile." "DataLabelerColumn.update",
    return_value=None,
)
@mock.patch("dataprofiler.profilers.profile_builder.DataLabeler", spec=BaseDataLabeler)
class TestProfilerOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(data=pd.DataFrame([1, 2]), data_type="csv")

    def test_default_profiler_options(self, *mocks):
        # Allowing Profiler to create default options
        profile = Profiler(self.data)
        self.assertIsNotNone(profile.options)
        self.assertTrue(profile.options.data_labeler.is_enabled)
        for column in profile.options.properties:
            # TODO: remove the check for correlation option once it's updated to True
            if column == "correlation" or column == "null_replication_metrics":
                self.assertFalse(profile.options.properties[column].is_enabled)
            elif column == "null_values" or column == "column_null_values":
                self.assertIsNone(profile.options.properties[column])
            elif column == "sampling_ratio":
                self.assertEqual(profile.options.properties[column], 0.2)
            else:
                self.assertTrue(profile.options.properties[column].is_enabled)

        for column_type in ["int", "float", "text"]:
            column = profile.options.properties[column_type]
            self.assertTrue(column.properties["histogram_and_quantiles"])
            self.assertTrue(column.properties["min"])
            self.assertTrue(column.properties["max"])
            self.assertTrue(column.properties["sum"])
            self.assertTrue(column.properties["variance"])
            self.assertTrue(column.properties["is_numeric_stats_enabled"])
            if column_type != "text":
                self.assertTrue(column.properties["num_zeros"].is_enabled)
                self.assertTrue(column.properties["num_negatives"].is_enabled)
            else:
                self.assertFalse(column.properties["num_zeros"].is_enabled)
                self.assertFalse(column.properties["num_negatives"].is_enabled)
        # Using ProfilerOptions with the default options
        options = ProfilerOptions()
        profile2 = Profiler(self.data, options=options)
        # Stored in Profiler as StructuredOptions
        self.assertEqual(profile2.options, options.structured_options)

    def test_set_failures(self, *mocks):

        options = ProfilerOptions()

        # check if no '*.' it raises an error bc attribute not found
        expected_error = (
            "type object 'ProfilerOptions' has no attribute " "'is_enabled'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            options.set({"is_enabled": False})

        # check if attribute doesn't exist, it raises an error
        expected_error = "type object 'structured_options' has no attribute " "'test'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            options.set({"structured_options.test": False})

    def test_numerical_stats_option(self, *mocks):
        # Assert that the stats are disabled
        options = ProfilerOptions()
        options.set(
            {"*.is_numeric_stats_enabled": False, "bias_correction.is_enabled": False}
        )
        profile = Profiler(self.data, options=options)

        for col_profiler in profile.profile:
            profile_column = col_profiler.profile
            if (
                profile_column["statistics"]
                and "histogram" in profile_column["statistics"].keys()
                and profile_column["statistics"]["histogram"]
            ):
                self.assertIsNone(
                    profile_column["statistics"]["histogram"]["bin_counts"]
                )
                self.assertIsNone(
                    profile_column["statistics"]["histogram"]["bin_edges"]
                )
                self.assertIsNone(profile_column["statistics"]["min"])
                self.assertIsNone(profile_column["statistics"]["max"])
                self.assertTrue(np.isnan(profile_column["statistics"]["variance"]))
                self.assertIsNone(profile_column["statistics"]["quantiles"][0])
                self.assertTrue(np.isnan(profile_column["statistics"]["skewness"]))
                self.assertTrue(np.isnan(profile_column["statistics"]["kurtosis"]))

        # Assert that the stats are enabled
        options.set(
            {"*.is_numeric_stats_enabled": True, "bias_correction.is_enabled": True}
        )
        profile = Profiler(self.data, options=options)

        for col_profiler in profile.profile:
            profile_column = col_profiler.profile
            if (
                profile_column["statistics"]
                and "histogram" in profile_column["statistics"].keys()
                and profile_column["statistics"]["histogram"]
            ):
                self.assertIsNotNone(
                    profile_column["statistics"]["histogram"]["bin_counts"]
                )
                self.assertIsNotNone(
                    profile_column["statistics"]["histogram"]["bin_edges"]
                )
                self.assertIsNotNone(profile_column["statistics"]["min"])
                self.assertIsNotNone(profile_column["statistics"]["max"])
                self.assertEqual(0.5, profile_column["statistics"]["variance"])
                self.assertIsNotNone(profile_column["statistics"]["quantiles"][0])
                self.assertTrue(profile_column["statistics"]["skewness"] is np.nan)
                self.assertTrue(profile_column["statistics"]["kurtosis"] is np.nan)

    def test_disable_labeler_in_profiler_options(self, *mocks):
        options = ProfilerOptions()
        options.structured_options.data_labeler.enable = False
        profile = Profiler(self.data, options=options)
        for col_profiler in profile.profile:
            profile_column = col_profiler.profile
            if (
                profile_column["statistics"]
                and "data_label_probability" in profile_column["statistics"].keys()
            ):
                self.assertIsNone(
                    profile_column["statistics"]["data_label_probability"]
                )

    def test_disabling_all_columns(self, *mocks):
        options = ProfilerOptions()
        options.structured_options.text.is_enabled = False
        options.structured_options.float.is_enabled = False
        options.structured_options.int.is_enabled = False
        options.structured_options.datetime.is_enabled = False
        options.structured_options.order.is_enabled = False
        options.structured_options.category.is_enabled = False
        options.structured_options.chi2_homogeneity.is_enabled = False
        options.structured_options.data_labeler.is_enabled = False
        profile = Profiler(self.data, options=options)
        for col_profiler in profile.profile:
            profile_column = col_profiler.profile
            self.assertIsNone(profile_column["data_type"])
            self.assertTrue("data_label" not in profile_column.keys())
            self.assertIsNone(profile_column["categorical"])
            self.assertIsNone(profile_column["order"])
            self.assertDictEqual(
                {
                    "sample_size": 2,
                    "null_count": 0,
                    "null_types": [],
                    "null_types_index": {},
                },
                profile_column["statistics"],
            )

    @mock.patch(
        "dataprofiler.profilers.text_column_profile.TextColumn" "._update_vocab"
    )
    def test_disabling_vocab(self, vocab_mock, *mocks):
        # Check to see disabling vocab prevents vocab from updating
        options = ProfilerOptions()
        options.structured_options.text.vocab.is_enabled = False
        profile = Profiler(self.data, options=options)
        vocab_mock.assert_not_called()

        # Check to see default options enable vocab
        multi_options = ProfilerOptions()
        multi_options.structured_options.multiprocess.is_enabled = False
        profile = Profiler(self.data, options=multi_options)
        vocab_mock.assert_called()

    def test_disabling_all_stats(self, *mocks):
        options = ProfilerOptions()
        statistical_options = {
            "histogram_and_quantiles.is_enabled": False,
            "min.is_enabled": False,
            "max.is_enabled": False,
            "mode.is_enabled": False,
            "median.is_enabled": False,
            "sum.is_enabled": False,
            "variance.is_enabled": False,
            "skewness.is_enabled": False,
            "kurtosis.is_enabled": False,
            "num_zeros.is_enabled": False,
            "num_negatives.is_enabled": False,
            "median_abs_deviation.is_enabled": False,
        }
        options.set(statistical_options)

        # Assert the numerics are properly set
        text_options = options.structured_options.text.properties
        float_options = options.structured_options.float.properties
        int_options = options.structured_options.int.properties
        for option in [
            "histogram_and_quantiles",
            "min",
            "max",
            "sum",
            "mode",
            "variance",
            "skewness",
            "kurtosis",
            "median_abs_deviation",
            "num_zeros",
            "num_negatives",
        ]:
            self.assertFalse(text_options[option].is_enabled)
            self.assertFalse(float_options[option].is_enabled)
            self.assertFalse(int_options[option].is_enabled)

        # Run the profiler
        profile = Profiler(self.data, options=options)

        # Assert that the stats are non-existent
        for col_profiler in profile.profile:
            profile_column = col_profiler.profile
            if (
                profile_column["statistics"]
                and "histogram" in profile_column["statistics"].keys()
                and profile_column["statistics"]["histogram"]
            ):
                self.assertIsNone(
                    profile_column["statistics"]["histogram"]["bin_counts"]
                )
                self.assertIsNone(
                    profile_column["statistics"]["histogram"]["bin_edges"]
                )
                self.assertIsNone(profile_column["statistics"]["min"])
                self.assertIsNone(profile_column["statistics"]["max"])
                self.assertTrue(np.isnan(profile_column["statistics"]["variance"]))
                self.assertIsNone(profile_column["statistics"]["quantiles"][0])
                self.assertTrue(profile_column["statistics"]["skewness"] is np.nan)
                self.assertTrue(profile_column["statistics"]["kurtosis"] is np.nan)
                self.assertTrue(
                    profile_column["statistics"]["median_abs_deviation"] is np.nan
                )
                self.assertTrue(np.isnan(profile_column["statistics"]["mode"]))
                self.assertTrue(np.isnan(profile_column["statistics"]["median"]))

    def test_validate(self, *mocks):
        options = ProfilerOptions()

        options.structured_options.data_labeler.is_enabled = "Invalid"
        options.structured_options.data_labeler.data_labeler_dirpath = 5
        options.structured_options.int.max = "Invalid"

        expected_error = (
            "ProfilerOptions.structured_options.int.max must be a "
            "BooleanOption.\n"
            "ProfilerOptions.structured_options.data_labeler.is_enabled must be"
            " a Boolean.\n"
            "ProfilerOptions.structured_options.data_labeler."
            "data_labeler_dirpath must be a string."
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate()

    def test_validate_numeric_stats(self, *mocks):
        options = ProfilerOptions()
        numerical_options = {
            "histogram_and_quantiles.is_enabled": False,
            "min.is_enabled": False,
            "max.is_enabled": False,
            "mode.is_enabled": False,
            "median.is_enabled": False,
            "sum.is_enabled": False,
            "variance.is_enabled": True,
            "skewness.is_enabled": False,
            "kurtosis.is_enabled": False,
            "median_abs_deviation.is_enabled": False,
        }
        # Asserts error since sum must be toggled on if variance is
        expected_error = (
            "ProfilerOptions.structured_options.int: The numeric stats must "
            "toggle on the sum if the variance is toggled on.\n"
            "ProfilerOptions.structured_options.float: The numeric stats must "
            "toggle on the sum if the variance is toggled on.\n"
            "ProfilerOptions.structured_options.text: The numeric stats must "
            "toggle on the sum if the variance is toggled on."
        )
        options.set(numerical_options)

        with self.assertRaisesRegex(ValueError, expected_error):
            options.validate()

        # test warns if is_numeric_stats_enabled = False
        numerical_options = {
            "*.is_numeric_stats_enabled": False,
        }
        options.set(numerical_options)
        with self.assertWarnsRegex(
            UserWarning,
            "ProfilerOptions.structured_options.int."
            "numeric_stats: The numeric stats are "
            "completely disabled.",
        ):
            options.validate()

    def test_setting_options(self, *mocks):
        options = ProfilerOptions()

        # Ensure set works appropriately
        options.set(
            {
                "data_labeler.is_enabled": False,
                "min.is_enabled": False,
                "structured_options.data_labeler.data_labeler_dirpath": "test",
                "data_labeler.max_sample_size": 15,
            }
        )

        text_options = options.structured_options.text.properties
        float_options = options.structured_options.float.properties
        int_options = options.structured_options.int.properties
        data_labeler_options = options.structured_options.data_labeler.properties

        self.assertFalse(options.structured_options.data_labeler.is_enabled)
        self.assertFalse(text_options["min"].is_enabled)
        self.assertFalse(float_options["min"].is_enabled)
        self.assertFalse(int_options["min"].is_enabled)
        self.assertEqual(data_labeler_options["data_labeler_dirpath"], "test")
        self.assertEqual(data_labeler_options["max_sample_size"], 15)

        # Ensure direct attribute setting works appropriately
        options.structured_options.data_labeler.max_sample_size = 12
        options.structured_options.text.histogram_and_quantiles.is_enabled = True
        options.structured_options.text.is_enabled = False

        text_options = options.structured_options.text.properties
        data_labeler_options = options.structured_options.data_labeler.properties
        self.assertEqual(data_labeler_options["max_sample_size"], 12)
        self.assertTrue(text_options["histogram_and_quantiles"].is_enabled)
        self.assertFalse(text_options["is_enabled"])

        # check direct attribute access after set
        float_options = FloatOptions()
        float_options.set(
            {
                "precision.is_enabled": False,
                "min.is_enabled": False,
                "*.is_enabled": False,
            }
        )

        self.assertFalse(float_options.precision.is_enabled)
        self.assertFalse(float_options.min.is_enabled)
        self.assertFalse(float_options.is_enabled)

    def test_improper_profile_options(self, *mocks):
        with self.assertRaisesRegex(
            ValueError,
            "The profile options must be passed as a " "ProfileOptions object.",
        ):
            profile = Profiler(self.data, options="Strings are not accepted")

        with self.assertRaisesRegex(
            ValueError,
            "ProfilerOptions.structured_options.text.max."
            "is_enabled must be a Boolean.",
        ):
            profile_options = ProfilerOptions()
            profile_options.structured_options.text.max.is_enabled = "String"
            profile_options.validate()

    def test_invalid_options_type(self, *mocks):
        # Test incorrect data labeler options
        options = ProfilerOptions()
        options.structured_options.data_labeler = IntOptions()
        with self.assertRaisesRegex(
            ValueError, r"data_labeler must be a\(n\) DataLabelerOptions."
        ):
            profile = Profiler(self.data, options=options)
        # Test incorrect float options
        options = ProfilerOptions()
        options.structured_options.float = IntOptions()
        with self.assertRaisesRegex(ValueError, r"float must be a\(n\) FloatOptions."):
            profile = Profiler(self.data, options=options)

    @mock.patch(
        "dataprofiler.profilers.float_column_profile.FloatColumn." "_update_precision"
    )
    def test_float_precision(self, update_precision, *mocks):
        options = ProfilerOptions()
        options.structured_options.float.precision.is_enabled = False
        options.structured_options.multiprocess.is_enabled = False

        profile = Profiler(self.data, options=options)
        update_precision.assert_not_called()

        multi_options = ProfilerOptions()
        multi_options.structured_options.multiprocess.is_enabled = False
        profile = Profiler(self.data, options=multi_options)
        update_precision.assert_called()

    def test_set_attribute_error(self, *mocks):
        options = ProfilerOptions()

        with self.assertRaisesRegex(
            AttributeError,
            "type object 'structured_options."
            "data_labeler.is_enabled' has no attribute"
            " 'is_here'",
        ):
            options.set({"data_labeler.is_enabled.is_here": False})

    def test_is_prop_enabled(self, *mocks):
        options = ProfilerOptions()
        with self.assertRaisesRegex(
            AttributeError, 'Property "Invalid" does not exist in ' "TextOptions."
        ):
            options.structured_options.text.is_prop_enabled("Invalid")

        # This test is to ensure is_prop_enabled works for BooleanOption objects
        options.structured_options.int.min.is_enabled = True
        self.assertTrue(options.structured_options.int.is_prop_enabled("min"))

        # This test is to ensure is_prop_enabled works for bools
        options.structured_options.int.max.is_enabled = True
        options.structured_options.int.variance.is_enabled = True
        options.structured_options.int.histogram_and_quantiles.is_enabled = True
        options.structured_options.int.sum.is_enabled = True
        self.assertTrue(
            options.structured_options.int.is_prop_enabled("is_numeric_stats_enabled")
        )

    def test_setting_overlapping_option(self, *mocks):
        options = ProfilerOptions()

        # Raises error if overlap option set
        sets = [
            {"data_labeler.data_labeler_object": 3},
            {"data_labeler.data_labeler_dirpath": 3},
            {"text.is_enabled": False},
            {"text.vocab.is_enabled": False},
        ]

        for set_dict in sets:
            msg = (
                f"Attempted to set options {set_dict} in ProfilerOptions "
                f"without specifying whether to set them for "
                f"StructuredOptions or UnstructuredOptions."
            )
            with self.assertRaisesRegex(ValueError, msg):
                options.set(set_dict)

        # Works still for disabling both labelers
        options.set({"data_labeler.is_enabled": False})
        self.assertFalse(options.structured_options.data_labeler.is_enabled)
        self.assertFalse(options.unstructured_options.data_labeler.is_enabled)

        # Works for specifying which to set for
        options = ProfilerOptions()
        options.set({"unstructured_options.data_labeler.data_labeler_object": 23})
        self.assertIsNone(options.structured_options.data_labeler.data_labeler_object)
        self.assertEqual(
            23, options.unstructured_options.data_labeler.data_labeler_object
        )

        options = ProfilerOptions()
        options.set(
            {"structured_options.data_labeler.data_labeler_dirpath": "Hello There"}
        )
        self.assertEqual(
            "Hello There", options.structured_options.data_labeler.data_labeler_dirpath
        )
        self.assertIsNone(
            options.unstructured_options.data_labeler.data_labeler_dirpath
        )

        options = ProfilerOptions()
        options.set({"unstructured_options.text.is_enabled": False})
        self.assertTrue(options.structured_options.text.is_enabled)
        self.assertFalse(options.unstructured_options.text.is_enabled)

        options = ProfilerOptions()
        options.set({"structured_options.text.vocab.is_enabled": False})
        self.assertFalse(options.structured_options.text.vocab.is_enabled)
        self.assertTrue(options.unstructured_options.text.vocab.is_enabled)

    def test_eq(self, *mocks):
        options = ProfilerOptions()
        options2 = ProfilerOptions()
        options.unstructured_options.data_labeler.is_enabled = False
        self.assertNotEqual(options, options2)
        options2.unstructured_options.data_labeler.is_enabled = False
        self.assertEqual(options, options2)

        options.structured_options.float.precision.sample_ratio = 0.1
        self.assertNotEqual(options, options2)
        options2.structured_options.float.precision.sample_ratio = 0.15
        self.assertNotEqual(options, options2)
        options.structured_options.float.precision.sample_ratio = 0.1
        self.assertNotEqual(options, options2)


@mock.patch(
    "dataprofiler.profilers.data_labeler_column_profile." "DataLabelerColumn.update",
    return_value=None,
)
@mock.patch("dataprofiler.profilers.profile_builder.DataLabeler")
class TestDataLabelerCallWithOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(data=pd.DataFrame([1, 2]), data_type="csv")

    def test_data_labeler(self, *mocks):
        options = ProfilerOptions()
        options.structured_options.data_labeler.data_labeler_dirpath = "Test_Dirpath"
        options.structured_options.data_labeler.max_sample_size = 50
        options.structured_options.multiprocess.is_enabled = False

        profile = Profiler(self.data, options=options)

        # Mock[0] is the Datalabeler Object mock
        mocks[0].assert_called_with(
            dirpath="Test_Dirpath", labeler_type="structured", load_options=None
        )
        actual_sample_size = (
            profile._profile[0]
            .profiles["data_label_profile"]
            ._profiles["data_labeler"]
            ._max_sample_size
        )
        self.assertEqual(actual_sample_size, 50)

        data_labeler = mock.Mock(spec=BaseDataLabeler)
        data_labeler.reverse_label_mapping = dict()
        data_labeler.model.num_labels = 0
        data_labeler.model.requires_zero_mapping = False
        options.structured_options.set(
            {"data_labeler.data_labeler_object": data_labeler}
        )
        with self.assertWarnsRegex(
            UserWarning,
            "The data labeler passed in will be used,"
            " not through the directory of the default"
            " model",
        ):
            options.validate()

        profile = Profiler(self.data, options=options)
        self.assertEqual(
            data_labeler,
            # profile, col prof, compiler
            (
                profile._profile[0]
                .profiles["data_label_profile"]
                .
                # column profiler
                _profiles["data_labeler"]
                .data_labeler
            ),
        )
