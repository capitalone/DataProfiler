import unittest
from unittest import mock

from dataprofiler import ProfilerOptions
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler


@mock.patch(
    "dataprofiler.profilers.data_labeler_column_profile." "DataLabelerColumn.update",
    return_value=None,
)
@mock.patch("dataprofiler.profilers.profile_builder.DataLabeler", spec=BaseDataLabeler)
class TestProfilerPresets(unittest.TestCase):
    def test_profiler_preset_complete(self, *mocks):
        options = ProfilerOptions(presets="complete")
        self.assertTrue(options.structured_options.correlation)
        self.assertTrue(options.structured_options.null_replication_metrics.is_enabled)

    def test_profiler_preset_data_types(self, *mocks):
        options = ProfilerOptions(presets="data_types")
        self.assertTrue(options.unstructured_options.data_labeler.is_enabled)
        self.assertTrue(options.structured_options.data_labeler.is_enabled)
        self.assertFalse(options.structured_options.correlation.is_enabled)
        self.assertFalse(options.structured_options.null_replication_metrics.is_enabled)

    def test_profiler_preset_numeric_stats_disabled(self, *mocks):
        options = ProfilerOptions(presets="numeric_stats_disabled")
        self.assertTrue(options.structured_options.data_labeler.is_enabled)
        self.assertFalse(options.structured_options.int.is_numeric_stats_enabled)
        self.assertFalse(options.structured_options.text.is_numeric_stats_enabled)
        self.assertFalse(options.structured_options.float.is_numeric_stats_enabled)
        self.assertFalse(options.structured_options.correlation.is_enabled)
        self.assertFalse(options.structured_options.null_replication_metrics.is_enabled)
        self.assertTrue(options.structured_options.category.is_enabled)
        self.assertTrue(options.structured_options.order.is_enabled)

    def test_profiler_preset_lower_memory_sketching(self, *mocks):
        options = ProfilerOptions(presets="lower_memory_sketching")
        self.assertEqual(
            options.structured_options.row_statistics.unique_count.hashing_method, "hll"
        )
        self.assertEqual(
            options.structured_options.category.max_sample_size_to_check_stop_condition,
            5000,
        )
        self.assertEqual(
            options.structured_options.category.stop_condition_unique_value_ratio, 0.5
        )

    def test_profiler_preset_failure(self, *mocks):
        expected_error = "The preset entered is not a valid preset."
        with self.assertRaisesRegex(ValueError, expected_error):
            ProfilerOptions(presets="failing_preset")
