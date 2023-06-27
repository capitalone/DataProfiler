import unittest
from collections import defaultdict
from unittest import mock

from dataprofiler import ProfilerOptions
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler
from dataprofiler.plugins.__init__ import getPlugins
from dataprofiler.plugins.decorators import plugin_decorator, plugins_dict


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

    def test_plugin_presets(self, *mocks):
        mock_plugin_execution = mock.Mock()
        with mock.patch.dict(plugins_dict, defaultdict(dict)) as mock_plugin_dict:

            @plugin_decorator(typ="test_preset", name="mock_test")
            def test_plugin():
                mock_plugin_execution()

            expected_default_dict = defaultdict(dict)
            expected_default_dict["test_preset"]["mock_test"] = test_plugin
            self.assertDictEqual(expected_default_dict, mock_plugin_dict)

            test_get_dict = getPlugins("test_preset")
            self.assertDictEqual({"mock_test": test_plugin}, test_get_dict)
