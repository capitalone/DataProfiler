import unittest
from collections import defaultdict
from unittest import mock

from dataprofiler.plugins.__init__ import get_plugins
from dataprofiler.plugins.decorators import plugin_decorator, plugins_dict


class TestPlugins(unittest.TestCase):
    def test_plugin_presets(self, *mocks):
        mock_plugin_execution = mock.Mock()
        with mock.patch.dict(plugins_dict, defaultdict(dict)) as mock_plugin_dict:

            @plugin_decorator(typ="test_preset", name="mock_test")
            def test_plugin():
                mock_plugin_execution()

            expected_default_dict = defaultdict(dict)
            expected_default_dict["test_preset"]["mock_test"] = test_plugin
            self.assertDictEqual(expected_default_dict, mock_plugin_dict)

            test_get_dict = get_plugins("test_preset")
            self.assertDictEqual({"mock_test": test_plugin}, test_get_dict)
