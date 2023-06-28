import unittest
from collections import defaultdict
from unittest import mock

from dataprofiler.plugins.__init__ import get_plugins
from dataprofiler.plugins.decorators import plugin_decorator, plugins_dict


class TestPlugins(unittest.TestCase):
    def test_decorator_get_plugin(self, *mocks):
        mock_plugin_execution = mock.Mock()
        with mock.patch.dict(plugins_dict, defaultdict(dict)) as mock_plugin_dict:

            @plugin_decorator(typ="test", name="mock_test")
            def test_plugin():
                mock_plugin_execution()

            expected_default_dict = defaultdict(dict)
            expected_default_dict["test"]["mock_test"] = test_plugin
            self.assertDictEqual(expected_default_dict, mock_plugin_dict)

            test_get_dict = get_plugins("test")
            self.assertDictEqual({"mock_test": test_plugin}, test_get_dict)

    @mock.patch("..plugins.__init__.load_plugins")
    def test_load_plugin(self, *mocks):
        return None
