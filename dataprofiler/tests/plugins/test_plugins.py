import unittest
from collections import defaultdict
from unittest import mock

from dataprofiler.plugins.__init__ import get_plugins, load_plugins
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
            self.assertDictEqual(
                expected_default_dict["test"], mock_plugin_dict["test"]
            )

            test_get_dict = get_plugins("test")
            self.assertDictEqual({"mock_test": test_plugin}, test_get_dict)

    @mock.patch("dataprofiler.plugins.__init__.importlib.util")
    @mock.patch("dataprofiler.plugins.__init__.os.path.isdir")
    @mock.patch("dataprofiler.plugins.__init__.os.listdir")
    def test_load_plugin(self, mock_listdir, mock_isdir, mock_importlib_util):
        mock_listdir.side_effect = (
            lambda folder_dir: ["__pycache__", "py"]
            if folder_dir.endswith("plugins")
            else ["stillnotrealpy", "a.json", None]
        )
        mock_isdir.return_value = True
        mock_importlib_util.spec_from_file_location.return_value = None
        load_plugins()
        mock_importlib_util.spec_from_file_location.assert_not_called()

        mock_listdir.side_effect = (
            lambda folder_dir: ["folder"]
            if folder_dir.endswith("plugins")
            else ["file.py"]
        )
        mock_spec = mock.Mock()
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        load_plugins()
        mock_importlib_util.module_from_spec.assert_called_with(mock_spec)
