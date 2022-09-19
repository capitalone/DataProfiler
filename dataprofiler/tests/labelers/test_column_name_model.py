import json
import os
import sys
import unittest
from io import StringIO
from unittest import mock

import numpy as np
import pkg_resources

import dataprofiler as dp
from dataprofiler.labelers.column_name_model import ColumnNameModel

_file_dir = os.path.dirname(os.path.abspath(__file__))
_resource_labeler_dir = pkg_resources.resource_filename("resources", "labelers")

mock_model_parameters = {
    "true_positive_dict": [
        {"attribute": "ssn", "label": "ssn"},
        {"attribute": "suffix", "label": "name"},
        {"attribute": "my_home_address", "label": "address"},
    ],
    "false_positive_dict": [
        {
            "attribute": "contract_number",
            "label": "ssn",
        },
        {
            "attribute": "role",
            "label": "name",
        },
        {
            "attribute": "send_address",
            "label": "address",
        },
    ],
    "negative_threshold_config": 50,
    "include_label": True,
}

mock_label_mapping = {"ssn": 1, "name": 2, "address": 3}


def mock_open(filename, *args):
    if filename.find("model_parameters") >= 0:
        return StringIO(json.dumps(mock_model_parameters))
    elif filename.find("label_mapping") >= 0:
        return StringIO(json.dumps(mock_label_mapping))


def setup_save_mock_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


class TestColumnNameModel(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # data
        cls.data = ["ssn", "role_name", "wallet_address"]

        cls.invalid_parameters = [
            {
                "false_positive_dict": [
                    {
                        "attribute": "test_attribute",
                        "label": "test_label",
                    }
                ],
                # fails, requires `true_positive_dict`
            },
            {
                "false_positive_dict": {},  # fails, required type list
                "true_positive_dict": [
                    {"attribute": "test_attribute", "label": "test_label"}
                ],
            },
        ]

        cls.parameters = {
            "true_positive_dict": [
                {"attribute": "ssn", "label": "ssn"},
                {"attribute": "suffix", "label": "name"},
                {"attribute": "my_home_address", "label": "address"},
            ],
            "false_positive_dict": [
                {
                    "attribute": "contract_number",
                    "label": "ssn",
                },
                {
                    "attribute": "role",
                    "label": "name",
                },
                {
                    "attribute": "send_address",
                    "label": "address",
                },
            ],
            "negative_threshold_config": 50,
            "include_label": True,
        }

        cls.test_label_mapping = {"ssn": 1, "name": 2, "address": 3}

    def test_param_validation(self):

        model = ColumnNameModel(
            label_mapping=self.test_label_mapping, parameters=mock_model_parameters
        )
        self.assertDictEqual(mock_model_parameters, model._parameters)

        for invalid_param_set in self.invalid_parameters:
            with self.assertRaises(ValueError):
                ColumnNameModel(
                    label_mapping=self.test_label_mapping, parameters=invalid_param_set
                )

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_help(self, mock_stdout):
        ColumnNameModel.help()
        self.assertIn("ColumnNameModel", mock_stdout.getvalue())
        self.assertIn("Parameters", mock_stdout.getvalue())

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_predict(self, mock_stdout):
        # test show confidences
        model = ColumnNameModel(
            label_mapping=self.test_label_mapping, parameters=mock_model_parameters
        )
        expected_output = [[100.0, 0]]
        with self.assertLogs(
            "DataProfiler.labelers.column_name_model", level="INFO"
        ) as logs:
            model_output = model.predict(data=self.data)
        self.assertTrue(np.array_equal(expected_output, model_output))
        self.assertTrue(len(logs.output))

        # `show_confidences` is disabled currently
        # should raise error if set to `True`
        with self.assertRaisesRegex(
            Warning,
            """`show_confidences` parameter is disabled
                for MVP implementation. Due to the requirement
                of having the data point in the post processor.
                Note: Confidence values are returned by default.""",
        ):
            model.predict(data=self.data, show_confidences=True)

        # clear stdout
        mock_stdout.seek(0)
        mock_stdout.truncate(0)

        # test verbose = False
        # Want to ensure no INFO logged
        with self.assertRaisesRegex(
            AssertionError,
            "no logs of level INFO or higher triggered "
            "on DataProfiler.labelers.column_name_model",
        ):
            with self.assertLogs(
                "DataProfiler.labelers.column_name_model", level="INFO"
            ):
                model.predict(["test_verbose_param"], verbose=False)

        # Not in stdout
        self.assertNotIn("Data Samples", mock_stdout.getvalue())

    @mock.patch("builtins.open", side_effect=mock_open)
    def test_save(self, mock_open, *mocks):

        # setup mock
        mock_file = setup_save_mock_open(mock_open)

        model = ColumnNameModel(
            label_mapping=mock_model_parameters, parameters=self.parameters
        )

        model.save_to_disk(".")
        self.assertEqual(
            '{"true_positive_dict": [{"attribute": "ssn", "label": "ssn"}, '
            '{"attribute": "suffix", "label": "name"}, {"attribute": "my_home_address", '
            '"label": "address"}], "false_positive_dict": [{"attribute": '
            '"contract_number", "label": "ssn"}, {"attribute": "role", '
            '"label": "name"}, {"attribute": "send_address", "label": "address"}], '
            '"negative_threshold_config": 50, "include_label": true}{"true_positive_dict": '
            '[{"attribute": "ssn", "label": "ssn"}, {"attribute": "suffix", '
            '"label": "name"}, {"attribute": "my_home_address", "label": "address"}], '
            '"false_positive_dict": [{"attribute": "contract_number", "label": "ssn"}, '
            '{"attribute": "role", "label": "name"}, {"attribute": "send_address", "label": '
            '"address"}], "negative_threshold_config": 50, "include_label": true}',
            mock_file.getvalue(),
        )

        # close mock
        StringIO.close(mock_file)

    @mock.patch("builtins.open", side_effect=mock_open)
    def test_load(self, *mocks):
        dir = os.path.join(_resource_labeler_dir, "column_name_model")
        loaded_model = ColumnNameModel.load_from_disk(dir)
        self.assertIsInstance(loaded_model, ColumnNameModel)

        self.assertEqual(
            mock_model_parameters["true_positive_dict"],
            loaded_model._parameters["true_positive_dict"],
        )
        self.assertEqual(
            mock_model_parameters["false_positive_dict"],
            loaded_model._parameters["false_positive_dict"],
        )
        self.assertEqual(
            mock_model_parameters["include_label"],
            loaded_model._parameters["include_label"],
        )
        self.assertEqual(
            mock_model_parameters["negative_threshold_config"],
            loaded_model._parameters["negative_threshold_config"],
        )

    def test_reverse_label_mapping(self):
        """test reverse label mapping is propograting
        through the classes correctly"""
        reverse_label_mapping = {v: k for k, v in self.test_label_mapping.items()}
        model = ColumnNameModel(
            label_mapping=self.test_label_mapping, parameters=self.parameters
        )
        self.assertEqual(model.reverse_label_mapping, reverse_label_mapping)

    def missing_module_test(self, class_name, module_name):
        orig_import = __import__

        # necessary for any wrapper around the library to test if
        # the code catches the needed import in the predict method

        def import_mock(name, *args, **kwargs):
            if name.startswith(module_name):
                raise ImportError("test")
            return orig_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_mock):
            with self.assertRaises(TypeError):
                modules_to_remove = [
                    "dataprofiler.labelers.column_name_model",
                    module_name,
                ]

                for module in modules_to_remove:
                    if module in sys.modules:
                        del sys.modules[module]

                # re-add module for testing
                for module in modules_to_remove[:-1]:
                    import importlib

                    importlib.import_module(module)

                class_name(parameters=mock_model_parameters).predict(data=["ssn"])

    def test_no_rapidfuzz(self):
        self.missing_module_test(
            dp.labelers.column_name_model.ColumnNameModel, "rapidfuzz"
        )


if __name__ == "__main__":
    unittest.main()
