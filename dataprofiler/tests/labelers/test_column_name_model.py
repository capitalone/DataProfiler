import json
import os
import unittest
from io import StringIO
from unittest import mock

import numpy as np
import pkg_resources

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
}


def mock_open(filename, *args):
    if filename.find("model_parameters") >= 0:
        return StringIO(json.dumps(mock_model_parameters))


def setup_save_mock_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


class TestColumnNameModel(unittest.TestCase):
    def setUp(self):
        # data
        data = [
            "ssn",
            "role_name",
            "wallet_address",
        ]

    def test_param_validation(self):
        parameters = {
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
            "true_positive_dict": [
                {"attribute": "ssn", "label": "ssn"},
                {"attribute": "suffix", "label": "name"},
                {"attribute": "my_home_address", "label": "address"},
            ],
        }
        invalid_parameters = [
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

        model = ColumnNameModel(parameters=parameters)
        self.assertDictEqual(parameters, model._parameters)

        for invalid_param_set in invalid_parameters:
            with self.assertRaises(ValueError):
                ColumnNameModel(parameters=invalid_param_set)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_help(self, mock_stdout):
        ColumnNameModel.help()
        self.assertIn("ColumnNameModel", mock_stdout.getvalue())
        self.assertIn("Parameters", mock_stdout.getvalue())

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_predict(self, mock_stdout):
        parameters = {
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
        }
        model = ColumnNameModel(parameters=parameters)

        expected_output = {
            "ssn": {
                "pred": "ssn",
            }
        }

        with self.assertLogs(
            "DataProfiler.labelers.column_name_model", level="INFO"
        ) as logs:
            model_output = model.predict(data=["ssn", "role_name", "wallet_address"])

        self.assertIn("pred", model_output["ssn"])
        self.assertTrue(np.array_equal(expected_output, model_output))

        self.assertTrue(len(logs.output))

        # test show confidences
        expected_output = {"ssn": {"pred": "ssn", "conf": 100.0}}
        model_output = model.predict(
            data=["ssn", "role_name", "wallet_address"], show_confidences=True
        )
        self.assertTrue(np.array_equal(expected_output, model_output))

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

        # Save and load a Model with custom parameters
        parameters = {
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
        }

        model = ColumnNameModel(parameters)

        model.save_to_disk(".")
        self.assertDictEqual(
            parameters,
            json.loads(mock_file.getvalue()),
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


if __name__ == "__main__":
    unittest.main()
