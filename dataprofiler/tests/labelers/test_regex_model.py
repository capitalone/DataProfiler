import json
import os
import unittest
from io import StringIO
from unittest import mock

import numpy as np
import pkg_resources

from dataprofiler.labelers.regex_model import RegexModel

_file_dir = os.path.dirname(os.path.abspath(__file__))
_resource_labeler_dir = pkg_resources.resource_filename("resources", "labelers")


mock_model_parameters = {
    "regex_patterns": {"PAD": [r"\W"], "UNKNOWN": [".*"]},
    "encapsulators": {
        "start": r"(?<![\w.\$\%\-])",
        "end": r"(?:(?=(\b|[ ]))|(?=[^\w\%\$]([^\w]|$))|$)",
    },
    "ignore_case": True,
    "default_label": "UNKNOWN",
}


mock_label_mapping = {
    "PAD": 0,
    "CITY": 1,  # ensure that overlapping labels get removed.
    "UNKNOWN": 1,
    "ADDRESS": 2,
}


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


class TestRegexModel(unittest.TestCase):
    def setUp(self):
        # data
        data = [
            "this is my test sentence",
            "this also is a test sentence.",
        ]

        self.label_mapping = {
            "PAD": 0,
            "CITY": 1,  # ensure that overlapping labels get removed.
            "UNKNOWN": 1,
            "ADDRESS": 2,
        }

    def test_label_mapping(self, *mocks):

        # load default
        model = RegexModel(self.label_mapping)

        self.assertDictEqual(self.label_mapping, model.label_mapping)

    def test_labels(self, *mocks):

        # load default
        model = RegexModel(self.label_mapping)

        labels = ["PAD", "UNKNOWN", "ADDRESS"]

        self.assertListEqual(labels, model.labels)

    def test_reverse_label_mapping(self, *mocks):

        # load default
        model = RegexModel(self.label_mapping)

        # should notice that CITY does not exist in reverse
        reverse_label_mapping = {0: "PAD", 1: "UNKNOWN", 2: "ADDRESS"}

        self.assertDictEqual(reverse_label_mapping, model.reverse_label_mapping)

    def test_set_label_mapping(self, *mocks):

        # load default
        model = RegexModel(self.label_mapping)

        # test not dict
        label_mapping = None
        with self.assertRaisesRegex(
            TypeError,
            "Labels must either be a non-empty encoding dict "
            "which maps labels to index encodings or a list.",
        ):
            model.set_label_mapping(label_mapping)

        # test label_mapping
        label_mapping = {
            "PAD": 0,
            "CITY": 1,  # SAME AS UNKNOWN
            "UNKNOWN": 1,
            "ADDRESS": 2,
        }
        model.set_label_mapping(label_mapping)
        self.assertDictEqual(label_mapping, model.label_mapping)

    def test_param_validation(self):
        # Make sure all parameters can be altered. Make sure non-valid params
        # are caught
        parameters = {
            "regex_patterns": {"PAD": [r"\W"], "UNKNOWN": [r"\w"]},
            "encapsulators": {
                "start": r"(?<![\w.\$\%\-])",
                "end": r"(?:(?=(\b|[ ]))|(?=[^\w\%\$]([^\w]|$))|$)",
            },
            "ignore_case": True,
            "default_label": "UNKNOWN",
        }
        invalid_parameters = [
            {
                "regex_patterns": -1,
                "encapsulators": "words",
                "default_label": None,
                "ignore_case": None,
            },
            {
                "regex_patterns": 1,
                "encapsulators": None,
                "default_label": [],
                "ignore_case": "true",
            },
            {
                "regex_patterns": 1,
                "encapsulators": tuple(),
                "default_label": -1,
                "ignore_case": 2,
            },
            {
                "regex_patterns": None,
                "encapsulators": 3,
                "default_label": 1.2,
                "ignore_case": -1,
            },
            {
                "regex_patterns": 2.2,
                "encapsulators": 3,
                "default_label": None,
                "ignore_case": {},
            },
        ]
        model = RegexModel(label_mapping=self.label_mapping, parameters=parameters)
        self.assertDictEqual(parameters, model._parameters)

        for invalid_param_set in invalid_parameters:
            with self.assertRaises(ValueError):
                RegexModel(
                    label_mapping=self.label_mapping, parameters=invalid_param_set
                )

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_help(self, mock_stdout):
        RegexModel.help()
        self.assertIn("RegexModel", mock_stdout.getvalue())
        self.assertIn("Parameters", mock_stdout.getvalue())

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_predict(self, mock_stdout):
        parameters = {
            "regex_patterns": {"PAD": [r"\W"], "UNKNOWN": [r"\w"]},
            "ignore_case": True,
            "default_label": "UNKNOWN",
        }
        model = RegexModel(label_mapping=self.label_mapping, parameters=parameters)

        # test only pad and background separate
        expected_output = {
            "pred": [
                np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            ]
        }
        with self.assertLogs("DataProfiler.labelers.regex_model", level="INFO") as logs:
            model_output = model.predict(["   ", "hello"])
        self.assertIn("pred", model_output)
        for expected, output in zip(expected_output["pred"], model_output["pred"]):
            self.assertTrue(np.array_equal(expected, output))

        # check verbose printing
        self.assertIn("Data Samples", mock_stdout.getvalue())
        # check verbose logging
        self.assertTrue(len(logs.output))

        # test pad with background
        expected_output = {
            "pred": [np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]])]
        }
        model_output = model.predict([" h w."])
        self.assertIn("pred", model_output)
        for expected, output in zip(expected_output["pred"], model_output["pred"]):
            self.assertTrue(np.array_equal(expected, output))

        # test show confidences
        expected_output = {
            "pred": [np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]])],
            "conf": [np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]])],
        }
        model_output = model.predict([" h w."], show_confidences=True)
        self.assertIn("pred", model_output)
        self.assertIn("conf", model_output)
        for expected, output in zip(expected_output["pred"], model_output["pred"]):
            self.assertTrue(np.array_equal(expected, output))
        for expected, output in zip(expected_output["conf"], model_output["conf"]):
            self.assertTrue(np.array_equal(expected, output))

        # clear stdout
        mock_stdout.seek(0)
        mock_stdout.truncate(0)

        # test verbose = False
        # Want to ensure no INFO logged
        with self.assertRaisesRegex(
            AssertionError,
            "no logs of level INFO or higher triggered "
            "on DataProfiler.labelers.regex_model",
        ):
            with self.assertLogs("DataProfiler.labelers.regex_model", level="INFO"):
                model.predict(["hello world."], verbose=False)

        # Not in stdout
        self.assertNotIn("Data Samples", mock_stdout.getvalue())

    @mock.patch("tensorflow.keras.models.load_model", return_value=None)
    @mock.patch("builtins.open", side_effect=mock_open)
    def test_save(self, mock_open, *mocks):

        # setup mock
        mock_file = setup_save_mock_open(mock_open)

        # Save and load a Model with custom parameters
        parameters = {
            "regex_patterns": {"PAD": [r"\W"], "UNKNOWN": [r"\w"]},
            "encapsulators": {
                "start": r"(?<![\w.\$\%\-])",
                "end": r"(?:(?=(\b|[ ]))|(?=[^\w\%\$]([^\w]|$))|$)",
            },
            "ignore_case": True,
            "default_label": "UNKNOWN",
        }
        label_mapping = {
            "PAD": 0,
            "CITY": 1,  # SAME AS UNKNOWN
            "UNKNOWN": 1,
            "ADDRESS": 2,
        }
        model = RegexModel(label_mapping, parameters)

        # save and test
        model.save_to_disk(".")
        self.assertEqual(
            # model parameters
            '{"regex_patterns": {"PAD": ["\\\\W"], "UNKNOWN": ["\\\\w"]}, '
            '"encapsulators": {"start": "(?<![\\\\w.\\\\$\\\\%\\\\-])", '
            '"end": '
            '"(?:(?=(\\\\b|[ ]))|(?=[^\\\\w\\\\%\\\\$]([^\\\\w]|$))|$)"}, '
            '"ignore_case": true, "default_label": "UNKNOWN"}'
            # label mapping
            '{"PAD": 0, "CITY": 1, "UNKNOWN": 1, "ADDRESS": 2}',
            mock_file.getvalue(),
        )

        # close mock
        StringIO.close(mock_file)

    @mock.patch("tensorflow.keras.Model.save", return_value=None)
    @mock.patch("tensorflow.keras.models.load_model", return_value=None)
    @mock.patch("builtins.open", side_effect=mock_open)
    def test_load(self, *mocks):
        dir = os.path.join(_resource_labeler_dir, "regex_model/")
        loaded_model = RegexModel.load_from_disk(dir)
        self.assertIsInstance(loaded_model, RegexModel)

        self.assertEqual(
            mock_model_parameters["encapsulators"],
            loaded_model._parameters["encapsulators"],
        )
        self.assertEqual(
            mock_model_parameters["regex_patterns"],
            loaded_model._parameters["regex_patterns"],
        )


if __name__ == "__main__":
    unittest.main()
