import json
import os
import unittest
from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf

from dataprofiler.labelers.char_load_tf_model import CharLoadTFModel

_file_dir = os.path.dirname(os.path.abspath(__file__))
_resource_labeler_dir = pkg_resources.resource_filename("resources", "labelers")


mock_model_parameters = {
    "model_path": "project/example/path/fake_model.h5",
    "default_label": "UNKNOWN",
}


mock_label_mapping = {
    "PAD": 0,
    "CITY": 1,  # ensure that overlapping labels get removed.
    "UNKNOWN": 1,
    "ADDRESS": 2,
}


def mock_tf_model(*args, **kwargs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(None,), dtype=tf.int64))
    model.add(
        tf.keras.layers.Embedding(
            input_dim=100,
            output_dim=30,
            embeddings_initializer="normal",
            trainable=True,
        )
    )
    model.add(tf.keras.layers.Dense(units=10, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model


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


@mock.patch("tensorflow.keras.models.load_model", side_effect=mock_tf_model)
class TestCharLoadTFModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # data
        cls.df = pd.DataFrame(
            {
                0: [
                    "MUCH xerophytic GOOFPROOF. Ranch Declarerevise health WITH "
                    "zinc Rhizoctinia.INCULCATION suntrapMordacity `GUAN... "
                    "NECROMANTIC` HAVE mastopathy_nonfeasance_DEMOCRAT 26/09/95 "
                    "18:16 HE sugarcoat [8eec39e5-8acc-40ca-b424-7171ac49131b] "
                    "ourselves"
                ],
                1: [[[164, 178, "DATETIME"], [193, 229, "UUID"]]],
            }
        )
        cls.label_mapping = {
            "PAD": 0,
            "CITY": 1,  # SAME AS UNKNOWN, ensure that overlapping
            "UNKNOWN": 1,  # labels get removed.
            "ADDRESS": 2,
            "BAN": 3,
            "CREDIT_CARD": 4,
            "EMAIL_ADDRESS": 5,
            "UUID": 6,
            "HASH_OR_KEY": 7,
            "IPV4": 8,
            "IPV6": 9,
            "MAC_ADDRESS": 10,
            "NAME": 11,  # SAME AS PERSON
            "PERSON": 11,
            "PHONE_NUMBER": 12,
            "SSN": 13,
            "URL": 14,
            "DATETIME": 15,
            "INTEGER_BIG": 16,  # SAME AS INTEGER
        }
        cls.model_path = "project/example/path/fake_model.h5"

    def test_init(self, *mocks):

        # load default
        model = CharLoadTFModel(self.model_path, self.label_mapping)
        expected_labels = [
            "PAD",
            "UNKNOWN",
            "ADDRESS",
            "BAN",
            "CREDIT_CARD",
            "EMAIL_ADDRESS",
            "UUID",
            "HASH_OR_KEY",
            "IPV4",
            "IPV6",
            "MAC_ADDRESS",
            "PERSON",
            "PHONE_NUMBER",
            "SSN",
            "URL",
            "DATETIME",
            "INTEGER_BIG",
        ]

        self.assertDictEqual(self.label_mapping, model.label_mapping)
        self.assertEqual(self.model_path, model._parameters["model_path"])
        self.assertListEqual(expected_labels, model.labels)

    def test_reverse_label_mapping(self, *mocks):

        # load default
        model = CharLoadTFModel(self.model_path, self.label_mapping)

        # should notice that CITY does not exist in reverse
        expected_reverse_label_mapping = {
            0: "PAD",
            1: "UNKNOWN",
            2: "ADDRESS",
            3: "BAN",
            4: "CREDIT_CARD",
            5: "EMAIL_ADDRESS",
            6: "UUID",
            7: "HASH_OR_KEY",
            8: "IPV4",
            9: "IPV6",
            10: "MAC_ADDRESS",
            11: "PERSON",
            12: "PHONE_NUMBER",
            13: "SSN",
            14: "URL",
            15: "DATETIME",
            16: "INTEGER_BIG",
        }

        self.assertDictEqual(
            expected_reverse_label_mapping, model.reverse_label_mapping
        )

    def test_set_label_mapping(self, *mocks):

        # load default
        model = CharLoadTFModel(self.model_path, self.label_mapping)

        # test not dict
        label_mapping = None
        with self.assertRaisesRegex(
            TypeError,
            "Labels must either be a non-empty encoding dict "
            "which maps labels to index encodings or a list.",
        ):
            model.set_label_mapping(label_mapping)

        # test label_mapping without PAD
        label_mapping = {
            "CITY": 1,  # SAME AS UNKNOWN
            "UNKNOWN": 1,
            "ADDRESS": 2,
        }
        model.set_label_mapping(label_mapping)
        label_mapping["PAD"] = 0
        self.assertDictEqual(label_mapping, model.label_mapping)

        # test list without pad sets PAD: 0
        labels = [
            "UNKNOWN",
            "ADDRESS",
        ]
        label_mapping = {
            "PAD": 1,
            "UNKNOWN": 2,
            "ADDRESS": 3,
        }
        model.set_label_mapping(labels)
        self.assertDictEqual(label_mapping, model.label_mapping)

        # test label_mapping with PAD: 0
        label_mapping = {
            "PAD": 0,
            "CITY": 1,  # SAME AS UNKNOWN
            "UNKNOWN": 1,
            "ADDRESS": 2,
        }
        model.set_label_mapping(label_mapping)
        self.assertDictEqual(label_mapping, model.label_mapping)

        # test if pad not set, but 0 taken set to last ind
        # test label_mapping without PAD
        label_mapping = {
            "CITY": 0,
            "UNKNOWN": 1,
            "ADDRESS": 2,
        }
        model.set_label_mapping(label_mapping)
        label_mapping["PAD"] = 3
        self.assertDictEqual(label_mapping, model.label_mapping)

    def test_predict(self, *mocks):
        # model
        model = CharLoadTFModel(self.model_path, self.label_mapping)
        data_gen = [np.array([[1, 3], [1, 2]])]
        result = model.predict(data_gen)
        self.assertIn("pred", result)
        self.assertEqual((2, 2), np.array(result["pred"]).shape)

        result = model.predict(data_gen, show_confidences=True)
        self.assertIn("pred", result)
        self.assertIn("conf", result)
        self.assertEqual((2, 2, model.num_labels), np.array(result["conf"]).shape)

    def test_fit_and_predict(self, *mocks):
        # model
        model = CharLoadTFModel(self.model_path, self.label_mapping)

        # data for model
        data_gen = [
            [
                np.array([[1, 3], [1, 2]]),  # x_data
                np.zeros((2, 2, model.num_labels)),  # y_data
            ]
        ]
        cv_gen = data_gen

        # Basic Fit with Validation Data
        with self.assertLogs(
            "DataProfiler.labelers.char_load_tf_model", level="INFO"
        ) as logs:
            history, f1, f1_report = model.fit(data_gen, cv_gen, reset_weights=True)

        # Ensure info was logged during fit
        self.assertTrue(len(logs.output))

        data_gen = [np.array([[1, 3], [1, 2]])]
        model.predict(data_gen)

        # fit with new labels
        new_label_mapping = {
            "PAD": 0,
            "TEST": 1,
            "NEW": 2,
            "MAPPING": 3,
            model._parameters["default_label"]: 4,
        }
        data_gen = [
            [
                np.array([[1, 3], [1, 2]]),  # x_data
                np.zeros((2, 2, len(new_label_mapping))),  # y_data
            ]
        ]
        history, f1, f1_report = model.fit(
            data_gen, cv_gen, label_mapping=new_label_mapping
        )

        # predict after fitting on just the text
        model.predict([data_gen[0][0]])

    @mock.patch("os.makedirs", return_value=None)
    def test_validation_evaluate_and_classification_report(self, *mocks):
        model = CharLoadTFModel(self.model_path, self.label_mapping)
        model._construct_model()  # must make model to do priv validate func

        # validation data
        val_gen = [
            [
                np.ones((2, 20)),  # x_data
                np.zeros((2, 20, model.num_labels)),  # y_data
            ]
        ]
        val_gen[0][1][0, :11, self.label_mapping["ADDRESS"]] = 1

        f1, f1_report = model._validate_training(val_gen, 32, True, True)
        self.assertIsNotNone(f1)
        self.assertIsNotNone(f1_report)
        self.assertEqual(11, f1_report["ADDRESS"]["support"])

    def test_param_validation(self, *mocks):
        # Make sure all parameters can be altered. Make sure non-valid params
        # are caught
        parameters = {
            "default_label": "UNKNOWN",
        }
        invalid_parameters = {
            "fake_extra_param": "fails",
        }
        model = CharLoadTFModel(
            self.model_path, label_mapping=self.label_mapping, parameters=parameters
        )
        model._construct_model()
        self.assertDictEqual(parameters, model._parameters)
        with self.assertRaises(ValueError):
            CharLoadTFModel(
                self.model_path,
                label_mapping=self.label_mapping,
                parameters=invalid_parameters,
            )

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_help(self, mock_stdout, *mocks):
        CharLoadTFModel.help()
        self.assertIn("CharLoadTFModel", mock_stdout.getvalue())
        self.assertIn("Parameters", mock_stdout.getvalue())

    @mock.patch("tensorflow.keras.Model.save", return_value=None)
    @mock.patch("builtins.open")
    def test_save(self, mock_open, mock_tf_save, *mocks):
        # setup mock
        mock_file = setup_save_mock_open(mock_open)

        # Save and load a CNN Model with custom parameters
        parameters = {}
        label_mapping = mock_label_mapping
        model = CharLoadTFModel(self.model_path, label_mapping, parameters)

        # save file and test
        save_path = "./fake/path"
        model.save_to_disk(save_path)
        self.assertEqual(
            # model parameters
            '{"default_label": "UNKNOWN", "pad_label": "PAD"}'
            # label_mapping
            '{"PAD": 0, "CITY": 1, "UNKNOWN": 1, "ADDRESS": 2}',
            mock_file.getvalue(),
        )
        mock_tf_save.assert_called_with(save_path)

        # close mock
        StringIO.close(mock_file)

    @mock.patch("tensorflow.keras.Model.save", return_value=None)
    @mock.patch("builtins.open", side_effect=mock_open)
    def test_load(self, *mocks):
        dir = "fake/path/"
        loaded_model = CharLoadTFModel.load_from_disk(dir)
        self.assertIsInstance(loaded_model, CharLoadTFModel)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_model_details(self, mock_stdout, *mocks):
        # Default Model Construct
        model = CharLoadTFModel(self.model_path, self.label_mapping)

        # Test Details
        model.details()
        self.assertIn("input", mock_stdout.getvalue())
        self.assertIn("dense", mock_stdout.getvalue())
        self.assertIn("softmax_output", mock_stdout.getvalue())
        self.assertIn("Total params", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
