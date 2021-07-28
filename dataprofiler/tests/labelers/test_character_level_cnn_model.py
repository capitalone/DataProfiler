import os
import unittest
from unittest import mock
import json
from io import StringIO
import pkg_resources

import pandas as pd
import numpy as np
import tensorflow as tf

from dataprofiler.labelers.character_level_cnn_model \
    import CharacterLevelCnnModel


_file_dir = os.path.dirname(os.path.abspath(__file__))
_resource_labeler_dir = pkg_resources.resource_filename('resources', 'labelers')


mock_model_parameters = {
    "max_char_encoding_id": 127,
    "size_conv": 13,
    "max_length": 10,
    "dim_embed": 64,
    "size_fc": [96, 96],
    "dropout": 0.073,
    "default_label": "UNKNOWN",
    "num_fil": [48, 48, 48, 48]
}


mock_label_mapping = {
    "PAD": 0,
    "CITY": 1,  # ensure that overlapping labels get removed.
    "UNKNOWN": 1,
    "ADDRESS": 2,
}


def mock_open(filename, *args):
    if filename.find('model_parameters') >= 0:
        return StringIO(json.dumps(mock_model_parameters))
    elif filename.find('label_mapping') >= 0:
        return StringIO(json.dumps(mock_label_mapping))


def setup_save_mock_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


class TestCharacterLevelCNNModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # data
        cls.df = pd.DataFrame(
            {0: ["MUCH xerophytic GOOFPROOF. Ranch Declarerevise health WITH "
                 "zinc Rhizoctinia.INCULCATION suntrapMordacity `GUAN... "
                 "NECROMANTIC` HAVE mastopathy_nonfeasance_DEMOCRAT 26/09/95 "
                 "18:16 HE sugarcoat [8eec39e5-8acc-40ca-b424-7171ac49131b] "
                 "ourselves"],
             1: [[[164, 178, "DATETIME"], [193, 229, "UUID"]]]
            })
        cls.label_mapping = {
            'PAD': 0,
            'CITY': 1,        # SAME AS UNKNOWN, ensure that overlapping
            'UNKNOWN': 1,  # labels get removed.
            'ADDRESS': 2,
            'BAN': 3,
            'CREDIT_CARD': 4,
            'EMAIL_ADDRESS': 5,
            'UUID': 6,
            'HASH_OR_KEY': 7,
            'IPV4': 8,
            'IPV6': 9,
            'MAC_ADDRESS': 10,
            'NAME': 11,  # SAME AS PERSON
            'PERSON': 11,
            'PHONE_NUMBER': 12,
            'SSN': 13,
            'URL': 14,
            'DATETIME': 15,
            'INTEGER_BIG': 16,  # SAME AS INTEGER
        }

    def test_label_mapping(self, *mocks):

        # load default
        cnn_model = CharacterLevelCnnModel(self.label_mapping)

        self.assertDictEqual(self.label_mapping, cnn_model.label_mapping)

    def test_labels(self, *mocks):

        # load default
        cnn_model = CharacterLevelCnnModel(self.label_mapping)

        labels = ['PAD', 'UNKNOWN', 'ADDRESS', 'BAN', 'CREDIT_CARD',
                  'EMAIL_ADDRESS', 'UUID', 'HASH_OR_KEY', 'IPV4', 'IPV6',
                  'MAC_ADDRESS', 'PERSON', 'PHONE_NUMBER', 'SSN', 'URL',
                  'DATETIME', 'INTEGER_BIG']

        self.assertListEqual(labels, cnn_model.labels)

    def test_reverse_label_mapping(self, *mocks):

        # load default
        cnn_model = CharacterLevelCnnModel(self.label_mapping)

        # should notice that CITY does not exist in reverse
        reverse_label_mapping = {
            0: 'PAD',
            1: 'UNKNOWN',
            2: 'ADDRESS',
            3: 'BAN',
            4: 'CREDIT_CARD',
            5: 'EMAIL_ADDRESS',
            6: 'UUID',
            7: 'HASH_OR_KEY',
            8: 'IPV4',
            9: 'IPV6',
            10: 'MAC_ADDRESS',
            11: 'PERSON',
            12: 'PHONE_NUMBER',
            13: 'SSN',
            14: 'URL',
            15: 'DATETIME',
            16: 'INTEGER_BIG'}

        self.assertDictEqual(reverse_label_mapping,
                             cnn_model.reverse_label_mapping)

    def test_set_label_mapping(self, *mocks):

        # load default
        cnn_model = CharacterLevelCnnModel(self.label_mapping)

        # test not dict
        label_mapping = None
        with self.assertRaisesRegex(
                TypeError, "Labels must either be a non-empty encoding dict "
                           "which maps labels to index encodings or a list."):
            cnn_model.set_label_mapping(label_mapping)

        # test raise error for setting non PAD to 0
        label_mapping = {'TEST': 0}
        with self.assertRaisesRegex(ValueError,
                                    "`PAD` must map to index zero."):
            cnn_model.set_label_mapping(label_mapping)

        # test raise error for setting PAD other than 0
        label_mapping = {'PAD': 1}
        with self.assertRaisesRegex(ValueError,
                                    "`PAD` must map to index zero."):
            cnn_model.set_label_mapping(label_mapping)

        # test raise error if default label not in mapping
        label_mapping = {'PAD': 0}
        with self.assertRaisesRegex(ValueError,
                                    "The `default_label` of UNKNOWN must "
                                    "exist in the label mapping."):
            cnn_model.set_label_mapping(label_mapping)

        # test label_mapping without PAD
        label_mapping = {
            'CITY': 1,  # SAME AS UNKNOWN
            'UNKNOWN': 1,
            'ADDRESS': 2,
        }
        cnn_model.set_label_mapping(label_mapping)

        self.assertNotEqual(label_mapping, cnn_model.label_mapping)
        label_mapping['PAD'] = 0
        self.assertDictEqual(label_mapping, cnn_model.label_mapping)

        # test label_mapping with PAD: 0
        label_mapping = {
            'PAD': 0,
            'CITY': 1,  # SAME AS UNKNOWN
            'UNKNOWN': 1,
            'ADDRESS': 2,
        }
        cnn_model.set_label_mapping(label_mapping)
        self.assertDictEqual(label_mapping, cnn_model.label_mapping)

    @mock.patch("pandas.DataFrame.to_csv", return_value=None)
    def test_fit_and_predict_with_reset_weights(self, *mocks):
        # model
        cnn_model = CharacterLevelCnnModel(self.label_mapping)

        # data for model
        data_gen = [
            [np.array([['test']]),  # x_data
             np.zeros((1, 3400, max(self.label_mapping.values())+1))]  # y_data
        ]
        cv_gen = data_gen

        # Basic Fit with Validation Data
        with self.assertLogs('DataProfiler.labelers.character_level_cnn_model',
                             level='INFO') as logs:
            history, f1, f1_report = cnn_model.fit(data_gen, cv_gen,
                                                   reset_weights=True)

        # Ensure info was logged during fit
        self.assertTrue(len(logs.output))

        data_gen = [
            np.array([['test']])
        ]
        cnn_model.predict(data_gen)

    @mock.patch("pandas.DataFrame.to_csv", return_value=None)
    @mock.patch("os.makedirs", return_value=None)
    def test_validation_evaluate_and_classification_report(self, *mocks):
        cnn_model = CharacterLevelCnnModel(self.label_mapping)
        cnn_model._construct_model()

        # validation data
        val_gen = [
            [np.array([['123 fake st']]),
             np.zeros((1, 3400, max(self.label_mapping.values()) + 1))]
        ]
        val_gen[0][1][:, :11, self.label_mapping['ADDRESS']] = 1

        f1, f1_report = cnn_model._validate_training(val_gen, 32, True, True)
        self.assertIsNotNone(f1)
        self.assertIsNotNone(f1_report)
        self.assertEqual(11, f1_report['ADDRESS']['support'])

    def test_fit_and_predict_with_new_labels(self):
        # Initialize model
        cnn_model = CharacterLevelCnnModel(self.label_mapping)

        # data for model
        data_gen = [
            [np.array([['test']]),  # x_data
             np.zeros((1, 3400, max(self.label_mapping.values()) + 1))]
            # y_data
        ]
        cv_gen = data_gen

        cnn_model._construct_model()
        
        # fit with new labels
        history, f1, f1_report = cnn_model.fit(
            data_gen, cv_gen, label_mapping=self.label_mapping)

        # predict after fitting on just the text
        cnn_model.predict(data_gen[0][0])

    def test_fit_and_predict_with_new_labels_set_via_method(self):
        # Initialize model
        invalid_entities = {"PAD": 0, "UNKNOWN": 1, "test3": 2}
        cnn_model = CharacterLevelCnnModel(invalid_entities)
        cnn_model._construct_model()
        invalid_entities2 = {"PAD": 0, "UNKNOWN": 1}
        cnn_model.set_label_mapping(invalid_entities2)
        cnn_model._reconstruct_model()
        cnn_model.set_label_mapping(self.label_mapping)

        # data for model
        data_gen = [
            [np.array([['test']]),  # x_data
             np.zeros((1, 3400, max(self.label_mapping.values()) + 1))]
            # y_data
        ]
        cv_gen = data_gen

        cnn_model._construct_model()

        # set different labels
        cnn_model.set_label_mapping(self.label_mapping)
        history, f1, f1_report = cnn_model.fit(data_gen, cv_gen)

        # test predict on just the text
        cnn_model.predict(data_gen[0][0])

    def test_validation(self):
        
        # model
        cnn_model = CharacterLevelCnnModel(label_mapping=self.label_mapping)
        cnn_model._construct_model()

        # data for model
        cv_data_gen = [
            [np.array([['test']]),  # x_data
             np.zeros((1, 3400, max(self.label_mapping.values()) + 1))]
            # y_data
        ]
        
        # validation
        cnn_model._validate_training(
            cv_data_gen, batch_size_test=32, verbose_log=True,
            verbose_keras=False)

    def test_param_validation(self):
        # Make sure all parameters can be altered. Make sure non-valid params
        # are caught
        parameters = {
            'max_length': 10, 'max_char_encoding_id': 11, 'size_fc': [64, 64],
            'dropout': 0.9, 'size_conv': 11, 'default_label': "UNKNOWN",
            'num_fil': [48 for _ in range(2)]
        }
        invalid_parameters = {
            'max_length': -1, 'max_char_encoding_id': "words", 'size_fc': 5,
            'dropout': 0.9, 'size_conv': 11, 'optimizer': 6,
            'num_fil': [48 for _ in range(2)], 'fake_extra_param': "fails"}
        cnn_model = CharacterLevelCnnModel(label_mapping=self.label_mapping,
                                           parameters=parameters)
        cnn_model._construct_model()
        self.assertDictEqual(parameters, cnn_model._parameters)
        with self.assertRaises(ValueError):
            CharacterLevelCnnModel(label_mapping=self.label_mapping,
                                   parameters=invalid_parameters)

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        CharacterLevelCnnModel.help()
        self.assertIn("CharacterLevelCnnModel", mock_stdout.getvalue())
        self.assertIn("Parameters", mock_stdout.getvalue())
        
    def test_input_encoding(self):
        cnn_model = CharacterLevelCnnModel(self.label_mapping)
        
        input_str_tensor = tf.convert_to_tensor(['test'])
        max_char_encoding_id = 127
        max_len = 10
        
        encode_output = cnn_model._char_encoding_layer(
            input_str_tensor, max_char_encoding_id, max_len).numpy()[0]
        expected_output = [117, 102, 116, 117, 0, 0, 0, 0, 0, 0]
        self.assertCountEqual(encode_output, expected_output)
        
    def test_threshold_layer(self):
        cnn_model = CharacterLevelCnnModel(self.label_mapping)
        
        confidences = tf.convert_to_tensor([[[0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.6, 0.4, 0.0],
                                             [0.9, 0.0, 0.0, 0.1]]])
        
        argmax = tf.convert_to_tensor([[2, 1, 0]])
        expected_threshold_output = argmax.numpy()[0]
        num_labels = 4
        
        threshold_layer = cnn_model._argmax_threshold_layer(
            num_labels, threshold=0.0, default_ind=1)
        threshold_output = threshold_layer(argmax, confidences).numpy()[0]
        self.assertCountEqual(threshold_output, expected_threshold_output)

    @mock.patch("tensorflow.keras.Model.save", return_value=None)
    @mock.patch("tensorflow.keras.models.load_model", return_value=None)
    @mock.patch("dataprofiler.labelers.character_level_cnn_model.callable",
                return_value=True)
    @mock.patch("builtins.open")
    def test_save(self, mock_open, *mocks):
        # setup mock
        mock_file = setup_save_mock_open(mock_open)

        # Save and load a CNN Model with custom parameters
        parameters = {'max_char_encoding_id': 100, 'size_conv': 6}
        label_mapping = {
            'PAD': 0,
            'CITY': 1,  # SAME AS UNKNOWN
            'UNKNOWN': 1,
            'ADDRESS': 2,
        }
        cnn_model = CharacterLevelCnnModel(label_mapping, parameters)
        cnn_model._model = mock.Mock()
        cnn_model._model_num_labels = 3
        cnn_model._model_default_ind = 1

        # save file and test
        cnn_model.save_to_disk(".")
        self.assertEqual(
            # model parameters
            '{"max_char_encoding_id": 100, "size_conv": 6, "max_length": 3400, '
            '"dim_embed": 64, "size_fc": [96, 96], "dropout": 0.073, '
            '"default_label": "UNKNOWN", "num_fil": [48, 48, 48, 48], '
            '"pad_label": "PAD"}'
            # label_mapping 
            '{"PAD": 0, "CITY": 1, "UNKNOWN": 1, "ADDRESS": 2}',
            mock_file.getvalue())

        # close mock
        StringIO.close(mock_file)

    @mock.patch("tensorflow.keras.Model.save", return_value=None)
    @mock.patch("tensorflow.keras.models.load_model", return_value=None)
    @mock.patch("builtins.open", side_effect=mock_open)
    @mock.patch("dataprofiler.labelers.character_level_cnn_model.callable",
                return_value=True)
    def test_load(self, *mocks):
        dir = os.path.join(
            _resource_labeler_dir,
            'unstructured_model/')
        loaded_model = CharacterLevelCnnModel.load_from_disk(dir)
        self.assertIsInstance(loaded_model, CharacterLevelCnnModel)

    def test_model_construct(self):
        # Default Model Construct

        cnn_model = CharacterLevelCnnModel(label_mapping=self.label_mapping)
        cnn_model._construct_model()
        # Test Details
        cnn_model.details()

        expected_layers = [
            'input_1',
            'lambda',
            'embedding',
            'conv1d',
            'dropout',
            'batch_normalization',
            'conv1d_1',
            'dropout_1',
            'batch_normalization_1',
            'conv1d_2',
            'dropout_2',
            'batch_normalization_2',
            'conv1d_3',
            'dropout_3',
            'batch_normalization_3',
            'dense',
            'dropout_4',
            'dense_1',
            'dropout_5',
            'dense_2',
            'tf_op_layer_ArgMax',
            'thresh_arg_max_layer'
        ]
        model_layers = [layer.name for layer in cnn_model._model.layers]
        self.assertEqual(len(expected_layers), len(model_layers))
        self.assertEqual(17, cnn_model.num_labels)
        
if __name__ == '__main__':
    unittest.main()
