import os
import unittest
from unittest import mock
import json
from io import StringIO

import pandas as pd
import numpy as np

import data_profiler as dp
from data_profiler.data_readers.csv_data import CSVData
from data_profiler.data_readers.csv_data import JSONData
from data_profiler.data_readers.csv_data import ParquetData
from data_profiler.data_readers.csv_data import AVROData

from data_profiler.labelers.data_labelers import BaseDataLabeler, \
    TrainableDataLabeler
from data_profiler.labelers import data_processing
from data_profiler.labelers import StructCharPreprocessor
from data_profiler.labelers.base_model import BaseModel, BaseTrainableModel


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def setup_save_mock_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


class TestDataLabelerTrainer(unittest.TestCase):

    def test_train_method_exists(self):

        self.assertTrue(hasattr(dp, 'train_structured_labeler'))

        with mock.patch('data_profiler.train_structured_labeler') as mock_obj:
            dp.train_structured_labeler(None, None)
            self.assertEqual(mock_obj.call_args, mock.call(None, None))

    def test_accepted_inputs(self):
        with self.assertRaisesRegex(TypeError,
                                    "Input data must be either a "
                                    "`pd.DataFrame` or a `data_profiler.Data` "
                                    "and not of type `TextData`."):
            dp.train_structured_labeler(None, None)

        with self.assertRaisesRegex(TypeError,
                                    "The output dirpath must be a string."):
            dp.train_structured_labeler(pd.DataFrame([]), save_dirpath=0)

        # doesn't accept text data
        text_data = dp.Data(data='test', data_type='text')
        with self.assertRaisesRegex(TypeError,
                                    "Input data must be either a "
                                    "`pd.DataFrame` or a `data_profiler.Data` "
                                    "and not of type `TextData`."):
            dp.train_structured_labeler(text_data, None)

        with self.assertRaisesRegex(ValueError,
                                    "The `save_dirpath` is not valid or not "
                                    "accessible."):
            dp.train_structured_labeler(pd.DataFrame([]), "/a/test")

        try:
            data = {'BACKGROUND': ["Beep", "Boop"],
                    'PERSON': ["GRANT", "MENSHENG"]}
            df = pd.DataFrame(data=data)
            dp.train_structured_labeler(df, save_dirpath=None)

            fake_data = dp.Data(data=df, data_type='csv')
            dp.train_structured_labeler(fake_data, save_dirpath=None)

            fake_data = dp.Data(data=df, data_type='json')
            dp.train_structured_labeler(fake_data, save_dirpath=None)

            fake_data = dp.Data(data=df, data_type='parquet')
            dp.train_structured_labeler(fake_data, save_dirpath=None)

        except Exception as e:
            self.fail(str(e))


class TestDataLabeler(unittest.TestCase):

    @mock.patch("data_profiler.labelers.data_labelers."
                "BaseDataLabeler._load_data_labeler")
    def test_load_data_labeler(self, *mocks):

        # error if no labeler specified
        with self.assertRaisesRegex(TypeError,
                                    r'__new__\(\) missing 1 required positional'
                                    r' argument: \'labeler_type\''):
            data_labeler = dp.DataLabeler()

        # error if no labeler specified
        with self.assertRaisesRegex(ValueError,
                                    r'No DataLabeler class types matched the '
                                    r'input, `fake_labeler`. Allowed types '
                                    r'\[\'structured\', \'unstructured\'\].'):
            data_labeler = dp.DataLabeler(labeler_type='fake_labeler')

        # test loads a structured data labeler
        data_labeler = dp.DataLabeler(labeler_type='structured')
        self.assertIsInstance(data_labeler, BaseDataLabeler)

        # test loads an unstructured data labeler
        data_labeler = dp.DataLabeler(labeler_type='unstructured')
        self.assertIsInstance(data_labeler, BaseDataLabeler)


label_encoding = {
  "encoder": {
    'PAD': 0,
    'CITY': 1,  # SAME AS BACKGROUND
    'BACKGROUND': 1,
    'ADDRESS': 2,
    'PERSON': 3,
  },
  "decoder": {
    0: "PAD",
    1: "BACKGROUND",
    2: "ADDRESS",
    3: "PERSON"
  },
  "count": 4
}


unstruct_data_labeler_parameters = {
    'model': {
        'class': 'CharacterLevelCnnModel',
        'parameters': {}
    },
    'label_mapping': {
        'PAD': 0,
        'CITY': 1,  # SAME AS BACKGROUND
        'BACKGROUND': 1,
        'ADDRESS': 2,
        'PERSON': 3,
    },
    'preprocessor': {
        'class': 'CharPreprocessor'
    },
    'postprocessor': {
        'class': 'CharPostprocessor'
    },
}


struct_data_labeler_parameters = {
    'model': {
        'class': 'CharacterLevelCnnModel',
        'parameters': {}
    },
    'label_mapping': {
        'PAD': 0,
        'CITY': 1,  # SAME AS BACKGROUND
        'BACKGROUND': 1,
        'ADDRESS': 2,
        'PERSON': 3,
    },
    'preprocessor': {
        'class': 'StructCharPreprocessor'
    },
    'postprocessor': {
        'class': 'StructCharPostprocessor'
    },
}


def mock_open(filename, *args):
    if filename.find('model_parameters.json') >= 0:
        return StringIO('{}')
    elif filename.find('label_mapping.json') >= 0:
        return StringIO(json.dumps(label_encoding['encoder']))
    elif filename.find('data_labeler_parameters') >= 0:
        if filename.find('unstructured_model') >= 0:
            return StringIO(json.dumps(unstruct_data_labeler_parameters))
        elif filename.find('structured_model') >= 0:
            return StringIO(json.dumps(struct_data_labeler_parameters))
        else:
            return StringIO('{}')
    elif filename.find('preprocessor_parameters') >= 0:
        return StringIO('{}')
    elif filename.find('postprocessor_parameters') >= 0:
        return StringIO('{}')


@mock.patch("keras.models.load_model")
@mock.patch("builtins.open", side_effect=mock_open)
class TestLoadedDataLabeler(unittest.TestCase):

    def test_has_public_functions(self, *args):
        public_functions = [
            "predict",
            "save_to_disk",
            "load_with_components",
            "load_from_disk",
        ]

        for func in public_functions:
            self.assertTrue(hasattr(BaseDataLabeler, func))

    @staticmethod
    def _setup_mock_load_model(mock_load_model):
        mock_load_model.return_value = mock.Mock()

    def test_load_labeler(self, mock_open, mock_load_model):

        # TODO: Mock exist functions

        self._setup_mock_load_model(mock_load_model)

        # load default
        data_labeler = dp.DataLabeler(labeler_type='structured')

        self.assertDictEqual(label_encoding['encoder'],
                             struct_data_labeler_parameters['label_mapping'])
        self.assertIsInstance(data_labeler.preprocessor,
                              data_processing.BaseDataPreprocessor)
        self.assertIsInstance(data_labeler.postprocessor,
                              data_processing.BaseDataPostprocessor)

    def test_invalid_data_formats(self, mock_open, mock_load_model):

        def _invalid_check(data):
            with self.assertRaisesRegex(TypeError,
                                        "Data must either be imported using "
                                        "the data_readers or pd.DataFrame."):
                BaseDataLabeler._check_and_return_valid_data_format(data)

        invalid_data = ["string", 1, None, dict()]
        print("\nInvalid Data Checks:")
        for data in invalid_data:
            print("\tChecking data format: {}".format(str(type(data))))
            _invalid_check(data)

            # cannot predict dict
            _invalid_check(data=None)

            # cannot predict dict
            _invalid_check(data={})

    def test_valid_fit_data_formats(self, mock_open, mock_load_model):

        def _valid_check(data):
            try:
                print("\tChecking data format: {}".format(str(type(data))))
                data = BaseDataLabeler._check_and_return_valid_data_format(
                    data, fit_or_predict='fit'
                )
            except Exception as e:
                self.fail("Exception raised on input of accepted types.")
            return data

        valid_data = [
            CSVData(data=pd.DataFrame([])),
            JSONData(data=pd.DataFrame([])),
            ParquetData(data=pd.DataFrame([])),
            AVROData(data=pd.DataFrame([])),
            pd.DataFrame([]),
            list(),
            np.array([])
        ]
        print("\nValid Data Fit Checks:")
        for data in valid_data:
            data = _valid_check(data)
            self.assertIsInstance(data, pd.DataFrame)

    def test_valid_predict_data_formats(self, mock_open, mock_load_model):

        def _valid_check(data):
            try:
                print("\tChecking data format: {}".format(str(type(data))))
                data = BaseDataLabeler._check_and_return_valid_data_format(
                    data, fit_or_predict='predict'
                )
            except Exception as e:
                self.fail("Exception raised on input of accepted types.")
            return data

        valid_data = [
            CSVData(data=pd.DataFrame([])),
            JSONData(data=pd.DataFrame([])),
            ParquetData(data=pd.DataFrame([])),
            AVROData(data=pd.DataFrame([])),
            pd.DataFrame([]),
            list(),
            np.array([]),
            pd.Series([], dtype=object)
        ]
        print("\nValid Data Predict Checks:")
        for data in valid_data:
            data = _valid_check(data)
            self.assertTrue(
                isinstance(data, np.ndarray) or
                isinstance(data, pd.Series) or
                isinstance(data, pd.DataFrame)
            )

    def test_set_labels(self, mock_open, mock_load_model):

        # setup
        self._setup_mock_load_model(mock_load_model)
        data_labeler = dp.DataLabeler(labeler_type='structured')
        data_labeler._model = mock.Mock()
        data_labeler._model.set_label_mapping.return_value = None

        # test require pad true
        data_labeler._model.requires_zero_mapping = True
        data_labeler.set_labels(['a', 'b'])
        data_labeler._model.set_label_mapping.assert_called_with(
            label_mapping={'a': 1, 'b': 2})

        # test require pad false
        data_labeler._model.requires_zero_mapping = False
        data_labeler.set_labels(['a', 'b'])
        data_labeler._model.set_label_mapping.assert_called_with(
            label_mapping={'a': 0, 'b': 1})

    def test_equality(self, mock_open, mock_load_model):
        self._setup_mock_load_model(mock_load_model)

        struct_data_labeler1 = dp.DataLabeler(labeler_type='structured')
        struct_data_labeler2 = dp.DataLabeler(labeler_type='structured')
        unstruct_data_labeler1 = dp.DataLabeler(labeler_type='unstructured')
        unstruct_data_labeler2 = dp.DataLabeler(labeler_type='unstructured')

        # Assert they are equal
        self.assertEqual(struct_data_labeler1, struct_data_labeler2)
        self.assertEqual(unstruct_data_labeler1, unstruct_data_labeler2)

        # Assert they are unequal because of type
        self.assertNotEqual(struct_data_labeler1, unstruct_data_labeler1)

        # Assert they are unequal because of _label_encoding
        struct_data_labeler1.set_labels(['BACKGROUND', 'b', 'c'])
        self.assertNotEqual(struct_data_labeler1, struct_data_labeler2)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_help(self, mock_stdout, *mock):
        data_labeler = dp.DataLabeler(labeler_type='structured')
        data_labeler.help()

        self.assertIn("Process Input Format", mock_stdout.getvalue())
        self.assertIn("Process Output Format:", mock_stdout.getvalue())
        self.assertIn("Parameters", mock_stdout.getvalue())

    @mock.patch("data_profiler.labelers.data_labelers."
                "BaseDataLabeler._load_data_labeler")
    def test_save_labeler(self, mock_load_data_labeler, mock_open,
                          mock_load_model):

        # setup mocks
        mock_file = setup_save_mock_open(mock_open)

        base_data_labeler = BaseDataLabeler('fake_path')
        base_data_labeler._model = mock.Mock()
        base_data_labeler._preprocessor = mock.Mock()
        base_data_labeler._postprocessor = mock.Mock()

        base_data_labeler.save_to_disk('test')

        self.assertEqual(
            '{"model": {"class": "Mock"}, "preprocessor": {"class": "Mock"}, '
            '"postprocessor": {"class": "Mock"}}',
            mock_file.getvalue())
        mock_open.assert_called_with('test/data_labeler_parameters.json', 'w')
        base_data_labeler._model.save_to_disk.assert_called_with('test')
        base_data_labeler._preprocessor.save_to_disk.assert_called_with('test')
        base_data_labeler._postprocessor.save_to_disk.assert_called_with('test')

        # close mock
        StringIO.close(mock_file)

    def test_load_with_components(self, *mocks):

        mock_preprocessor = mock.Mock(spec=data_processing.BaseDataPreprocessor)
        mock_preprocessor._parameters = {"test": 1}
        mock_postprocessor = mock.Mock(
            spec=data_processing.BaseDataPostprocessor)
        mock_postprocessor._parameters = {"test": 2}
        mock_model = mock.Mock(spec=BaseTrainableModel)
        mock_model._parameters = {"test": 3}

        data_labeler = BaseDataLabeler.load_with_components(
            preprocessor=mock_preprocessor,
            model=mock_model,
            postprocessor=mock_postprocessor)

        self.assertIsInstance(data_labeler, BaseDataLabeler)
        self.assertEqual('CustomDataLabeler', data_labeler.__class__.__name__)
        self.assertEqual(mock_preprocessor, data_labeler.preprocessor)
        self.assertEqual({"test": 1}, data_labeler.preprocessor._parameters)
        self.assertEqual(mock_model, data_labeler.model)
        self.assertEqual({"test": 3}, data_labeler.model._parameters)
        self.assertEqual(mock_postprocessor, data_labeler.postprocessor)
        self.assertEqual({"test": 2}, data_labeler.postprocessor._parameters)


class TestDataLabelerNoMock(unittest.TestCase):
    def test_multi_labelers(self, *mocks):
        """
        Test Multiple labelers called consecutively.
        :return:
        """
        data = dp.Data(data=pd.DataFrame([12, 2, 3, 4, 5]).astype(str),
                       data_type='parquet')
        data2 = dp.Data(data=pd.DataFrame(['atest', 'b', 'c']), data_type='csv')

        structured_labeler_1 = dp.DataLabeler(labeler_type='structured')
        structured_labeler_1.predict(data)
        unstructured_labeler = dp.DataLabeler(labeler_type='unstructured')
        unstructured_labeler._label_encoding = {
            'PAD': 0,
            'CITY': 1,  # SAME AS BACKGROUND
            'BACKGROUND': 1,
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
            'INTEGER': 16,
            'FLOAT': 17,
            'QUANTITY': 18,
            'ORDINAL': 19
        }

        unstructured_labeler.predict(data)
        structured_labeler_2 = dp.DataLabeler(labeler_type='structured')
        structured_labeler_2.predict(data2)


class TestTrainDataLabeler(unittest.TestCase):

    def test_has_public_functions(self, *args):
        public_functions = [
            "predict",
            "save_to_disk",
            "load_from_disk",
            "load_with_components",
            "fit",
        ]

        for func in public_functions:
            self.assertTrue(hasattr(TrainableDataLabeler, func))

    def test_non_trainable_model_error(self):

        mock_model = mock.Mock(spec=BaseModel)
        data_labeler = TrainableDataLabeler()

        with self.assertRaisesRegex(ValueError, '`model` must have a fit '
                                                'function to be trainable.'):
            data_labeler.set_model(mock_model)

    def test_load_with_components(self):

        mock_preprocessor = mock.Mock(spec=data_processing.BaseDataPreprocessor)
        mock_preprocessor._parameters = {"test": 1}
        mock_postprocessor = mock.Mock(
            spec=data_processing.BaseDataPostprocessor)
        mock_postprocessor._parameters = {"test": 2}

        # assert raises error with non-trainable model
        mock_model = mock.Mock(spec=BaseModel)

        with self.assertRaisesRegex(ValueError, '`model` must have a fit '
                                                'function to be trainable.'):
            data_labeler = TrainableDataLabeler.load_with_components(
                preprocessor=mock_preprocessor,
                model=mock_model,
                postprocessor=mock_postprocessor)

        # assert functional with trainable model
        mock_model = mock.Mock(spec=BaseTrainableModel)
        mock_model._parameters = {"test": 3}

        data_labeler = TrainableDataLabeler.load_with_components(
            preprocessor=mock_preprocessor,
            model=mock_model,
            postprocessor=mock_postprocessor)

        self.assertTrue(hasattr(data_labeler, 'fit'))
        self.assertEqual('CustomTrainableDataLabeler',
                         data_labeler.__class__.__name__)
        self.assertEqual(mock_preprocessor, data_labeler.preprocessor)
        self.assertEqual({"test": 1}, data_labeler.preprocessor._parameters)
        self.assertEqual(mock_model, data_labeler.model)
        self.assertEqual({"test": 3}, data_labeler.model._parameters)
        self.assertEqual(mock_postprocessor, data_labeler.postprocessor)
        self.assertEqual({"test": 2}, data_labeler.postprocessor._parameters)
