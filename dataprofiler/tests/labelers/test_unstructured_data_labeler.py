import json
import os
import re
import unittest
import warnings
from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.data_readers.csv_data import AVROData, CSVData, JSONData, ParquetData
from dataprofiler.labelers import DataLabeler, UnstructuredDataLabeler, data_processing
from dataprofiler.labelers.character_level_cnn_model import CharacterLevelCnnModel

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data_labeler_parameters = {
    "model": {"class": "CharacterLevelCnnModel", "parameters": {}},
    "label_mapping": {
        "PAD": 0,
        "CITY": 1,  # SAME AS UNKNOWN
        "UNKNOWN": 1,
        "ADDRESS": 2,
        "PERSON": 3,
    },
    "preprocessor": {"class": "CharPreprocessor"},
    "postprocessor": {"class": "CharPostprocessor"},
}

preprocessor_parameters = {
    "flatten_split": 0,
    "flatten_separator": " ",
    "is_separate_at_max_len": True,
}

postprocessor_parameters = {
    "use_word_level_argmax": True,
    "output_format": "character_argmax",
    "separators": (" ", ",", ";", "'", '"', ":", "\n", "\t", "."),
    "word_level_min_percent": 0.75,
}


def mock_open(filename, *args):
    if filename.find("data_labeler_parameters") >= 0:
        return StringIO(json.dumps(data_labeler_parameters))
    elif filename.find("preprocessor_parameters") >= 0:
        return StringIO(json.dumps(preprocessor_parameters))
    elif filename.find("postprocessor_parameters") >= 0:
        return StringIO(json.dumps(postprocessor_parameters))


def setup_save_mock_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


@mock.patch("dataprofiler.labelers.data_processing.BaseDataProcessor")
@mock.patch(
    "dataprofiler.labelers.character_level_cnn_model."
    "CharacterLevelCnnModel.load_from_disk"
)
@mock.patch("builtins.open", side_effect=mock_open)
class TestDataLabeler(unittest.TestCase):
    @staticmethod
    def _setup_mock_load_model(mock_load_model):
        model_mock = mock.Mock(spec=CharacterLevelCnnModel)
        model_mock.__class__.__name__ = "CharacterLevelCnnModel"
        model_mock.set_num_labels = mock.Mock()
        mock_load_model.return_value = model_mock
        model_mock.requires_zero_mapping = True
        model_mock.labels = ["PAD", "UNKNOWN", "ADDRESS", "PERSON"]
        model_mock.label_mapping = {
            "PAD": 0,
            "CITY": 1,  # SAME AS UNKNOWN
            "UNKNOWN": 1,
            "ADDRESS": 2,
            "PERSON": 3,
        }
        model_mock.reverse_label_mapping = {
            0: "PAD",
            1: "UNKNOWN",
            2: "ADDRESS",
            3: "PERSON",
        }

    @staticmethod
    def _setup_mock_load_processor(mock_base_processor):
        def side_effect(arg):
            processor = {
                "CharPreprocessor": mock.Mock(spec=data_processing.CharPreprocessor),
                "CharPostprocessor": mock.Mock(spec=data_processing.CharPostprocessor),
            }[arg]
            processor.load_from_disk.return_value = processor
            return processor

        mock_base_processor.get_class.side_effect = side_effect

    def test_has_public_functions(self, *args):
        public_functions = [
            "predict",
            "set_preprocessor",
            "set_model",
            "set_postprocessor",
            "save_to_disk",
            "load_from_disk",
            "load_from_library",
            "check_pipeline",
        ]

        for func in public_functions:
            self.assertTrue(hasattr(UnstructuredDataLabeler, func))

    def test_load_parameters(self, mock_open, mock_load_model, mock_base_processor):

        # test not model class
        with self.assertRaisesRegex(TypeError, "`model_class` must be a BaseModel"):
            load_options = dict(model_class="fake")
            UnstructuredDataLabeler._load_parameters("test/path", load_options)

        # test not correct model class
        with self.assertRaisesRegex(
            ValueError,
            "The load_options model class does not "
            "match the required DataLabeler model.\n "
            ".* != .*",
        ):
            mocked_model = mock.Mock(spec=CharacterLevelCnnModel)
            mocked_model.__class__.__name__ = "FakeClassName"
            load_options = dict(model_class=mocked_model)
            UnstructuredDataLabeler._load_parameters("test/path", load_options)

        # test not preprocessor class
        with self.assertRaisesRegex(
            TypeError, "`preprocessor_class` must be a " "BaseDataPreprocessor"
        ):
            load_options = dict(preprocessor_class="fake")
            UnstructuredDataLabeler._load_parameters("test/path", load_options)

        # test not correct preprocessor class
        with self.assertRaisesRegex(
            ValueError,
            "The load_options preprocessor class does "
            "not match the required DataLabeler "
            "preprocessor.\n .* != .*",
        ):
            mocked_preprocessor = mock.Mock(spec=data_processing.BaseDataPreprocessor)
            mocked_preprocessor.__class__.__name__ = "FakeProcessorName"
            load_options = dict(preprocessor_class=mocked_preprocessor)
            UnstructuredDataLabeler._load_parameters("test/path", load_options)

        # test not postprocessor class
        with self.assertRaisesRegex(
            TypeError, "`postprocessor_class` must be a " "BaseDataPostprocessor"
        ):
            load_options = dict(postprocessor_class="fake")
            UnstructuredDataLabeler._load_parameters("test/path", load_options)

        # test not correct postprocessor class
        with self.assertRaisesRegex(
            ValueError,
            "The load_options postprocessor class does "
            "not match the required DataLabeler "
            "postprocessor.\n .* != .*",
        ):
            mocked_postprocessor = mock.Mock(spec=data_processing.BaseDataPostprocessor)
            mocked_postprocessor.__name__ = "FakeProcessorName"
            load_options = dict(postprocessor_class=mocked_postprocessor)
            UnstructuredDataLabeler._load_parameters("test/path", load_options)

    def test_load_labeler(self, mock_open, mock_load_model, mock_base_processor):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        # load default
        data_labeler = UnstructuredDataLabeler()

        self.assertDictEqual(
            data_labeler.label_mapping, data_labeler_parameters["label_mapping"]
        )
        self.assertListEqual(
            data_labeler.labels, ["PAD", "UNKNOWN", "ADDRESS", "PERSON"]
        )
        self.assertIsInstance(
            data_labeler.preprocessor, data_processing.BaseDataPreprocessor
        )
        self.assertIsInstance(
            data_labeler.postprocessor, data_processing.BaseDataPostprocessor
        )

    def test_load_from_library(self, mock_open, mock_load_model, mock_base_processor):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        # load default
        data_labeler = UnstructuredDataLabeler.load_from_library("default")

        self.assertDictEqual(
            data_labeler.label_mapping, data_labeler_parameters["label_mapping"]
        )
        self.assertListEqual(
            data_labeler.labels, ["PAD", "UNKNOWN", "ADDRESS", "PERSON"]
        )
        self.assertIsInstance(
            data_labeler.preprocessor, data_processing.BaseDataPreprocessor
        )
        self.assertIsInstance(
            data_labeler.postprocessor, data_processing.BaseDataPostprocessor
        )

    def test_load_from_disk(self, mock_open, mock_load_model, mock_base_processor):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        # load default
        data_labeler = UnstructuredDataLabeler.load_from_disk("fake/path")

        self.assertDictEqual(
            data_labeler.label_mapping, data_labeler_parameters["label_mapping"]
        )
        self.assertListEqual(
            data_labeler.labels, ["PAD", "UNKNOWN", "ADDRESS", "PERSON"]
        )
        self.assertIsInstance(
            data_labeler.preprocessor, data_processing.BaseDataPreprocessor
        )
        self.assertIsInstance(
            data_labeler.postprocessor, data_processing.BaseDataPostprocessor
        )

    def test_reverse_label_mappings(
        self, mock_open, mock_load_model, mock_load_processor, *mocks
    ):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        # load default
        data_labeler = UnstructuredDataLabeler()

        reverse_label_mapping = {
            0: "PAD",
            1: "UNKNOWN",
            2: "ADDRESS",
            3: "PERSON",
        }

        self.assertDictEqual(reverse_label_mapping, data_labeler.reverse_label_mapping)

    def test_labels(self, mock_open, mock_load_model, mock_load_processor, *mocks):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        # load default
        labels = ["PAD", "UNKNOWN", "ADDRESS", "PERSON"]
        data_labeler = UnstructuredDataLabeler()

        self.assertListEqual(labels, data_labeler.labels)

    def test_invalid_data_formats(self, *mocks):
        def _invalid_check(data):
            with self.assertRaisesRegex(
                TypeError,
                "Data must be imported using the "
                "data_readers, pd.DataFrames, "
                "np.ndarrays, or lists.",
            ):
                UnstructuredDataLabeler._check_and_return_valid_data_format(data)

        invalid_data = ["string", 1, None, dict()]
        print("\nInvalid Data Checks:")
        for data in invalid_data:
            print("\tChecking data format: {}".format(str(type(data))))
            _invalid_check(data)

        # cannot predict dict
        _invalid_check(data=None)

        # cannot predict dict
        _invalid_check(data={})

    def test_valid_fit_data_formats(self, *mocks):
        def _valid_check(data):
            try:
                print("\tChecking data format: {}".format(str(type(data))))
                data = UnstructuredDataLabeler._check_and_return_valid_data_format(
                    data, fit_or_predict="fit"
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
        ]
        print("\nValid Data Fit Checks:")
        for data in valid_data:
            data = _valid_check(data)
            self.assertIsInstance(data, np.ndarray)

    def test_valid_predict_data_formats(self, *mocks):
        def _valid_check(data):
            try:
                print("\tChecking data format: {}".format(str(type(data))))
                data = UnstructuredDataLabeler._check_and_return_valid_data_format(
                    data, fit_or_predict="predict"
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
            pd.Series([], dtype=object),
        ]
        print("\nValid Data Predict Checks:")
        for data in valid_data:
            data = _valid_check(data)
            self.assertTrue(
                isinstance(data, np.ndarray)
                or isinstance(data, pd.Series)
                or isinstance(data, pd.DataFrame)
            )

    def test_set_params(self, mock_open, mock_load_model, mock_base_processor):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        # load default
        data_labeler = UnstructuredDataLabeler()

        # check empty sent
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "The params dict must have the following "
                "format:\nparams=dict(preprocessor=dict(..."
                "), model=dict(...), postprocessor=dict(..."
                ")), where each sub-dict contains "
                "parameters of the specified data_labeler "
                "pipeline components."
            ),
        ):
            data_labeler.set_params(None)

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "The params dict must have the following "
                "format:\nparams=dict(preprocessor=dict(..."
                "), model=dict(...), postprocessor=dict(..."
                ")), where each sub-dict contains "
                "parameters of the specified data_labeler "
                "pipeline components."
            ),
        ):
            data_labeler.set_params({})

        # test if invalid key sent
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "The params dict must have the following "
                "format:\nparams=dict(preprocessor=dict(..."
                "), model=dict(...), postprocessor=dict(..."
                ")), where each sub-dict contains "
                "parameters of the specified data_labeler "
                "pipeline components."
            ),
        ):
            data_labeler.set_params({"bad key": None})

        # validate no errors occur when correct params are sent
        data_labeler._preprocessor.get_parameters.return_value = dict()
        data_labeler._model.get_parameters.return_value = dict()
        data_labeler._postprocessor.get_parameters.return_value = dict()

        data_labeler.set_params(
            {
                "preprocessor": {"test": 1},
                "model": {"test": 1},
                "postprocessor": {"test2": 3},
            }
        )

        # validate warning on overlaps.
        # here we presume parameters are set as dict(test=1), dict(test=2)
        data_labeler._preprocessor.get_parameters.return_value = dict(test=1)
        data_labeler._model.get_parameters.return_value = dict(test=2)
        with self.assertWarnsRegex(
            RuntimeWarning,
            "Model and preprocessor value for `test` do " "not match. 2 != 1",
        ):
            data_labeler.set_params({"preprocessor": {"test": 1}, "model": {"test": 2}})

        # check if param sent for missing pipeline component
        data_labeler._preprocessor = None
        with self.assertRaisesRegex(
            ValueError,
            "Parameters for the preprocessor, model, or"
            " postprocessor were specified when one or "
            "more of these were not set in the "
            "DataLabeler.",
        ):
            data_labeler.set_params({"preprocessor": {"test": 1}})

        data_labeler._model = None
        with self.assertRaisesRegex(
            ValueError,
            "Parameters for the preprocessor, model, or"
            " postprocessor were specified when one or "
            "more of these were not set in the "
            "DataLabeler.",
        ):
            data_labeler.set_params({"model": {"test": 1}})

        data_labeler._postprocessor = None
        with self.assertRaisesRegex(
            ValueError,
            "Parameters for the preprocessor, model, or"
            " postprocessor were specified when one or "
            "more of these were not set in the "
            "DataLabeler.",
        ):
            data_labeler.set_params({"postprocessor": {"test": 1}})

    def test_set_labels(self, mock_open, mock_load_model, mock_load_processor, *mocks):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        data_labeler = UnstructuredDataLabeler()
        # test label list to label_mapping
        labels = ["a", "b", "d", "c"]
        data_labeler.set_labels(labels)
        mock_load_model.return_value.set_label_mapping.assert_called_with(
            label_mapping=["a", "b", "d", "c"]
        )

        # test label dict to label_mapping
        labels = dict(b=1, c=2, d=3, e=4)
        data_labeler.set_labels(labels)
        mock_load_model.return_value.set_label_mapping.assert_called_with(
            label_mapping=dict(b=1, c=2, d=3, e=4)
        )

    def test_set_model(self, mock_open, mock_load_model, mock_load_processor, *mocks):
        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        # load default
        data_labeler = UnstructuredDataLabeler()

        # test setting model
        model_mock = mock.Mock(spec=CharacterLevelCnnModel)
        data_labeler.set_model(model_mock)
        self.assertEqual(model_mock, data_labeler.model)

        # test failure bc not model object
        with self.assertRaisesRegex(
            TypeError,
            "The specified model was not of the correct" " type, `BaseModel`.",
        ):
            data_labeler.set_model(1)

    def test_set_preprocessor(
        self, mock_open, mock_load_model, mock_load_processor, *mocks
    ):
        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        # load default
        data_labeler = UnstructuredDataLabeler()

        # test setting preprocessor
        processor_mock = mock.Mock(spec=data_processing.CharPreprocessor)
        data_labeler.set_preprocessor(processor_mock)
        self.assertEqual(processor_mock, data_labeler.preprocessor)

        # test failure bc not processing object
        with self.assertRaisesRegex(
            TypeError,
            "The specified preprocessor was not of the "
            "correct type, `DataProcessing`.",
        ):
            data_labeler.set_preprocessor(1)

    def test_set_postprocessor(
        self, mock_open, mock_load_model, mock_load_processor, *mocks
    ):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        # load default
        data_labeler = UnstructuredDataLabeler()

        # test setting postprocessor
        processor_mock = mock.Mock(spec=data_processing.CharPostprocessor)
        data_labeler.set_postprocessor(processor_mock)
        self.assertEqual(processor_mock, data_labeler.postprocessor)

        # test failure bc not processing object
        with self.assertRaisesRegex(
            TypeError,
            "The specified postprocessor was not of "
            "the correct type, `DataProcessing`.",
        ):
            data_labeler.set_postprocessor(1)

    def test_save_to_disk(
        self, mock_open, mock_load_model, mock_load_processor, *mocks
    ):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_load_processor)

        # call func
        data_labeler = UnstructuredDataLabeler()

        # setup save mock
        mock_file = setup_save_mock_open(mock_open)

        # save and test
        data_labeler.save_to_disk("test/path")
        self.assertEqual(
            '{"model": {"class": "CharacterLevelCnnModel"}, '
            '"preprocessor": {"class": "CharPreprocessor"}, '
            '"postprocessor": {"class": "CharPostprocessor"}}',
            mock_file.getvalue(),
        )

        # close mock
        StringIO.close(mock_file)

    def test_check_pipeline(self, mock_open, mock_load_model, mock_base_processor):
        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        data_labeler = UnstructuredDataLabeler()

        # check pipeline with no errors w overlap
        data_labeler._model = mock.Mock()
        data_labeler._model.get_parameters.return_value = dict(a=1)
        data_labeler._model.get_parameters.return_value = dict(a=1)
        data_labeler._preprocessor.get_parameters.return_value = dict(a=1)
        data_labeler._postprocessor.get_parameters.return_value = dict(a=1)
        with warnings.catch_warnings(record=True) as w:
            data_labeler.check_pipeline()
        self.assertEqual(0, len(w))  # assert no warnings raised

        # invalid pipeline, model != preprocessor
        data_labeler._model.get_parameters.return_value = dict(a=1)
        data_labeler.preprocessor.get_parameters.return_value = dict(a=2)
        with self.assertWarnsRegex(
            RuntimeWarning,
            "Model and preprocessor value for `a` do " "not match. 1 != 2",
        ):
            data_labeler.check_pipeline()

        # invalid pipeline, model != postprocessor
        data_labeler.preprocessor.get_parameters.return_value = dict(a=1)
        data_labeler.postprocessor.get_parameters.return_value = dict(a=2)
        with self.assertWarnsRegex(
            RuntimeWarning,
            "Model and postprocessor value for `a` do " "not match. 1 != 2",
        ):
            data_labeler.check_pipeline()

        # invalid pipeline, preprocessor != postprocessor
        data_labeler._model = mock.Mock()
        data_labeler._model.get_parameters.return_value = dict(a=1)
        data_labeler.preprocessor.get_parameters.return_value = dict(a=1, b=1)
        data_labeler.postprocessor.get_parameters.return_value = dict(a=1, b=2)
        with self.assertWarnsRegex(
            RuntimeWarning,
            "Preprocessor and postprocessor value for " "`b` do not match. 1 != 2",
        ):
            data_labeler.check_pipeline()

        # valid pipeline, preprocessor != postprocessor but skips processor
        data_labeler._model = mock.Mock()
        data_labeler._model.get_parameters.return_value = dict(a=1)
        data_labeler.preprocessor.get_parameters.return_value = dict(a=1, b=1)
        data_labeler.postprocessor.get_parameters.return_value = dict(a=1, b=2)
        with warnings.catch_warnings(record=True) as w:
            data_labeler.check_pipeline(skip_postprocessor=True)
        self.assertEqual(0, len(w))

        # assert raises error instead of warning
        with self.assertRaisesRegex(
            RuntimeError,
            "Preprocessor and postprocessor value for " "`b` do not match. 1 != 2",
        ):
            data_labeler.check_pipeline(error_on_mismatch=True)

    def test_fit(self, mock_open, mock_load_model, mock_base_processor):
        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        def reset_mocks(dl):
            dl.preprocessor.process.reset_mock()
            dl._model.set_label_mapping.reset_mock()
            dl._model.reset_weights.reset_mock()
            dl._model.fit.reset_mock()

        # setup data
        samples = [["1"], ["2"], ["3"], ["4"], ["5"]]
        labels = [[1], [2], [3], [4], [5]]
        data = list(zip(samples, labels))
        fake_results = dict(pred="fake pred", conf="fake conf")
        fake_results_no_conf = dict(pred="fake pred")

        # setup mocks
        data_labeler = DataLabeler(labeler_type="unstructured", trainable=True)
        data_labeler._model = mock.Mock()
        data_labeler._preprocessor.get_parameters.return_value = dict()
        data_labeler._model.get_parameters.return_value = dict()
        data_labeler._model.label_mapping = dict(fake=1)
        data_labeler._model.requires_zero_mapping = True
        data_labeler._model.fit.return_value = "fake result"

        # test empty data
        with self.assertRaisesRegex(ValueError, "No data or labels to fit."):
            data_labeler.fit(x=[], y=[])

        # test default
        output = data_labeler.fit(x=samples, y=labels)
        self.assertEqual(2, data_labeler.preprocessor.process.call_count)
        data_labeler._model.set_label_mapping.assert_not_called()
        data_labeler._model.reset_weights.assert_not_called()
        self.assertEqual(1, data_labeler._model.fit.call_count)
        self.assertEqual(1, len(output))

        # test validation_split = 1
        with self.assertRaisesRegex(
            ValueError, "`validation_split` must be >= 0 and less " "than 1.0"
        ):
            data_labeler.fit(x=samples, y=labels, validation_split=1)

        # test validation_split = 0
        reset_mocks(data_labeler)

        output = data_labeler.fit(x=samples, y=labels, validation_split=0)
        self.assertEqual(1, data_labeler.preprocessor.process.call_count)
        data_labeler._model.set_label_mapping.assert_not_called()
        data_labeler._model.reset_weights.assert_not_called()
        self.assertEqual(1, data_labeler._model.fit.call_count)
        self.assertEqual(1, len(output))

        # test labels
        reset_mocks(data_labeler)

        output = data_labeler.fit(x=samples, y=labels, labels=["new", "labels"])
        self.assertEqual(2, data_labeler.preprocessor.process.call_count)
        data_labeler._model.set_label_mapping.assert_called_with(
            label_mapping=["new", "labels"]
        )
        data_labeler._model.reset_weights.assert_not_called()
        self.assertEqual(1, data_labeler._model.fit.call_count)
        self.assertEqual(1, len(output))

        # test reset_weights=True
        reset_mocks(data_labeler)

        output = data_labeler.fit(x=samples, y=labels, reset_weights=True)
        self.assertEqual(2, data_labeler.preprocessor.process.call_count)
        data_labeler._model.set_label_mapping.assert_not_called()
        data_labeler._model.reset_weights.assert_called_once()
        self.assertEqual(1, data_labeler._model.fit.call_count)
        self.assertEqual(1, len(output))

        # test multiple epochs
        reset_mocks(data_labeler)

        output = data_labeler.fit(x=samples, y=labels, epochs=3)
        self.assertEqual(6, data_labeler.preprocessor.process.call_count)
        data_labeler._model.set_label_mapping.assert_not_called()
        data_labeler._model.reset_weights.assert_not_called()
        self.assertEqual(3, data_labeler._model.fit.call_count)
        self.assertEqual(3, len(output))
