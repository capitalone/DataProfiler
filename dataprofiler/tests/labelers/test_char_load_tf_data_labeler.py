import json
import os
import unittest
from io import StringIO
from unittest import mock

from dataprofiler.labelers import DataLabeler, UnstructuredDataLabeler, data_processing
from dataprofiler.labelers.char_load_tf_model import CharLoadTFModel

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data_labeler_parameters = {
    "model": {"class": "CharLoadTFModel", "parameters": {}},
    "label_mapping": {
        "PAD": 0,
        "CITY": 1,  # SAME AS UNKNOWN
        "UNKNOWN": 1,
        "ADDRESS": 2,
        "PERSON": 3,
    },
    "preprocessor": {"class": "CharEncodedPreprocessor"},
    "postprocessor": {"class": "CharPostprocessor"},
}

preprocessor_parameters = {
    "encoding_map": {"t": 1, "s": 2},
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
    "dataprofiler.labelers.char_load_tf_model." "CharLoadTFModel.load_from_disk"
)
@mock.patch("builtins.open", side_effect=mock_open)
class TestCharTFLoadDataLabeler(unittest.TestCase):
    @staticmethod
    def _setup_mock_load_model(mock_load_model):
        model_mock = mock.Mock(spec=CharLoadTFModel)
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
                "CharEncodedPreprocessor": mock.Mock(
                    spec=data_processing.CharEncodedPreprocessor
                ),
                "CharPostprocessor": mock.Mock(spec=data_processing.CharPostprocessor),
            }[arg]
            processor.load_from_disk.return_value = processor
            return processor

        mock_base_processor.get_class.side_effect = side_effect

    def test_load_from_disk(self, mock_open, mock_load_model, mock_base_processor):

        self._setup_mock_load_model(mock_load_model)
        self._setup_mock_load_processor(mock_base_processor)

        # load default
        data_labeler = DataLabeler.load_from_disk("fake/path")

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
            '{"model": {"class": "CharLoadTFModel"}, '
            '"preprocessor": {"class": "CharEncodedPreprocessor"}, '
            '"postprocessor": {"class": "CharPostprocessor"}}',
            mock_file.getvalue(),
        )

        # close mock
        StringIO.close(mock_file)
