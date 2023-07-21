import json
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.labelers import BaseDataLabeler
from dataprofiler.profilers import profiler_utils
from dataprofiler.profilers.data_labeler_column_profile import DataLabelerColumn
from dataprofiler.profilers.json_decoder import load_column_profile
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import DataLabelerOptions

from . import utils as test_utils


@mock.patch(
    "dataprofiler.profilers.data_labeler_column_profile.DataLabeler",
    spec=BaseDataLabeler,
)
class TestDataLabelerColumnProfiler(unittest.TestCase):
    @staticmethod
    def _setup_data_labeler_mock(mock_instance):
        mock_DataLabeler = mock_instance.return_value
        mock_DataLabeler.label_mapping = {"a": 0, "b": 1}
        mock_DataLabeler.reverse_label_mapping = {0: "a", 1: "b"}
        mock_DataLabeler.model.num_labels = 2
        mock_DataLabeler.model.requires_zero_mapping = False
        mock_instance.load_from_library.side_effect = mock_instance

        def mock_predict(data, *args, **kwargs):
            len_data = len(data)
            output = [[1, 0], [0, 1]] * (len_data // 2)
            if len_data % 2:
                output += [[1, 0]]
            conf = np.array(output)
            if mock_DataLabeler.model.requires_zero_mapping:
                conf = np.concatenate([[[0]] * len_data, conf], axis=1)
            pred = np.argmax(conf, axis=1)
            return {"pred": pred, "conf": conf}

        mock_DataLabeler.predict.side_effect = mock_predict

    def test_base_case(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        data = pd.Series([], dtype=object)
        profiler = DataLabelerColumn(data.name)

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            profiler.update(data)

            self.assertEqual(0, profiler.sample_size)
            self.assertEqual(["a", "b"], profiler.possible_data_labels)
            self.assertEqual(None, profiler.data_label)
            self.assertEqual(None, profiler.avg_predictions)
            self.assertCountEqual(
                ["data_label", "avg_predictions", "data_label_representation", "times"],
                list(profiler.profile.keys()),
            )
            self.assertEqual(
                {
                    "data_label": None,
                    "avg_predictions": None,
                    "data_label_representation": None,
                    "times": defaultdict(),
                },
                profiler.profile,
            )

    def test_update(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        data = pd.Series(["1", "2", "3"])
        profiler = DataLabelerColumn(data.name)
        profiler.update(data)

        self.assertEqual(3, profiler.sample_size)
        self.assertEqual(["a", "b"], profiler.possible_data_labels)
        self.assertEqual("a", profiler.data_label)
        self.assertDictEqual(dict(a=2 / 3, b=1 / 3), profiler.avg_predictions)
        self.assertDictEqual(dict(a=2, b=1), profiler.rank_distribution)
        self.assertDictEqual(dict(a=2 / 3, b=1 / 3), profiler.label_representation)

    def test_update_reserve_label_mapping(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)
        mock_DataLabeler = mock_instance.return_value
        mock_DataLabeler.model.requires_zero_mapping = True
        mock_DataLabeler.label_mapping = {"PAD": 0, "a": 1, "b": 2}
        mock_DataLabeler.reverse_label_mapping = {0: "PAD", 1: "a", 2: "b"}
        mock_DataLabeler.model.num_labels = 3

        data = pd.Series(["1", "2", "3"])
        profiler = DataLabelerColumn(data.name)
        profiler.update(data)

        self.assertEqual(["a", "b"], profiler.possible_data_labels)
        self.assertEqual("a", profiler.data_label)
        self.assertDictEqual(dict(a=2 / 3, b=1 / 3), profiler.avg_predictions)
        self.assertDictEqual(dict(a=2, b=1), profiler.rank_distribution)
        self.assertDictEqual(dict(a=2 / 3, b=1 / 3), profiler.label_representation)

    def test_data_label_low_accuracy(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        def mock_low_predict(data, *args, **kwargs):
            return {"pred": np.array([[0, 0]]), "conf": np.array([[0.2, 0.2]])}

        mock_instance.return_value.predict.side_effect = mock_low_predict

        data = pd.Series(["1"])
        profiler = DataLabelerColumn(data.name)
        profiler.update(data)
        self.assertEqual("could not determine", profiler.data_label)

    def test_multi_labels(self, mock_instance):
        mock_DataLabeler = mock_instance.return_value
        mock_DataLabeler.label_mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
        mock_DataLabeler.reverse_label_mapping = {0: "a", 1: "b", 2: "c", 3: "d"}
        mock_DataLabeler.model.num_labels = 4
        mock_DataLabeler.model.requires_zero_mapping = False

        def mock_low_predict(data, *args, **kwargs):
            return {
                "pred": None,
                "conf": np.array(
                    [
                        [1, 0, 0, 0],  # 4 repeated
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],  # 2 repeated
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],  # 3 repeated
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],  # 1 repeated
                    ]
                ),
            }  # counts [4, 2, 3, 1] => [a, b, c, d]

        mock_instance.return_value.predict.side_effect = mock_low_predict

        data = pd.Series(["1"] * 10)
        profiler = DataLabelerColumn(data.name)
        profiler.update(data)
        self.assertEqual("a|c|b", profiler.data_label)

    def test_profile(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        data = pd.Series(["1", "2", "3"])
        profiler = DataLabelerColumn(data.name)

        expected_profile = {
            "data_label": "a",
            "avg_predictions": dict(a=2 / 3, b=1 / 3),
            "data_label_representation": dict(a=2 / 3, b=1 / 3),
            "times": defaultdict(float, {"data_labeler_predict": 1.0}),
        }

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile["times"])
            profiler.update(data)

            # Validate the time in the DataLabeler class has the expected time.
            profile = profiler.profile
            self.assertDictEqual(expected_profile, profile)

            # Validate time in datetime class has expected time after second update
            profiler.update(data)
            expected = defaultdict(float, {"data_labeler_predict": 2.0})
            self.assertEqual(expected, profiler.profile["times"])

    def test_report(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        data = pd.Series(["1", "2", "3"])
        profile = DataLabelerColumn(data.name)

        report1 = profile.profile
        report2 = profile.report(remove_disabled_flag=False)
        report3 = profile.report(remove_disabled_flag=True)
        self.assertDictEqual(report1, report2)
        self.assertDictEqual(report1, report3)

    def test_label_match(self, mock_instance):
        """
        Test label match between avg_prediction and data_label_representation
        """
        mock_DataLabeler = mock_instance.return_value
        mock_DataLabeler.label_mapping = {
            "a": 0,
            "b": 1,
            "c": 1,
            "d": 2,
            "e": 2,
            "f": 3,
        }
        mock_DataLabeler.reverse_label_mapping = {0: "a", 1: "c", 2: "e", 3: "f"}
        mock_DataLabeler.model.num_labels = 4
        mock_DataLabeler.model.requires_zero_mapping = False

        data = pd.Series(["1", "2", "3", "4", "5", "6"])
        profiler = DataLabelerColumn(data.name)
        profiler.sample_size = 1

        self.assertEqual(["a", "c", "e", "f"], profiler.possible_data_labels)
        self.assertDictEqual(dict(a=0, c=0, e=0, f=0), profiler.label_representation)
        self.assertDictEqual(dict(a=0, c=0, e=0, f=0), profiler.avg_predictions)

    def test_profile_merge(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        data = pd.Series(["1", "2", "3", "11"])
        data2 = pd.Series(["4", "5", "6", "7", "9", "10", "12"])

        expected_profile = {
            "data_label": "a|b",
            "avg_predictions": dict(a=54 / 99, b=45 / 99),
            "data_label_representation": dict(a=54 / 99, b=45 / 99),
            "times": defaultdict(float, {"data_labeler_predict": 2.0}),
        }
        expected_sum_predictions = [6, 5]
        expected_rank_distribution = {"a": 6, "b": 5}

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            profiler = DataLabelerColumn(data.name)
            profiler.update(data)

            profiler2 = DataLabelerColumn(data2.name)
            profiler2.update(data2)

            profiler3 = profiler + profiler2

            # Assert correct values
            self.assertEqual(expected_profile, profiler3.profile)
            self.assertEqual(
                expected_sum_predictions, profiler3.sum_predictions.tolist()
            )
            self.assertEqual(expected_rank_distribution, profiler3.rank_distribution)
            self.assertEqual(expected_profile, profiler3.profile)
            self.assertEqual(profiler.data_labeler, profiler3.data_labeler)
            self.assertEqual(
                profiler.possible_data_labels, profiler3.possible_data_labels
            )
            self.assertEqual(profiler._top_k_voting, profiler3._top_k_voting)
            self.assertEqual(profiler._min_voting_prob, profiler3._min_voting_prob)
            self.assertEqual(
                profiler._min_prob_differential, profiler3._min_prob_differential
            )
            self.assertEqual(profiler._top_k_labels, profiler3._top_k_labels)
            self.assertEqual(
                profiler._min_top_label_prob, profiler3._min_top_label_prob
            )
            self.assertEqual(profiler._max_sample_size, profiler3._max_sample_size)
            self.assertEqual(profiler._top_k_voting, profiler3._top_k_voting)

            # Check adding even more profiles together
            profiler3 = profiler + profiler3
            expected_profile = {
                "data_label": "a|b",
                "avg_predictions": dict(a=8 / 15, b=7 / 15),
                "data_label_representation": dict(a=8 / 15, b=7 / 15),
                "times": defaultdict(float, {"data_labeler_predict": 3.0}),
            }
            expected_sum_predictions = [8, 7]
            expected_rank_distribution = {"a": 8, "b": 7}

            # Assert only the proper changes have been made
            self.assertEqual(expected_profile, profiler3.profile)
            self.assertEqual(
                expected_sum_predictions, profiler3._sum_predictions.tolist()
            )
            self.assertEqual(expected_rank_distribution, profiler3.rank_distribution)
            self.assertEqual(expected_profile, profiler3.profile)
            self.assertEqual(profiler.data_labeler, profiler3.data_labeler)
            self.assertEqual(
                profiler.possible_data_labels, profiler3.possible_data_labels
            )
            self.assertEqual(profiler._top_k_voting, profiler3._top_k_voting)
            self.assertEqual(profiler._min_voting_prob, profiler3._min_voting_prob)
            self.assertEqual(
                profiler._min_prob_differential, profiler3._min_prob_differential
            )
            self.assertEqual(profiler._top_k_labels, profiler3._top_k_labels)
            self.assertEqual(
                profiler._min_top_label_prob, profiler3._min_top_label_prob
            )
            self.assertEqual(profiler._max_sample_size, profiler3._max_sample_size)
            self.assertEqual(profiler._top_k_voting, profiler3._top_k_voting)

        # Check that error is thrown if profiles are unequal
        with self.assertRaises(ValueError):
            profiler._top_k_voting = 13
            test = profiler + profiler2

    def test_data_labeler_column_with_wrong_options(self, *mocks):
        with self.assertRaisesRegex(
            ValueError,
            "DataLabelerColumn parameter 'options' must "
            "be of type DataLabelerOptions.",
        ):
            profiler = DataLabelerColumn("Data_Labeler", options="wrong_data_type")

    def test_profile_merge_with_different_options(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        # Different max_sample_size values
        data = pd.Series(["1", "2", "3", "11"])
        data2 = pd.Series(["4", "5", "6", "7", "9", "10", "12"])
        options = DataLabelerOptions()
        options.max_sample_size = 20
        profiler = DataLabelerColumn(data.name, options=options)
        profiler.update(data)

        options2 = DataLabelerOptions()
        options2.max_sample_size = 15
        profiler2 = DataLabelerColumn(data2.name, options=options2)
        profiler2.update(data2)

        with self.assertRaisesRegex(
            AttributeError,
            "Cannot merge. The data labeler and/or the max "
            "sample size are not the same for both column "
            "profiles.",
        ):
            profiler3 = profiler + profiler2

        # Different labelers
        profiler = DataLabelerColumn(data.name)
        profiler.data_labeler = mock.MagicMock()
        profiler2 = DataLabelerColumn(data2.name)

        with self.assertRaisesRegex(
            AttributeError,
            "Cannot merge. The data labeler and/or the max "
            "sample size are not the same for both column "
            "profiles.",
        ):
            profiler3 = profiler + profiler2

    def test_diff(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)

        profiler1 = DataLabelerColumn("")
        profiler2 = DataLabelerColumn("")

        # Mock out the data_label, avg_predictions, and label_representation
        # properties
        with mock.patch(
            "dataprofiler.profilers.data_labeler_column_profile"
            ".DataLabelerColumn.data_label"
        ), mock.patch(
            "dataprofiler.profilers.data_labeler_column_profile."
            "DataLabelerColumn.avg_predictions"
        ), mock.patch(
            "dataprofiler.profilers.data_labeler_column_profile."
            "DataLabelerColumn.label_representation"
        ):
            profiler1.sample_size = 10
            profiler1.data_label = "a|b|c"
            profiler1.avg_predictions = {"a": 0.25, "b": 0.0, "c": 0.75}
            profiler1.label_representation = {"a": 0.15, "b": 0.01, "c": 0.84}

            profiler2.sample_size = 10
            profiler2.data_label = "b|c|d"
            profiler2.avg_predictions = {"a": 0.25, "b": 0.70, "c": 0.05}
            profiler2.label_representation = {"a": 0.99, "b": 0.01, "c": 0.0}

            diff = profiler1.diff(profiler2)
            expected_diff = {
                "data_label": profiler_utils.find_diff_of_lists_and_sets(
                    ["a", "b", "c"], ["b", "c", "d"]
                ),
                "avg_predictions": {"a": "unchanged", "b": -0.70, "c": 0.70},
                "label_representation": {"a": -0.84, "b": "unchanged", "c": 0.84},
            }
            self.assertDictEqual(expected_diff, diff)

    def test_empty_data(self, *mocks):
        # self._setup_data_labeler_mock(mock_instance)

        profiler1 = DataLabelerColumn("")
        profiler2 = DataLabelerColumn("")

        # Mock out the data_label, avg_predictions, and label_representation
        # properties
        profiler1.update(pd.Series())
        profiler2.update(pd.Series())

        merge_profile = profiler1 + profiler2
        self.assertIsNone(merge_profile._rank_distribution)

        diff_profile = profiler1.diff(profiler2)
        self.assertIsNone(merge_profile.data_label)

    def test_json_encode(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)
        profiler = DataLabelerColumn("")

        # Validates that error is raised if model loc is not set for labeler
        profiler.data_labeler._default_model_loc = None
        with self.assertRaisesRegex(
            ValueError,
            "Serialization cannot be done on labelers with _default_model_loc not set",
        ):
            _ = json.dumps(profiler, cls=ProfileEncoder)

        # Reset the model loc to its initial value
        profiler.data_labeler._default_model_loc = "this is a test model loc"
        serialized = json.dumps(profiler, cls=ProfileEncoder)

        expected = json.dumps(
            {
                "class": "DataLabelerColumn",
                "data": {
                    "name": "",
                    "col_index": float("nan"),
                    "sample_size": 0,
                    "metadata": {},
                    "times": {},
                    "thread_safe": False,
                    "_max_sample_size": 1000,
                    "data_labeler": {"from_library": "this is a test model loc"},
                    "_reverse_label_mapping": None,
                    "_possible_data_labels": None,
                    "_rank_distribution": None,
                    "_sum_predictions": None,
                    "_top_k_voting": 1,
                    "_min_voting_prob": 0.2,
                    "_min_prob_differential": 0.2,
                    "_top_k_labels": 3,
                    "_min_top_label_prob": 0.35,
                    "_DataLabelerColumn__calculations": {},
                },
            }
        )

        self.assertEqual(serialized, expected)

    def test_json_encode_after_update(self, mock_instance):
        self._setup_data_labeler_mock(mock_instance)
        data = pd.Series(["1", "2", "3", "4"], dtype=object)
        profiler = DataLabelerColumn(data.name)
        profiler.data_labeler._default_model_loc = "this is a test model loc"
        with test_utils.mock_timeit():
            profiler.update(data)

        serialized = json.dumps(profiler, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "DataLabelerColumn",
                "data": {
                    "name": None,
                    "col_index": float("nan"),
                    "sample_size": 4,
                    "metadata": {},
                    "times": {"data_labeler_predict": 1.0},
                    "thread_safe": False,
                    "_max_sample_size": 1000,
                    "data_labeler": {"from_library": "this is a test model loc"},
                    "_reverse_label_mapping": {0: "a", 1: "b"},
                    "_possible_data_labels": ["a", "b"],
                    "_rank_distribution": {
                        "a": 2,
                        "b": 2,
                    },
                    "_sum_predictions": [2.0, 2.0],
                    "_top_k_voting": 1,
                    "_min_voting_prob": 0.2,
                    "_min_prob_differential": 0.2,
                    "_top_k_labels": 3,
                    "_min_top_label_prob": 0.35,
                    "_DataLabelerColumn__calculations": {},
                },
            }
        )

        self.assertEqual(expected, serialized)

    @mock.patch(
        "dataprofiler.profilers.profiler_utils.DataLabeler", spec=BaseDataLabeler
    )
    def test_json_decode(self, mock_utils_DataLabeler, mock_BaseDataLabeler):
        self._setup_data_labeler_mock(mock_BaseDataLabeler)
        mock_utils_DataLabeler.load_from_library.side_effect = mock_BaseDataLabeler

        data = pd.Series(["1", "2", "3", "4"], dtype=object)
        expected = DataLabelerColumn(data.name)
        expected.data_labeler._default_model_loc = "structured_model"
        serialized = json.dumps(expected, cls=ProfileEncoder)

        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected)

        # test decode with options to override load labeler
        # create a new labeler ot load instead of from_library
        new_mock_data_labeler = mock.Mock(spec=BaseDataLabeler)
        new_mock_data_labeler.name = "new fake data labeler"
        new_mock_data_labeler._default_model_loc = "my/fake/path"
        config = {
            "DataLabelerColumn": {
                "from_library": {"structured_model": new_mock_data_labeler}
            }
        }

        mock_BaseDataLabeler.reset_mock()  # set to 0 calls as option should override
        mock_utils_DataLabeler.reset_mock()  # set to 0 calls as option should override
        deserialized = load_column_profile(json.loads(serialized), config)
        assert deserialized.data_labeler == new_mock_data_labeler
        mock_BaseDataLabeler.assert_not_called()
        mock_utils_DataLabeler.assert_not_called()

        # validate raises error when cannot properly load data.
        with self.assertRaisesRegex(
            NotImplementedError,
            "Models intialized from disk have not yet been made deserializable",
        ):
            class_as_dict = json.loads(serialized)
            class_as_dict["data"]["data_labeler"] = {"from_disk": "test"}
            deserialized = load_column_profile(class_as_dict, config)

    @mock.patch(
        "dataprofiler.profilers.profiler_utils.DataLabeler", spec=BaseDataLabeler
    )
    def test_json_decode_after_update(
        self, mock_utils_DataLabeler, mock_BaseDataLabeler
    ):
        self._setup_data_labeler_mock(mock_BaseDataLabeler)
        mock_utils_DataLabeler.load_from_library.side_effect = mock_BaseDataLabeler

        data = pd.Series(["1", "2", "3", "4"], dtype=object)
        expected = DataLabelerColumn(data.name)
        expected.data_labeler._default_model_loc = "structured_model"
        with test_utils.mock_timeit():
            expected.update(data)

        serialized = json.dumps(expected, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected)
        update_data = pd.Series(["4", "5", "6", "7"], dtype=object)
        deserialized.update(update_data)

        assert deserialized.sample_size == 8
        self.assertDictEqual({"a": 4, "b": 4}, deserialized.rank_distribution)
        np.testing.assert_array_equal(
            np.array([4.0, 4.0]), deserialized.sum_predictions
        )
