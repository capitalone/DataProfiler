import unittest
from collections import defaultdict
from unittest import mock

import pandas as pd

from dataprofiler.profilers import utils
from dataprofiler.profilers.unstructured_labeler_profile import (
    UnstructuredLabelerProfile,
)


class TestUnstructuredLabelerProfile(unittest.TestCase):
    def test_char_level_counts(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(["abc123", "Bob", "!@##$%"])

        # running update
        default.update(sample)
        # now getting entity_counts to check for existence
        self.assertIsNotNone(default.profile["entity_counts"]["true_char_level"])

        self.assertIsNotNone(default.profile["entity_counts"]["postprocess_char_level"])

        # assert it's not empty for now
        self.assertIsNotNone(default.profile)

        # then assert that correctly counted number of char samples
        self.assertEqual(default.char_sample_size, 15)

    def test_advanced_sample(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(
            [
                "Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
                "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912."
                "\n",
                "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n",
            ]
        )

        # running update
        default.update(sample)

        # now getting entity_counts to check for proper structure
        self.assertIsNotNone(default.profile["entity_counts"]["true_char_level"])

        # assert it's not empty for now
        self.assertIsNotNone(default.profile)

    def test_word_level_NER_label_counts(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(
            [
                "Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
                "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000049939232194912."
                "\n",
                "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n",
            ]
        )

        # running update
        default.update(sample)

        # now getting entity_counts to check for proper structure
        self.assertIsNotNone(default.profile["entity_counts"]["word_level"])

        # assert it's not empty for now
        self.assertIsNotNone(default.profile)

    def test_statistics(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(
            [
                "Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
                "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912."
                "\n",
                "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n",
            ]
        )

        # running update
        default.update(sample)

        self.assertIsNotNone(default.entity_percentages["word_level"])
        self.assertIsNotNone(default.entity_percentages["true_char_level"])
        self.assertIsNotNone(default.entity_percentages["postprocess_char_level"])
        current_word_sample_size = default.word_sample_size
        current_char_sample_size = default.char_sample_size
        self.assertIsNotNone(default.word_sample_size)
        self.assertIsNotNone(default.char_sample_size)
        self.assertIsNotNone(default.entity_counts["word_level"])
        self.assertIsNotNone(default.entity_counts["true_char_level"])
        self.assertIsNotNone(default.entity_counts["postprocess_char_level"])
        self.assertIsNone(default._get_percentages("WRONG_INPUT"))

        default.update(sample)
        self.assertNotEqual(current_word_sample_size, default.word_sample_size)
        self.assertNotEqual(current_char_sample_size, default.char_sample_size)

    @mock.patch("dataprofiler.profilers." "unstructured_labeler_profile.DataLabeler")
    @mock.patch(
        "dataprofiler.profilers." "unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_profile(self, processor_class_mock, model_class_mock):
        # setup mocks
        model_mock = mock.Mock()
        model_mock.reverse_label_mapping = {1: "UNKNOWN"}
        model_mock.predict.return_value = dict(pred=[[1]])
        model_class_mock.return_value = model_mock
        processor_mock = mock.Mock()
        processor_mock.process.return_value = dict(pred=[[]])
        processor_class_mock.return_value = processor_mock

        # initialize labeler profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(["a"])
        expected_profile = dict(
            entity_counts={
                "postprocess_char_level": defaultdict(int, {"UNKNOWN": 1}),
                "true_char_level": defaultdict(int, {"UNKNOWN": 1}),
                "word_level": defaultdict(int),
            },
            entity_percentages={
                "postprocess_char_level": defaultdict(int, {"UNKNOWN": 1.0}),
                "true_char_level": defaultdict(int, {"UNKNOWN": 1.0}),
                "word_level": defaultdict(int),
            },
            times=defaultdict(float, {"data_labeler_predict": 1.0}),
        )

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            default.update(sample)
        profile = default.profile

        # key and value populated correctly
        self.assertDictEqual(expected_profile, profile)

    @mock.patch("dataprofiler.profilers." "unstructured_labeler_profile.DataLabeler")
    @mock.patch(
        "dataprofiler.profilers." "unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_report(self, processor_class_mock, model_class_mock):
        # setup mocks
        model_mock = mock.Mock()
        model_mock.reverse_label_mapping = {1: "UNKNOWN"}
        model_mock.predict.return_value = dict(pred=[[1]])
        model_class_mock.return_value = model_mock
        processor_mock = mock.Mock()
        processor_mock.process.return_value = dict(pred=[[]])
        processor_class_mock.return_value = processor_mock

        # initialize labeler profile
        profile = UnstructuredLabelerProfile()

        sample = pd.Series(["a"])

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            profile.update(sample)

        report1 = profile.profile
        report2 = profile.report(remove_disabled_flag=False)
        report3 = profile.report(remove_disabled_flag=True)
        self.assertDictEqual(report1, report2)
        self.assertDictEqual(report1, report3)

    @mock.patch("dataprofiler.profilers." "unstructured_labeler_profile.DataLabeler")
    @mock.patch(
        "dataprofiler.profilers." "unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_entity_percentages(self, mock1, mock2):
        """
        Tests to see that entity percentages match the counts given
        """
        profile = UnstructuredLabelerProfile()
        profile.char_sample_size = 20
        profile.word_sample_size = 10
        profile.entity_counts["postprocess_char_level"]["UNKNOWN"] = 6
        profile.entity_counts["postprocess_char_level"]["TEST"] = 14
        profile.entity_counts["true_char_level"]["UNKNOWN"] = 4
        profile.entity_counts["true_char_level"]["TEST"] = 16
        profile.entity_counts["word_level"]["UNKNOWN"] = 5
        profile.entity_counts["word_level"]["TEST"] = 5
        profile.update(pd.Series(["a"]))

        expected_percentages = {
            "postprocess_char_level": defaultdict(int, {"UNKNOWN": 0.3, "TEST": 0.7}),
            "true_char_level": defaultdict(int, {"UNKNOWN": 0.2, "TEST": 0.8}),
            "word_level": defaultdict(int, {"UNKNOWN": 0.5, "TEST": 0.5}),
        }

        percentages = profile.profile["entity_percentages"]

        self.assertDictEqual(expected_percentages, percentages)

    @mock.patch("dataprofiler.profilers." "unstructured_labeler_profile.DataLabeler")
    def test_unstructured_labeler_profile_add(self, mock):
        # Test empty merge
        profile1 = UnstructuredLabelerProfile()
        profile2 = UnstructuredLabelerProfile()
        merged_profile = profile1 + profile2

        self.assertDictEqual(merged_profile.entity_counts["word_level"], {})
        self.assertDictEqual(merged_profile.entity_counts["true_char_level"], {})
        self.assertDictEqual(merged_profile.entity_counts["postprocess_char_level"], {})
        self.assertEqual(merged_profile.word_sample_size, 0)
        self.assertEqual(merged_profile.char_sample_size, 0)

        # Test merge with data
        profile1.word_sample_size = 7
        profile1.char_sample_size = 6
        profile1.entity_counts["word_level"]["UNKNOWN"] = 5
        profile1.entity_counts["word_level"]["TEST"] = 2
        profile1.entity_counts["true_char_level"]["PAD"] = 6
        profile1.entity_counts["postprocess_char_level"]["UNKNOWN"] = 3

        profile2.word_sample_size = 4
        profile2.char_sample_size = 4
        profile2.entity_counts["word_level"]["UNKNOWN"] = 3
        profile2.entity_counts["word_level"]["PAD"] = 1
        profile2.entity_counts["postprocess_char_level"]["UNKNOWN"] = 2

        merged_profile = profile1 + profile2
        expected_word_level = {"UNKNOWN": 8, "TEST": 2, "PAD": 1}
        expected_true_char = {"PAD": 6}
        expected_post_char = {"UNKNOWN": 5}

        self.assertDictEqual(
            merged_profile.entity_counts["word_level"], expected_word_level
        )
        self.assertDictEqual(
            merged_profile.entity_counts["true_char_level"], expected_true_char
        )
        self.assertDictEqual(
            merged_profile.entity_counts["postprocess_char_level"], expected_post_char
        )

        self.assertEqual(merged_profile.word_sample_size, 11)
        self.assertEqual(merged_profile.char_sample_size, 10)

        self.assertEqual(
            merged_profile.times["data_labeler_predict"],
            profile1.times["data_labeler_predict"]
            + profile2.times["data_labeler_predict"],
        )

    @mock.patch("dataprofiler.profilers." "unstructured_labeler_profile.DataLabeler")
    @mock.patch(
        "dataprofiler.profilers." "unstructured_labeler_profile." "CharPostprocessor"
    )
    def test_diff(self, mock1, mock2):
        """
        Tests to see that entity percentages match the counts given
        """
        profiler1 = UnstructuredLabelerProfile()
        profiler1.char_sample_size = 20
        profiler1.word_sample_size = 15
        profiler1.entity_counts["postprocess_char_level"]["UNKNOWN"] = 5
        profiler1.entity_counts["postprocess_char_level"]["TEST"] = 10
        profiler1.entity_counts["postprocess_char_level"]["UNIQUE1"] = 5
        profiler1.entity_counts["true_char_level"]["UNKNOWN"] = 4
        profiler1.entity_counts["true_char_level"]["TEST"] = 8
        profiler1.entity_counts["true_char_level"]["UNIQUE1"] = 8
        profiler1.entity_counts["word_level"]["UNKNOWN"] = 5
        profiler1.entity_counts["word_level"]["TEST"] = 5
        profiler1.entity_counts["word_level"]["UNIQUE1"] = 5
        profiler1.update(pd.Series(["a"]))

        profiler2 = UnstructuredLabelerProfile()
        profiler2.char_sample_size = 20
        profiler2.word_sample_size = 10
        profiler2.entity_counts["postprocess_char_level"]["UNKNOWN"] = 5
        profiler2.entity_counts["postprocess_char_level"]["TEST"] = 10
        profiler2.entity_counts["postprocess_char_level"]["UNIQUE2"] = 5
        profiler2.entity_counts["true_char_level"]["UNKNOWN"] = 8
        profiler2.entity_counts["true_char_level"]["TEST"] = 8
        profiler2.entity_counts["true_char_level"]["UNIQUE2"] = 4
        profiler2.entity_counts["word_level"]["UNKNOWN"] = 2
        profiler2.entity_counts["word_level"]["TEST"] = 4
        profiler2.entity_counts["word_level"]["UNIQUE2"] = 4
        profiler2.update(pd.Series(["a"]))

        expected_diff = {
            "entity_counts": {
                "postprocess_char_level": {
                    "UNKNOWN": "unchanged",
                    "TEST": "unchanged",
                    "UNIQUE1": [5, None],
                    "UNIQUE2": [None, 5],
                },
                "true_char_level": {
                    "UNKNOWN": -4,
                    "TEST": "unchanged",
                    "UNIQUE1": [8, None],
                    "UNIQUE2": [None, 4],
                },
                "word_level": {
                    "UNKNOWN": 3,
                    "TEST": 1,
                    "UNIQUE1": [5, None],
                    "UNIQUE2": [None, 4],
                },
            },
            "entity_percentages": {
                "postprocess_char_level": {
                    "UNKNOWN": "unchanged",
                    "TEST": "unchanged",
                    "UNIQUE1": [1 / 4, None],
                    "UNIQUE2": [None, 1 / 4],
                },
                "true_char_level": {
                    "UNKNOWN": -1 / 5,
                    "TEST": "unchanged",
                    "UNIQUE1": [2 / 5, None],
                    "UNIQUE2": [None, 1 / 5],
                },
                "word_level": {
                    "UNKNOWN": 1 / 3 - 1 / 5,
                    "TEST": 1 / 3 - 2 / 5,
                    "UNIQUE1": [1 / 3, None],
                    "UNIQUE2": [None, 2 / 5],
                },
            },
        }
        self.assertDictEqual(expected_diff, profiler1.diff(profiler2))

        # Test with empty profile
        profiler1 = UnstructuredLabelerProfile()
        profiler1.char_sample_size = 5
        profiler1.word_sample_size = 5
        profiler1.entity_counts["postprocess_char_level"]["UNKNOWN"] = 5
        profiler1.entity_counts["true_char_level"]["UNKNOWN"] = 5
        profiler1.entity_counts["word_level"]["UNKNOWN"] = 5
        profiler1.update(pd.Series(["a"]))

        profiler2 = UnstructuredLabelerProfile()
        profile2 = profiler2.profile

        expected_diff = {
            "entity_counts": {
                "postprocess_char_level": {
                    "UNKNOWN": [5, None],
                },
                "true_char_level": {
                    "UNKNOWN": [5, None],
                },
                "word_level": {
                    "UNKNOWN": [5, None],
                },
            },
            "entity_percentages": {
                "postprocess_char_level": {
                    "UNKNOWN": [1, None],
                },
                "true_char_level": {
                    "UNKNOWN": [1, None],
                },
                "word_level": {
                    "UNKNOWN": [1, None],
                },
            },
        }
        self.assertDictEqual(expected_diff, profiler1.diff(profiler2))


if __name__ == "__main__":
    unittest.main()
