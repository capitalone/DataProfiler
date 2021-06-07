import unittest
from unittest import mock
from collections import defaultdict

import pandas as pd

from dataprofiler.profilers.unstructured_labeler_profile import \
    UnstructuredLabelerProfile


class TestUnstructuredLabelerProfile(unittest.TestCase):

    def test_char_level_counts(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(["abc123", "Bob", "!@##$%"])

        # running update
        default.update(sample)
        # now getting entity_counts to check for existence
        self.assertIsNotNone(default.profile["entity_counts"]
                             ["true_char_level"])

        self.assertIsNotNone(default.profile["entity_counts"]
                             ["postprocess_char_level"])

        # assert it's not empty for now
        self.assertIsNotNone(default.profile)

        # then assert that correctly counted number of char samples
        self.assertEqual(default.char_sample_size, 15)

    def test_advanced_sample(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(
            ["Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
             "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912."
             "\n", "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n"])

        # running update
        default.update(sample)

        # now getting entity_counts to check for proper structure
        self.assertIsNotNone(default.profile["entity_counts"]
                             ["true_char_level"])

        # assert it's not empty for now
        self.assertIsNotNone(default.profile)

    def test_word_level_NER_label_counts(self):
        # setting up objects/profile
        default = UnstructuredLabelerProfile()

        sample = pd.Series(
            ["Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
             "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000049939232194912."
             "\n", "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n"])

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
            ["Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
             "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912."
             "\n", "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n"])

        # running update
        default.update(sample)

        self.assertIsNotNone(default.entity_percentages['word_level'])
        self.assertIsNotNone(default.entity_percentages['true_char_level'])
        self.assertIsNotNone(default.entity_percentages['postprocess_char_level'])
        current_word_sample_size = default.word_sample_size
        current_char_sample_size = default.char_sample_size
        self.assertIsNotNone(default.word_sample_size)
        self.assertIsNotNone(default.char_sample_size)
        self.assertIsNotNone(default.entity_counts['word_level'])
        self.assertIsNotNone(default.entity_counts['true_char_level'])
        self.assertIsNotNone(default.entity_counts['postprocess_char_level'])
        self.assertIsNone(default._get_percentages('WRONG_INPUT'))

        default.update(sample)
        self.assertNotEqual(current_word_sample_size, default.word_sample_size)
        self.assertNotEqual(current_char_sample_size, default.char_sample_size)

    @mock.patch('dataprofiler.profilers.'
                'unstructured_labeler_profile.DataLabeler')
    @mock.patch('dataprofiler.profilers.'
                'unstructured_labeler_profile.'
                'CharPostprocessor')
    def test_profile(self, processor_class_mock, model_class_mock):
        # setup mocks
        model_mock = mock.Mock()
        model_mock.reverse_label_mapping = {1: 'UNKNOWN'}
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
                'postprocess_char_level': defaultdict(int, {'UNKNOWN': 1}),
                'true_char_level': defaultdict(int, {'UNKNOWN': 1}),
                'word_level': defaultdict(int)
            },
            times=defaultdict(float, {'data_labeler_predict': 1.0})
        )

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            default.update(sample)
        profile = default.profile

        # key and value populated correctly
        self.assertDictEqual(expected_profile, profile)

    @mock.patch('dataprofiler.profilers.'
                'unstructured_labeler_profile.DataLabeler')
    def test_unstructured_labeler_profile_add(self, mock):
        # Test empty merge
        profile1 = UnstructuredLabelerProfile()
        profile2 = UnstructuredLabelerProfile()
        merged_profile = profile1 + profile2

        self.assertDictEqual(merged_profile.entity_counts["word_level"], {})
        self.assertDictEqual(merged_profile.entity_counts["true_char_level"], {})
        self.assertDictEqual(merged_profile.entity_counts
                             ["postprocess_char_level"], {})
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
        
        
        self.assertDictEqual(merged_profile.entity_counts["word_level"], 
                             expected_word_level)
        self.assertDictEqual(merged_profile.entity_counts["true_char_level"], 
                             expected_true_char)
        self.assertDictEqual(merged_profile.entity_counts["postprocess_char_level"],
                             expected_post_char)
        
        self.assertEqual(merged_profile.word_sample_size, 11)
        self.assertEqual(merged_profile.char_sample_size, 10)

        self.assertEqual(merged_profile.times["data_labeler_predict"],
                         profile1.times["data_labeler_predict"] +
                         profile2.times["data_labeler_predict"])
        

if __name__ == '__main__':
    unittest.main()
