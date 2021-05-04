import unittest

import pandas as pd

from dataprofiler.profilers.unstructured_text_profile import TextProfiler
from dataprofiler.profilers.profiler_options import TextProfilerOptions


class TestUnstructuredTextProfile(unittest.TestCase):
    
    def test_text_profile_update_and_name(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!",
                            "Bob and \"Grant\", 'are' friends"])
        text_profile.update(sample)
        self.assertEqual("Name", text_profile.name)
      
    def test_vocab(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!",
                            "Bob and \"Grant\", 'are' friends"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert vocab is correct
        expected_vocab = [' ', '!', '"', "'", ',', '.', ':', 'B', 'G', 'H', 
                          'a', 'b', 'd', 'e', 'f', 'i', 'l', 'm', 'n', 'o', 
                          'r', 's', 't', 'y']
        self.assertListEqual(sorted(expected_vocab), sorted(profile['vocab']))

        # Update the data again
        sample = pd.Series(["Grant knows how to code",
                            "Grant will code with Bob"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert vocab is correct
        expected_vocab = [' ', '!', '"', "'", ',', '.', ':', 'B', 'G', 'H',
                          'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 
                          'm', 'n', 'o', 'r', 's', 't', 'w', 'y']
        self.assertListEqual(sorted(expected_vocab), sorted(profile['vocab']))

    def test_words_and_word_count(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!",
                            "Bob and \"Grant\", 'are' friends"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert words is correct and stop words are not present
        expected_words = ['Hello', 'name', 'Grant', 'Bob', 'friends']
        self.assertListEqual(expected_words, profile['words'])
        self.assertNotIn("is", profile['words'])

        # Assert word counts are correct
        expected_word_count = {'Hello': 1, 'name': 1, 'Grant': 2, 'Bob': 1,
                               'friends': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])

        # Update the data again
        sample = pd.Series(["Grant knows how to code",
                            "Grant will code with Bob"])
        text_profile.update(sample)
        profile = text_profile.profile
        
        # Assert words is correct and stop words are not present
        expected_words = ['Hello', 'name', 'Grant', 'Bob', 'friends', 'knows',
                          'code']
        self.assertListEqual(expected_words, profile['words'])
        self.assertNotIn("with", profile['words'])

        # Assert word counts are correct
        expected_word_count = {'Hello': 1, 'name': 1, 'Grant': 4, 'Bob': 2,
                               'friends': 1, 'knows': 1, 'code': 2}
        self.assertDictEqual(expected_word_count, profile['word_count'])
        
    def test_sample_size(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!",
                            "Bob and \"Grant\", 'are' friends"])
        text_profile.update(sample)

        # Assert sample size is accurate
        self.assertEqual(2, text_profile.sample_size)

        # Update the data again
        sample = pd.Series(["Grant knows how to code",
                            "Grant will code with Bob"])
        text_profile.update(sample)

        # Assert sample size is accurate
        self.assertEqual(4, text_profile.sample_size)
    
    def test_timing(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!",
                            "Bob and \"Grant\", 'are' friends"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert timing is occurring
        self.assertIn("vocab", profile["times"])
        self.assertIn("words", profile["times"])
        
    def test_merge_profiles(self):
        text_profile1 = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!"])
        text_profile1.update(sample)
        
        text_profile2 = TextProfiler("Name")
        sample = pd.Series(["Bob and \"Grant\", 'are' friends"])
        text_profile2.update(sample)
        
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile
        
        self.assertEqual("Name", text_profile3.name)
        
        # Assert sample size is accurate
        self.assertEqual(2, text_profile3.sample_size)
        
        # Assert vocab is correct
        expected_vocab = [' ', '!', '"', "'", ',', '.', ':', 'B', 'G', 'H',
                          'a', 'b', 'd', 'e', 'f', 'i', 'l', 'm', 'n', 'o',
                          'r', 's', 't', 'y']
        self.assertListEqual(sorted(expected_vocab), sorted(profile['vocab']))
        
        # Assert words is correct and stop words are not present
        expected_words = ['Bob', 'Grant', 'friends', 'Hello', 'name']
        self.assertListEqual(expected_words, profile['words'])
        self.assertNotIn("is", profile['words'])

        # Assert word counts are correct
        expected_word_count = {'Hello': 1, 'name': 1, 'Grant': 2, 'Bob': 1,
                               'friends': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])
        
        # Assert timing is occurring
        self.assertIn("vocab", profile["times"])
        self.assertIn("words", profile["times"])
        
    def test_case_sensitivity(self):
        text_profile1 = TextProfiler("Name")
        text_profile1._is_case_sensitive = False
        sample = pd.Series(["Hello my name is: Grant.!!!"])
        text_profile1.update(sample)
        profile = text_profile1.profile
        expected_word_count = {'grant': 1, 'hello': 1, 'name': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])

        text_profile2 = TextProfiler("Name")
        sample = pd.Series(["Bob and \"Grant\", 'are' friends"])
        text_profile2.update(sample)
        profile = text_profile2.profile
        expected_word_count = {'Grant': 1, 'Bob': 1, 'friends': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])
        
        with self.assertWarnsRegex(UserWarning,
                "The merged Text Profile will not be case sensitive since there"
                " were conflicting values for case sensitivity between the two "
                "profiles being merged."):
            text_profile3 = text_profile1 + text_profile2
            profile = text_profile3.profile
            # Assert word counts are correct
            expected_word_count = {'hello': 1, 'name': 1, 'grant': 2, 'bob': 1,
                                   'friends': 1}
            self.assertDictEqual(expected_word_count, profile['word_count'])

    def test_options_validation(self):
        # option check for the invalid values
        options_case_sensitive = [2, 'string']
        for option_case_sensitive in options_case_sensitive:
            options = TextProfilerOptions()
            options.is_case_sensitive = option_case_sensitive
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.is_case_sensitive must be a Boolean."):
                TextProfiler("Name", options=options)

        options_stop_words = [2, 'a', [1, 2]]
        for option_stop_words in options_stop_words:
            options = TextProfilerOptions()
            options.stop_words = option_stop_words
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.stop_words must be None "
                              "or list of strings."):
                TextProfiler("Name", options=options)

        options_words = [2, True]
        for option_words in options_words:
            options = TextProfilerOptions()
            options.words = option_words
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.words must be a BooleanOption "
                          "object."):
                TextProfiler("Name", options=options)

        options_vocab = [2, True]
        for option_vocab in options_vocab:
            options = TextProfilerOptions()
            options.vocab = option_vocab
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.vocab must be a BooleanOption "
                    "object."):
                TextProfiler("Name", options=options)

    def test_options_different_values(self):
        # default options
        options = TextProfilerOptions()

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {'sentence': 1, 'Test': 1, 'test': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, profile['word_count'])
        self.assertCountEqual(expected_vocab, profile['vocab'])

        # is_case_sensitive options
        options = TextProfilerOptions()
        options.is_case_sensitive = False

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {'sentence': 1, 'test': 2}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, profile['word_count'])
        self.assertCountEqual(expected_vocab, profile['vocab'])

        # stop_words options
        options = TextProfilerOptions()
        options.stop_words = ['hello', 'sentence', 'is', 'a']

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {'This': 1, 'Test': 1, 'test': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, profile['word_count'])
        self.assertCountEqual(expected_vocab, profile['vocab'])

        options = TextProfilerOptions()
        options.stop_words = [] # empty list of stopwords

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {'This': 1, 'is': 1, 'test': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])

        # words enabled options
        options = TextProfilerOptions()
        options.words.is_enabled = False

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, profile['word_count'])
        self.assertCountEqual(expected_vocab, profile['vocab'])

        # vocab enabled options
        options = TextProfilerOptions()
        options.vocab.is_enabled = False

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {'sentence': 1, 'Test': 1, 'test': 1}
        expected_vocab = []
        self.assertDictEqual(expected_word_count, profile['word_count'])
        self.assertCountEqual(expected_vocab, profile['vocab'])
        