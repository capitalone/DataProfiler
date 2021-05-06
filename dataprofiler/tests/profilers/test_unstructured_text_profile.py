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

    def test_text_profile_with_wrong_options(self):
        with self.assertRaisesRegex(ValueError,
                "TextProfiler parameter 'options' must be of type"
                " TextProfilerOptions."):
            TextProfiler("Name", options="wrong_data_type")

    def test_options_default(self):
        options = TextProfilerOptions()

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'sentence': 1, 'Test': 1, 'test': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'sentence': 1, 'Test': 1, 'test': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

    def test_options_case_sensitive(self):
        # change is_case_sensitive, other options remain the same as default values
        options = TextProfilerOptions()
        options.is_case_sensitive = False

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'sentence': 1, 'test': 2}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'sentence': 1, 'test': 2}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

    def test_options_stop_words(self):
        # change stop_words, other options remain the same as default values

        # with a list of stopwords
        options = TextProfilerOptions()
        options.stop_words = ['hello', 'sentence', 'is', 'a']

        ## input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'This': 1, 'Test': 1, 'test': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        ## input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'This': 1, 'Test': 1, 'test': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        # with an empty list
        options = TextProfilerOptions()
        options.stop_words = []

        ## input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'This': 1, 'is': 1, 'test': 1, 'a': 1,
                               'Test': 1, 'sentence': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        ## input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'This': 1, 'is': 1, 'test': 1, 'a': 1,
                               'Test': 1, 'sentence': 1}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

    def test_options_words_update(self):
        # change words.is_enabled, other options remain the same as default values

        options = TextProfilerOptions()
        options.words.is_enabled = False

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {}
        expected_vocab = [' ', ',', '.', '!', 'T', 'h', 'i', 's', 'a',
                          'e', 't', 'n', 'c']
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

    def test_options_vocab_update(self):
        # change vocab.is_enabled, other options remain the same as default values

        options = TextProfilerOptions()
        options.vocab.is_enabled = False

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'sentence': 1, 'Test': 1, 'test': 1}
        expected_vocab = []
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {'sentence': 1, 'Test': 1, 'test': 1}
        expected_vocab = []
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertCountEqual(expected_vocab, text_profile.vocab)
