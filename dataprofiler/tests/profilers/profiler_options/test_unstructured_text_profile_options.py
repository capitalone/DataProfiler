import unittest

import pandas as pd

from dataprofiler.profilers.unstructured_text_profile import TextProfiler
from dataprofiler.profilers.profiler_options import TextProfilerOptions


class TestUnstructuredTextProfileOptions(unittest.TestCase):

    def test_options_validation(self):
        """
        Option check for the invalid values
        :return:
        """
        # case sensitive
        options_case_sensitive = [2, 'string']
        for option_case_sensitive in options_case_sensitive:
            options = TextProfilerOptions()
            options.is_case_sensitive = option_case_sensitive
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.is_case_sensitive must be a Boolean."):
                TextProfiler("Name", options=options)

        # stop words
        options_stop_words = [2, 'a', [1, 2]]
        for option_stop_words in options_stop_words:
            options = TextProfilerOptions()
            options.stop_words = option_stop_words
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.stop_words must be None "
                    "or list of strings."):
                TextProfiler("Name", options=options)

        # words update
        options_words = [2, True]
        for option_words in options_words:
            options = TextProfilerOptions()
            options.words = option_words
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.words must be a BooleanOption "
                    "object."):
                TextProfiler("Name", options=options)

        # vocab update
        options_vocab = [2, True]
        for option_vocab in options_vocab:
            options = TextProfilerOptions()
            options.vocab = option_vocab
            with self.assertRaisesRegex(
                    ValueError,
                    "TextProfilerOptions.vocab must be a BooleanOption "
                    "object."):
                TextProfiler("Name", options=options)

    def test_options_default(self):
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

    def test_options_case_sensitive(self):
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

    def test_options_stop_words(self):
        # with a list of stopwords
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

        # with an empty list
        options = TextProfilerOptions()
        options.stop_words = []

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test"])
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {'This': 1, 'is': 1, 'test': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])

    def test_options_words_update(self):
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

    def test_options_vocab_update(self):
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
