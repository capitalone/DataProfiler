import unittest

import pandas as pd

from dataprofiler.profilers.profiler_options import TextProfilerOptions
from dataprofiler.profilers.unstructured_text_profile import TextProfiler


class TestUnstructuredTextProfile(unittest.TestCase):
    def test_text_profile_update_and_name(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(
            ["Hello my name is: Grant.!!!", "Bob and \"Grant\", 'are' friends"]
        )
        text_profile.update(sample)
        self.assertEqual("Name", text_profile.name)

    def test_vocab(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(
            ["Hello my name is: Grant.!!!", "Bob and \"Grant\", 'are' friends"]
        )
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert vocab is correct
        expected_vocab = [
            " ",
            "!",
            '"',
            "'",
            ",",
            ".",
            ":",
            "B",
            "G",
            "H",
            "a",
            "b",
            "d",
            "e",
            "f",
            "i",
            "l",
            "m",
            "n",
            "o",
            "r",
            "s",
            "t",
            "y",
        ]
        self.assertListEqual(sorted(expected_vocab), sorted(profile["vocab"]))

        # Update the data again
        sample = pd.Series(["Grant knows how to code", "Grant will code with Bob"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert vocab is correct
        expected_vocab = [
            " ",
            "!",
            '"',
            "'",
            ",",
            ".",
            ":",
            "B",
            "G",
            "H",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "h",
            "i",
            "k",
            "l",
            "m",
            "n",
            "o",
            "r",
            "s",
            "t",
            "w",
            "y",
        ]
        self.assertListEqual(sorted(expected_vocab), sorted(profile["vocab"]))

    def test_words_and_word_count(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(
            ["Hello my name is: Grant.!!!", "Bob and \"Grant\", 'are' friends"]
        )
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert words is correct and stop words are not present
        expected_words = ["Hello", "name", "Grant", "Bob", "friends"]
        self.assertListEqual(expected_words, profile["words"])
        self.assertNotIn("is", profile["words"])

        # Assert word counts are correct
        expected_word_count = {
            "Hello": 1,
            "name": 1,
            "Grant": 2,
            "Bob": 1,
            "friends": 1,
        }
        self.assertDictEqual(expected_word_count, profile["word_count"])

        # Update the data again
        sample = pd.Series(["Grant knows how to code", "Grant will code with Bob"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert words is correct and stop words are not present
        expected_words = ["Hello", "name", "Grant", "Bob", "friends", "knows", "code"]
        self.assertListEqual(expected_words, profile["words"])
        self.assertNotIn("with", profile["words"])

        # Assert word counts are correct
        expected_word_count = {
            "Hello": 1,
            "name": 1,
            "Grant": 4,
            "Bob": 2,
            "friends": 1,
            "knows": 1,
            "code": 2,
        }
        self.assertDictEqual(expected_word_count, profile["word_count"])

    def test_sample_size(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(
            ["Hello my name is: Grant.!!!", "Bob and \"Grant\", 'are' friends"]
        )
        text_profile.update(sample)

        # Assert sample size is accurate
        self.assertEqual(2, text_profile.sample_size)

        # Update the data again
        sample = pd.Series(["Grant knows how to code", "Grant will code with Bob"])
        text_profile.update(sample)

        # Assert sample size is accurate
        self.assertEqual(4, text_profile.sample_size)

    def test_timing(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(
            ["Hello my name is: Grant.!!!", "Bob and \"Grant\", 'are' friends"]
        )
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
        expected_vocab = [
            " ",
            "!",
            '"',
            "'",
            ",",
            ".",
            ":",
            "B",
            "G",
            "H",
            "a",
            "b",
            "d",
            "e",
            "f",
            "i",
            "l",
            "m",
            "n",
            "o",
            "r",
            "s",
            "t",
            "y",
        ]
        self.assertListEqual(sorted(expected_vocab), sorted(profile["vocab"]))

        # Assert words is correct and stop words are not present
        expected_words = ["Bob", "Grant", "friends", "Hello", "name"]
        self.assertCountEqual(expected_words, profile["words"])
        self.assertNotIn("is", profile["words"])

        # Assert word counts are correct
        expected_word_count = {
            "Hello": 1,
            "name": 1,
            "Grant": 2,
            "Bob": 1,
            "friends": 1,
        }
        self.assertDictEqual(expected_word_count, profile["word_count"])

        # Assert timing is occurring
        self.assertIn("vocab", profile["times"])
        self.assertIn("words", profile["times"])

    def test_diff_profiles(self):
        text_profile1 = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!"])
        text_profile1.update(sample)

        text_profile2 = TextProfiler("Name")
        sample = pd.Series(["Bob and \"grant\", 'are' friends Grant Grant"])
        text_profile2.update(sample)

        expected_diff = {
            "vocab": [
                ["H", "l", "m", "y", ":", ".", "!"],
                ["e", "o", " ", "n", "a", "i", "s", "G", "r", "t"],
                ["B", "b", "d", '"', "g", ",", "'", "f"],
            ],
            "vocab_count": [
                {"!": 3, "l": 2, "m": 2, "H": 1, "y": 1, ":": 1, ".": 1},
                {
                    " ": -2,
                    "e": "unchanged",
                    "n": -3,
                    "a": -3,
                    "o": "unchanged",
                    "i": "unchanged",
                    "s": "unchanged",
                    "G": -1,
                    "r": -4,
                    "t": -2,
                },
                {"d": 2, '"': 2, "'": 2, "B": 1, "b": 1, "g": 1, ",": 1, "f": 1},
            ],
            "words": [["Hello", "name"], ["Grant"], ["Bob", "grant", "friends"]],
            "word_count": [
                {"Hello": 1, "name": 1},
                {"Grant": -1},
                {"Bob": 1, "grant": 1, "friends": 1},
            ],
        }
        self.assertDictEqual(expected_diff, text_profile1.diff(text_profile2))

        # Test when one profiler is not case sensitive
        text_profile1 = TextProfiler("Name")
        sample = pd.Series(["Hello my name is: Grant.!!!"])
        text_profile1.update(sample)

        options = TextProfilerOptions()
        options.is_case_sensitive = False
        text_profile2 = TextProfiler("Name", options=options)
        sample = pd.Series(["Bob and \"grant\", 'are' friends Grant Grant"])
        text_profile2.update(sample)

        expected_diff = {
            "vocab": [
                ["H", "l", "m", "y", ":", ".", "!"],
                ["e", "o", " ", "n", "a", "i", "s", "G", "r", "t"],
                ["B", "b", "d", '"', "g", ",", "'", "f"],
            ],
            "vocab_count": [
                {"!": 3, "l": 2, "m": 2, "H": 1, "y": 1, ":": 1, ".": 1},
                {
                    " ": -2,
                    "e": "unchanged",
                    "n": -3,
                    "a": -3,
                    "o": "unchanged",
                    "i": "unchanged",
                    "s": "unchanged",
                    "G": -1,
                    "r": -4,
                    "t": -2,
                },
                {"d": 2, '"': 2, "'": 2, "B": 1, "b": 1, "g": 1, ",": 1, "f": 1},
            ],
            "words": [["hello", "name"], ["grant"], ["bob", "friends"]],
            "word_count": [
                {"hello": 1, "name": 1},
                {"grant": -2},
                {"bob": 1, "friends": 1},
            ],
        }
        self.assertDictEqual(expected_diff, text_profile1.diff(text_profile2))

    def test_case_sensitivity(self):
        text_profile1 = TextProfiler("Name")
        text_profile1._is_case_sensitive = False
        sample = pd.Series(["Hello my name is: Grant.!!!"])
        text_profile1.update(sample)
        profile = text_profile1.profile
        expected_word_count = {"grant": 1, "hello": 1, "name": 1}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        text_profile2 = TextProfiler("Name")
        sample = pd.Series(["Bob and \"Grant\", 'are' friends"])
        text_profile2.update(sample)
        profile = text_profile2.profile
        expected_word_count = {"Grant": 1, "Bob": 1, "friends": 1}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        with self.assertWarnsRegex(
            UserWarning,
            "The merged Text Profile will not be case sensitive since there"
            " were conflicting values for case sensitivity between the two "
            "profiles being merged.",
        ):
            text_profile3 = text_profile1 + text_profile2
            profile = text_profile3.profile
            # Assert word counts are correct
            expected_word_count = {
                "hello": 1,
                "name": 1,
                "grant": 2,
                "bob": 1,
                "friends": 1,
            }
            self.assertDictEqual(expected_word_count, profile["word_count"])

        # change the merge order
        with self.assertWarnsRegex(
            UserWarning,
            "The merged Text Profile will not be case sensitive since there"
            " were conflicting values for case sensitivity between the two "
            "profiles being merged.",
        ):
            text_profile3 = text_profile2 + text_profile1
            profile = text_profile3.profile
            # Assert word counts are correct
            expected_word_count = {
                "hello": 1,
                "name": 1,
                "grant": 2,
                "bob": 1,
                "friends": 1,
            }
            self.assertDictEqual(expected_word_count, profile["word_count"])

    def test_merge_most_common_chars_count(self):
        ### default values of most common chars for both profiles
        text_profile1 = TextProfiler("Name")
        sample1 = pd.Series(["this is test,", " this is a test sentence"])
        text_profile1.update(sample1)

        text_profile2 = TextProfiler("Name")
        sample2 = pd.Series(["this is", "this"])
        text_profile2.update(sample2)

        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        # as merged profile's vocab_count length is None, it is set to
        # the length of the merged vocab_count, which is 10
        expected_vocab_count = {
            "s": 10,
            "t": 9,
            " ": 8,
            "i": 7,
            "e": 5,
            "h": 4,
            "n": 2,
            ",": 1,
            "a": 1,
            "c": 1,
        }
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

        ### one profile has default values of most common chars
        ### the other profile has it set
        text_profile1._top_k_chars = 3
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        # as merged profile's vocab_count length is None, it is set to
        # the length of the merged vocab_count, which is 10
        expected_vocab_count = {
            "s": 10,
            "t": 9,
            " ": 8,
            "i": 7,
            "e": 5,
            "h": 4,
            "n": 2,
            ",": 1,
            "a": 1,
            "c": 1,
        }
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

        ### equal number of most common chars
        text_profile1._top_k_chars = 3
        text_profile2._top_k_chars = 3
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        expected_vocab_count = {"s": 10, "t": 9, " ": 8}
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

        ### different number of most common chars
        text_profile1._top_k_chars = 2
        text_profile2._top_k_chars = 3
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        expected_vocab_count = {"s": 10, "t": 9, " ": 8}
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

    def test_merge_most_common_words_count(self):
        ### default values of most common words for both profiles
        text_profile1 = TextProfiler("Name")
        text_profile1._stop_words = set()  # set stop_words to empty for easy inspection
        sample1 = pd.Series(["this is test,", " this is a test sentence"])
        text_profile1.update(sample1)

        text_profile2 = TextProfiler("Name")
        text_profile2._stop_words = set()  # set stop_words to empty for easy inspection
        sample2 = pd.Series(["this is", "this"])
        text_profile2.update(sample2)

        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        # as merged profile's word_count length is None, it is set to
        # the length of the merged word_count, which is 5
        expected_word_count = {"this": 4, "is": 3, "test": 2, "a": 1, "sentence": 1}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        ### one profile has default values of most common words
        ### the other profile has it set
        text_profile1._top_k_words = 3
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        # as merged profile's word_count length is None, it is set to
        # the length of the merged word_count, which is 5
        expected_word_count = {"this": 4, "is": 3, "test": 2, "a": 1, "sentence": 1}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        ### equal number of most common words
        text_profile1._top_k_words = 3
        text_profile2._top_k_words = 3
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        expected_word_count = {"this": 4, "is": 3, "test": 2}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        ### different number of most common words
        text_profile1._top_k_words = 2
        text_profile2._top_k_words = 3
        text_profile3 = text_profile1 + text_profile2
        profile = text_profile3.profile

        expected_word_count = {"this": 4, "is": 3, "test": 2}
        self.assertDictEqual(expected_word_count, profile["word_count"])

    def test_text_profile_with_wrong_options(self):
        with self.assertRaisesRegex(
            ValueError,
            "TextProfiler parameter 'options' must be of type" " TextProfilerOptions.",
        ):
            TextProfiler("Name", options="wrong_data_type")

    def test_options_default(self):
        options = TextProfilerOptions()

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"sentence": 1, "Test": 1, "test": 1}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"sentence": 1, "Test": 1, "test": 1}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

    def test_report(self):
        """Test report method in TextProfiler class under four (4) scenarios.
        First, test under scenario of disabling vocab and word. Second, test with no options and
        `remove_disabled_flag`=True. Third, test no options and default
        `remove_disabled_flag`. Lastly, test under scenario of disabling vocab but not word.
        """
        options = (
            TextProfilerOptions()
        )  # With TextProfilerOptions as False and remove_disabled_flag == True
        options.vocab.is_enabled = False
        options.words.is_enabled = False

        profiler = TextProfiler("Name", options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        profiler.update(sample)

        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())
        self.assertNotIn("vocab", report_keys)
        self.assertNotIn("words", report_keys)

        profiler = TextProfiler(
            "Name"
        )  # w/o TextProfilerOptions and remove_disabled_flag == True
        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())
        self.assertIn("vocab", report_keys)
        self.assertIn("words", report_keys)

        profiler = TextProfiler(
            "Name"
        )  # w/o TextProfilerOptions and remove_disabled_flag default
        report = profiler.report()
        report_keys = list(report.keys())
        self.assertIn("vocab", report_keys)
        self.assertIn("words", report_keys)

        options = (
            TextProfilerOptions()
        )  # With TextProfilerOptions True/False and remove_disabled_flag == True
        options.vocab.is_enabled = True
        options.words.is_enabled = False

        profiler = TextProfiler("Name", options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        profiler.update(sample)

        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())

        self.assertIn("vocab", report_keys)
        self.assertNotIn("words", report_keys)

    def test_options_case_sensitive(self):
        # change is_case_sensitive, other options remain the same as default values
        options = TextProfilerOptions()
        options.is_case_sensitive = False

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"sentence": 1, "test": 2}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"sentence": 1, "test": 2}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

    def test_options_stop_words(self):
        # change stop_words, other options remain the same as default values

        # with a list of stopwords
        options = TextProfilerOptions()
        options.stop_words = ["hello", "sentence", "is", "a"]

        ## input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"This": 1, "Test": 1, "test": 1}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        ## input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"This": 1, "Test": 1, "test": 1}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        # with an empty list
        options = TextProfilerOptions()
        options.stop_words = []

        ## input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {
            "This": 1,
            "is": 1,
            "test": 1,
            "a": 1,
            "Test": 1,
            "sentence": 1,
        }
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        ## input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {
            "This": 1,
            "is": 1,
            "test": 1,
            "a": 1,
            "Test": 1,
            "sentence": 1,
        }
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

    def test_options_vocab_update(self):
        # change vocab.is_enabled, other options remain the same as default values
        options = TextProfilerOptions()
        options.vocab.is_enabled = False

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"sentence": 1, "Test": 1, "test": 1}
        expected_vocab = dict()
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {"sentence": 1, "Test": 1, "test": 1}
        expected_vocab = dict()
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

    def test_options_words_update(self):
        # change words.is_enabled, other options remain the same as default values
        options = TextProfilerOptions()
        options.words.is_enabled = False

        # input with one sample
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test, a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

        # input with two samples
        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(["This is test,", " a Test sentence.!!!"])
        text_profile.update(sample)

        expected_word_count = {}
        expected_vocab = {
            "s": 5,
            " ": 5,
            "e": 5,
            "t": 4,
            "!": 3,
            "T": 2,
            "i": 2,
            "n": 2,
            "h": 1,
            ",": 1,
            "a": 1,
            "c": 1,
            ".": 1,
        }
        self.assertDictEqual(expected_word_count, text_profile.word_count)
        self.assertDictEqual(expected_vocab, text_profile.vocab_count)

    def test_options_most_common_chars_count(self):
        # None value for number of common chars
        options = TextProfilerOptions()
        options.top_k_chars = None

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(
            ["this is test,", " this is a test sentence", "this is", "this"]
        )
        text_profile.update(sample)
        profile = text_profile.profile

        expected_vocab_count = {
            "s": 10,
            "t": 9,
            " ": 8,
            "i": 7,
            "e": 5,
            "h": 4,
            "n": 2,
            ",": 1,
            "a": 1,
            "c": 1,
        }
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

        # set number of common chars to 3
        options.top_k_chars = 3

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(
            ["this is test,", " this is a test sentence", "this is", "this"]
        )
        text_profile.update(sample)
        profile = text_profile.profile

        expected_vocab_count = {"s": 10, "t": 9, " ": 8}
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

        # change number of common chars
        options.top_k_chars = 2
        text_profile = TextProfiler("Name", options=options)
        text_profile.update(sample)
        profile = text_profile.profile

        expected_vocab_count = {"s": 10, "t": 9}
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

        # change number of common chars greater than length of vocab_counts list
        options.top_k_chars = 300
        text_profile = TextProfiler("Name", options=options)
        text_profile.update(sample)
        profile = text_profile.profile

        expected_vocab_count = {
            "s": 10,
            "t": 9,
            " ": 8,
            "i": 7,
            "e": 5,
            "h": 4,
            "n": 2,
            ",": 1,
            "a": 1,
            "c": 1,
        }
        self.assertDictEqual(expected_vocab_count, profile["vocab_count"])

    def test_options_most_common_words_count(self):
        # None value for number of common words
        options = TextProfilerOptions()
        options.top_k_words = None
        options.stop_words = []  # set stop_words to empty list for easy inspection

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(
            ["this is test,", " this is a test sentence", "this is", "this"]
        )
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {"this": 4, "is": 3, "test": 2, "a": 1, "sentence": 1}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        # set number of common words to 3
        options.top_k_words = 3
        options.stop_words = []  # set stop_words to empty list for easy inspection

        text_profile = TextProfiler("Name", options=options)
        sample = pd.Series(
            ["this is test,", " this is a test sentence", "this is", "this"]
        )
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {"this": 4, "is": 3, "test": 2}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        # change number of common words
        options.top_k_words = 2
        text_profile = TextProfiler("Name", options=options)
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {"this": 4, "is": 3}
        self.assertDictEqual(expected_word_count, profile["word_count"])

        # change number of common words greater than length of word_counts list
        options.top_k_words = 10
        text_profile = TextProfiler("Name", options=options)
        text_profile.update(sample)
        profile = text_profile.profile

        expected_word_count = {"this": 4, "is": 3, "test": 2, "a": 1, "sentence": 1}
        self.assertDictEqual(expected_word_count, profile["word_count"])
