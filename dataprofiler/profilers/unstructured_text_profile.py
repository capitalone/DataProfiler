"""For profiling unstructured text data."""
from __future__ import annotations

import itertools
import string
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union

from numpy import ndarray
from pandas import DataFrame, Series

from . import BaseColumnProfiler, utils
from .profiler_options import TextProfilerOptions


class TextProfiler(object):
    """Profiles text data."""

    type = "text"

    def __init__(
        self, name: Optional[str], options: TextProfilerOptions = None
    ) -> None:
        """
        Initialize TextProfiler object.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Text Profiler
        :type options: TextProfilerOptions
        """
        self.name: Optional[str] = name
        self.sample_size: int = 0
        self.times: Dict = defaultdict(float)
        self.vocab_count: Counter = Counter()
        self.word_count: Counter = Counter()
        self.metadata: Dict = dict()

        # TODO: Add line length
        # self.line_length = {'max': None, 'min': None,...} #numeric stats mixin?

        if options and not isinstance(options, TextProfilerOptions):
            raise ValueError(
                "TextProfiler parameter 'options' must be of type"
                " TextProfilerOptions."
            )

        self._is_case_sensitive = True
        if options:
            self._is_case_sensitive = options.is_case_sensitive

        # these stop words are from nltk
        self._stop_words = {
            "back",
            "hereby",
            "your",
            "didn't",
            "didnt",
            "only",
            "may",
            "shouldn",
            "shouldnt",
            "yourself",
            "also",
            "be",
            "wouldnt",
            "thereby",
            "eight",
            "hence",
            "who",
            "might",
            "to",
            "o",
            "everyone",
            "hereafter",
            "she's",
            "should've",
            "therein",
            "m",
            "youve",
            "no",
            "wont",
            "couldn't",
            "sometime",
            "next",
            "you've",
            "isn't",
            "without",
            "whereas",
            "while",
            "aren",
            "doesn",
            "three",
            "wasn't",
            "needn't",
            "sometimes",
            "whenever",
            "amongst",
            "become",
            "via",
            "four",
            "say",
            "won",
            "enough",
            "fifty",
            "full",
            "neednt",
            "with",
            "now",
            "hadn't",
            "must",
            "'ll",
            "by",
            "n’t",
            "could",
            "nevertheless",
            "‘ve",
            "these",
            "six",
            "if",
            "seems",
            "such",
            "ain",
            "you'll",
            "too",
            "d",
            "hers",
            "ma",
            "one",
            "'s",
            "due",
            "because",
            "youre",
            "at",
            "’s",
            "wouldn",
            "n‘t",
            "how",
            "’m",
            "regarding",
            "mustn",
            "shes",
            "cannot",
            "do",
            "except",
            "have",
            "isn",
            "just",
            "made",
            "various",
            "either",
            "has",
            "this",
            "everything",
            "former",
            "thence",
            "put",
            "their",
            "call",
            "us",
            "along",
            "on",
            "towards",
            "below",
            "shan't",
            "since",
            "eleven",
            "n't",
            "can",
            "few",
            "go",
            "very",
            "before",
            "another",
            "anyway",
            "havent",
            "else",
            "fifteen",
            "using",
            "an",
            "somehow",
            "’re",
            "see",
            "beforehand",
            "aren't",
            "though",
            "someone",
            "yourselves",
            "became",
            "keep",
            "herself",
            "you",
            "when",
            "third",
            "nowhere",
            "seem",
            "its",
            "others",
            "ours",
            "latterly",
            "am",
            "it",
            "why",
            "hasn't",
            "empty",
            "still",
            "whole",
            "through",
            "what",
            "down",
            "show",
            "beyond",
            "weren't",
            "always",
            "much",
            "upon",
            "didn",
            "between",
            "had",
            "there",
            "moreover",
            "afterwards",
            "own",
            "among",
            "five",
            "were",
            "re",
            "front",
            "hadn",
            "hasn",
            "almost",
            "werent",
            "above",
            "wherein",
            "serious",
            "indeed",
            "every",
            "beside",
            "most",
            "together",
            "hes",
            "my",
            "should",
            "twelve",
            "however",
            "perhaps",
            "ever",
            "noone",
            "really",
            "latter",
            "mightnt",
            "thru",
            "more",
            "twenty",
            "whereafter",
            "mustn't",
            "but",
            "formerly",
            "both",
            "mightn",
            "dont",
            "some",
            "whatever",
            "would",
            "seemed",
            "then",
            "of",
            "theirs",
            "thatllwas",
            "'ve",
            "already",
            "wasn",
            "doing",
            "nothing",
            "shouldn't",
            "bottom",
            "’d",
            "used",
            "needn",
            "weren",
            "please",
            "wouldn't",
            "we",
            "itself",
            "anywhere",
            "s",
            "therefore",
            "couldn",
            "him",
            "mostly",
            "whom",
            "'m",
            "onto",
            "ten",
            "don't",
            "don",
            "all",
            "‘m",
            "further",
            "doesn't",
            "otherwise",
            "did",
            "under",
            "least",
            "becomes",
            "many",
            "give",
            "shouldve",
            "he",
            "ca",
            "neither",
            "t",
            "she",
            "toward",
            "thereafter",
            "nor",
            "wasnt",
            "won't",
            "during",
            "where",
            "‘ll",
            "part",
            "never",
            "nine",
            "whither",
            "being",
            "wherever",
            "about",
            "well",
            "hasnt",
            "the",
            "those",
            "anyhow",
            "somewhere",
            "cant",
            "two",
            "having",
            "elsewhere",
            "you'd",
            "for",
            "up",
            "across",
            "quite",
            "'re",
            "first",
            "and",
            "unless",
            "will",
            "nobody",
            "‘s",
            "ve",
            "around",
            "myself",
            "not",
            "’ll",
            "take",
            "herein",
            "again",
            "within",
            "other",
            "couldnt",
            "although",
            "or",
            "besides",
            "into",
            "they",
            "everywhere",
            "amount",
            "anything",
            "none",
            "per",
            "side",
            "thus",
            "which",
            "whoever",
            "isnt",
            "whether",
            "alone",
            "are",
            "‘re",
            "that'll",
            "meanwhile",
            "ourselves",
            "less",
            "her",
            "from",
            "was",
            "been",
            "last",
            "often",
            "shan",
            "same",
            "than",
            "mightn't",
            "forty",
            "off",
            "once",
            "does",
            "i",
            "hundred",
            "it's",
            "his",
            "a",
            "that",
            "in",
            "them",
            "move",
            "himself",
            "whence",
            "haven",
            "themselves",
            "throughout",
            "haven't",
            "namely",
            "whereby",
            "becoming",
            "sixty",
            "make",
            "after",
            "whose",
            "yet",
            "so",
            "is",
            "as",
            "even",
            "mightve",
            "top",
            "‘d",
            "out",
            "you're",
            "ll",
            "something",
            "y",
            "each",
            "any",
            "until",
            "here",
            "mine",
            "rather",
            "’ve",
            "thereupon",
            "over",
            "me",
            "our",
            "whereupon",
            "'d",
            "yours",
            "hereupon",
            "done",
            "against",
            "get",
            "behind",
            "several",
            "anyone",
            "seeming",
            "shoulve",
        }

        if options and options.stop_words is not None:
            self._stop_words = options.stop_words
            if isinstance(self._stop_words, list):
                self._stop_words = set(self._stop_words)

        self._top_k_chars = None
        if options:
            self._top_k_chars = options.top_k_chars

        self._top_k_words = None
        if options:
            self._top_k_words = options.top_k_words

        self.__calculations = {
            "vocab": TextProfiler._update_vocab,
            "words": TextProfiler._update_words,
        }
        BaseColumnProfiler._filter_properties_w_options(self.__calculations, options)

    @staticmethod
    def _merge_vocab(vocab_count1: Counter, vocab_count2: Counter) -> Counter:
        """
        Merge the vocab counts of two TextProfiler profiles.

        :param vocab_count1: vocab count of the first profile
        :param vocab_count2: vocab count of the second profile
        :type vocab_count1: Counter()
        :type vocab_count2: Counter()
        :return: merged vocab count
        """
        return vocab_count1 + vocab_count2

    def _merge_words(self, other: TextProfiler, merged_profile: TextProfiler) -> None:
        """
        Merge the words of two TextProfiler profiles.

        :param self: first profile
        :param other: second profile
        :param merged_profile: merged profile
        :type self: TextProfiler
        :type other: TextProfiler
        :type merged_profile: TextProfiler
        :return:
        """
        merged_profile.word_count = self.word_count + other.word_count

        # remove stop words
        merged_profile._stop_words = self._stop_words.union(other._stop_words)
        if len(merged_profile._stop_words) != len(self._stop_words):
            for w in list(merged_profile.word_count.keys()):
                if w.lower() in merged_profile._stop_words:
                    merged_profile.word_count.pop(w)

        if self._is_case_sensitive != other._is_case_sensitive:
            for w, c in list(merged_profile.word_count.items()):
                if not w.islower():
                    merged_profile.word_count.pop(w)
                    merged_profile.word_count.update({w.lower(): c})

    def __add__(self, other: TextProfiler) -> TextProfiler:
        """
        Merge the properties of two TextProfiler profiles.

        :param self: first profile
        :param other: second profile
        :type self: TextProfiler
        :type other: TextProfiler
        :return: New TextProfiler merged profile
        :rtype: TextProfiler
        """
        if not isinstance(other, TextProfiler):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'TextProfiler' and '{}'".format(other.__class__.__name__)
            )
        merged_profile = TextProfiler(None)

        if self.name == other.name:
            merged_profile.name = self.name
        else:
            raise ValueError(
                "Text names unmatched: {} != {}".format(self.name, other.name)
            )

        merged_profile.times = defaultdict(
            float,
            {
                key: (self.times.get(key, 0) + other.times.get(key, 0))
                for key in (set(self.times) | set(other.times))
            },
        )

        merged_profile._is_case_sensitive = False
        if self._is_case_sensitive and other._is_case_sensitive:
            merged_profile._is_case_sensitive = True
        elif self._is_case_sensitive or other._is_case_sensitive:
            warnings.warn(
                "The merged Text Profile will not be case sensitive "
                "since there were conflicting values for case "
                "sensitivity between the two profiles being merged."
            )

        merged_profile._top_k_chars = None
        if self._top_k_chars and other._top_k_chars:
            merged_profile._top_k_chars = max(self._top_k_chars, other._top_k_chars)

        merged_profile._top_k_words = None
        if self._top_k_words and other._top_k_words:
            merged_profile._top_k_words = max(self._top_k_words, other._top_k_words)

        BaseColumnProfiler._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )

        if "vocab" in merged_profile.__calculations:
            merged_profile.vocab_count = self._merge_vocab(
                self.vocab_count, other.vocab_count
            )
        if "words" in merged_profile.__calculations:
            self._merge_words(other, merged_profile)

        merged_profile.sample_size = self.sample_size + other.sample_size

        return merged_profile

    def diff(self, other_profile: TextProfiler, options: Dict = None) -> Dict:
        """
        Find the differences for two unstructured text profiles.

        :param other_profile: profile to find the difference with
        :type other_profile: TextProfiler
        :param options: options for diff output
        :type options: dict
        :return: the difference between profiles
        :rtype: dict
        """
        cls = self.__class__
        if not isinstance(other_profile, cls):
            raise TypeError(
                "Unsupported operand type(s) for diff: '{}' "
                "and '{}'".format(cls.__name__, other_profile.__class__.__name__)
            )

        self_words = list(self.word_count.keys())
        self_word_count = dict(self.word_count.most_common(self._top_k_words))

        other_words = list(other_profile.word_count.keys())
        other_word_count = dict(other_profile.word_count.most_common(self._top_k_words))

        if self._is_case_sensitive and not other_profile._is_case_sensitive:
            self_words = [each_string.lower() for each_string in self_words]
            self_word_count = {k.lower(): v for k, v in self_word_count.items()}
        if not self._is_case_sensitive and other_profile._is_case_sensitive:
            other_words = [each_string.lower() for each_string in other_words]
            other_word_count = {k.lower(): v for k, v in other_word_count.items()}

        diff: Dict = {}
        diff["vocab"] = utils.find_diff_of_lists_and_sets(
            list(self.vocab_count.keys()), list(other_profile.vocab_count.keys())
        )

        diff["vocab_count"] = utils.find_diff_of_dicts_with_diff_keys(
            dict(self.vocab_count.most_common(self._top_k_chars)),
            dict(other_profile.vocab_count.most_common(self._top_k_chars)),
        )

        diff["words"] = utils.find_diff_of_lists_and_sets(self_words, other_words)

        diff["word_count"] = utils.find_diff_of_dicts_with_diff_keys(
            self_word_count, other_word_count
        )

        return diff

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """Report profile attribute of class; potentially pop val from self.profile."""
        calcs_dict_keys = self._TextProfiler__calculations.keys()  # type: ignore
        profile = self.profile

        if remove_disabled_flag:
            profile_keys = list(profile.keys())
            for profile_key in profile_keys:
                if profile_key == "vocab":
                    if "vocab" in calcs_dict_keys:
                        continue
                if profile_key == "words":
                    if "words" in calcs_dict_keys:
                        continue
                profile.pop(profile_key)

        return profile

    @property
    def profile(self) -> Dict:
        """
        Return the profile of the column.

        :return: profile of the column
        :rtype: dict
        """
        profile = dict(
            vocab=list(self.vocab_count.keys()),
            vocab_count=dict(self.vocab_count.most_common(self._top_k_chars)),
            words=list(self.word_count.keys()),
            word_count=dict(self.word_count.most_common(self._top_k_words)),
            times=self.times,
        )
        return profile

    @BaseColumnProfiler._timeit(name="vocab")
    def _update_vocab(
        self,
        data: Union[List, ndarray, DataFrame],
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """
        Find the vocabulary counts used in the text samples.

        :param data: list or array of data from which to extract vocab
        :type data: Union[list, numpy.array, pandas.DataFrame]
        :param prev_dependent_properties: Contains all the previous properties
            that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
            subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        data_flat = list(itertools.chain(*data))
        self.vocab_count += Counter(data_flat)

    @BaseColumnProfiler._timeit(name="words")
    def _update_words(
        self,
        data: Union[List, ndarray, DataFrame],
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """
        Find unique words and word count used in the text samples.

        :param data: list or array of data from which to extract vocab
        :type data: Union[list, numpy.array, pandas.DataFrame]
        :param prev_dependent_properties: Contains all the previous properties
            that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
            subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        if not self._is_case_sensitive:
            words = (
                [w.strip(string.punctuation) for w in row.lower().split()]
                for row in data
            )
        else:
            words = ([w.strip(string.punctuation) for w in row.split()] for row in data)
        word_count = Counter(itertools.chain.from_iterable(words))

        for w, c in word_count.items():
            if w and w.lower() not in self._stop_words:
                self.word_count.update({w: c})

    def _update_helper(self, data: Series, profile: Dict) -> None:
        """
        Update col profile properties with clean dataset and its known null parameters.

        :param data: df series with nulls removed
        :type data: pandas.core.series.Series
        :param profile: text profile dictionary
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def update(self, data: Series) -> TextProfiler:
        """
        Update the column profile.

        :param data: df series
        :type data: pandas.core.series.Series
        :return: updated TextProfiler
        :rtype: TextProfiler
        """
        len_data = len(data)
        if len_data == 0:
            return self

        profile = dict(sample_size=len_data)

        BaseColumnProfiler._perform_property_calcs(
            self,  # type: ignore
            self.__calculations,
            df_series=data,
            prev_dependent_properties={},
            subset_properties=profile,
        )

        self._update_helper(data, profile)

        return self
