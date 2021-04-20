from collections import defaultdict
from . import utils, BaseColumnProfiler
import itertools

class TextProfiler(object):
    type = 'unstructured_text'

    def __init__(self, name, options=None):
        """
        Initialization of Text Profiler.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Text Profiler
        :type options: UnstructuredTextOptions
        """
        self.name = name
        self.sample_size = 0
        self.times = defaultdict(float)
        self.vocab = set()
        self.word_count = defaultdict(int)
        self.metadata = dict()

        # TODO: Add line length
        #self.line_length = {'max': None, 'min': None,...} #numeric stats mixin?

        # these stop words are from nltk
        self._stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
            'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
            'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
            'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
            "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
            'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
            'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
            'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }

        self.__calculations = {
            "vocab": TextProfiler._update_vocab,
            "words": TextProfiler._update_words,
        }
        BaseColumnProfiler._filter_properties_w_options(self.__calculations, options)

    def _add_helper(self, other1, other2):
        """
        Merges the properties of two BaseColumnProfile objects

        :param other1: first profile
        :param other2: second profile
        :type other1: BaseColumnProfiler
        :type other2: BaseColumnProfiler
        """
        if other1.name == other2.name:
            self.name = other1.name
        else:
            raise ValueError("Text names unmatched: {} != {}"
                             .format(other1.name, other2.name))

        self.times = defaultdict(
            float, {key: (other1.times.get(key, 0)
                          + other2.times.get(key, 0)
                          + self.times.get(key, 0))
                    for key in (set(other1.times) | set(other2.times)
                                | set(self.times))}
        )

        self.sample_size = other1.sample_size + other2.sample_size

    def __add__(self, other):
        """
        Merges the properties of two TextProfiler profiles

        :param self: first profile
        :param other: second profile
        :type self: TextProfiler
        :type other: TextProfiler
        :return: New TextProfiler merged profile
        """
        if not isinstance(other, TextProfiler):
            raise TypeError("Unsupported operand type(s) for +: "
                            "'TextProfiler' and '{}'".format(
                other.__class__.__name__))
        merged_profile = TextProfiler(None)
        
        if self.name == other.name:
            merged_profile.name = self.name
        else:
            raise ValueError("Text names unmatched: {} != {}"
                             .format(self.name, other.name))

        merged_profile.times = defaultdict(
            float, {key: (self.times.get(key, 0)
                          + other.times.get(key, 0))
                    for key in (set(self.times) | set(other.times))}
        )
        
        BaseColumnProfiler._merge_calculations(merged_profile.__calculations,
                                 self.__calculations,
                                 other.__calculations)

        if "vocab" in merged_profile.__calculations:
            merged_profile.vocab = self.vocab.copy()
            merged_profile._update_vocab(other.vocab)
            
        if "words" in merged_profile.__calculations:
            merged_profile.word_count = self.word_count.copy()
            for word in other.word_count:
                if word in merged_profile.word_count:
                    merged_profile.word_count[word] += other.word_count[word]
                elif word not in self._stop_words:
                    merged_profile.word_count[word] = other.word_count[word]
                    
        merged_profile.sample_size = self.sample_size + other.sample_size

        return merged_profile

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.

        :return:
        """
        profile = dict(
            vocab=self.vocab,
            words=list(self.word_count.keys()),
            word_count=dict(self.word_count),
            times=self.times,
        )
        return profile

    @BaseColumnProfiler._timeit(name='vocab')
    def _update_vocab(self, data, prev_dependent_properties=None,
                      subset_properties=None):
        """
        Finds the unique vocabulary used in the text samples.

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
        self.vocab = utils._combine_unique_sets(self.vocab, data_flat)

    @BaseColumnProfiler._timeit(name='words')
    def _update_words(self, data, prev_dependent_properties=None,
                      subset_properties=None):
        """
        Finds the unique words and word count used in the text samples.

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
        for word in list(" ".join(data).split(" ")):
            if word in self.word_count:
                self.word_count[word] += 1
            elif word not in self._stop_words:
                self.word_count[word] = 1

    def _update_helper(self, data, profile):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known null parameters of the dataset.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: text profile dictionary
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def update(self, data):
        """
        Updates the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: None
        """
        len_data = len(data)
        if len_data == 0:
            return self

        profile = dict(sample_size=len_data)

        BaseColumnProfiler._perform_property_calcs(
            self, self.__calculations, df_series=data,
            prev_dependent_properties={}, subset_properties=profile)
        
        self._update_helper(data, profile)

        return self
