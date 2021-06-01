from collections import defaultdict
import itertools
import re
import warnings

from . import utils, BaseColumnProfiler
from .profiler_options import TextProfilerOptions


class TextProfiler(object):
    type = 'text'

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

        if options and not isinstance(options, TextProfilerOptions):
            raise ValueError("TextProfiler parameter 'options' must be of type"
                             " TextProfilerOptions.")

        self._is_case_sensitive = True
        if options:
            self._is_case_sensitive = options.is_case_sensitive

        # these stop words are from nltk
        self._stop_words = {
            'back', 'hereby', 'your', "didn't", "didnt", 'only', 'may', 
            'shouldn', "shouldnt", 'yourself', 'also', 'be', 'wouldnt', 
            'thereby', 'eight', 'hence', 'who', 'might', 'to', 'o',
            'everyone', 'hereafter', "she's", "should've", 'therein', 'm', 
            'youve', 'no', "wont", "couldn't", 'sometime', 'next', "you've", 
            "isn't", 'without', 'whereas', 'while', 'aren', 'doesn', 'three', 
            "wasn't", "needn't", 'sometimes', 'whenever', 'amongst', 'become',
            'via', 'four', 'say', 'won', 'enough', 'fifty', 'full', 'neednt', 
            'with', 'now', "hadn't", 'must', "'ll", 'by', 'n’t', 'could', 
            'nevertheless', '‘ve', 'these', 'six', 'if', 'seems', 'such', 
            'ain', "you'll", 'too', 'd', 'hers', 'ma', 'one', "'s", 'due', 
            'because', 'youre', 'at', '’s', 'wouldn', 'n‘t', 'how', '’m', 
            'regarding', 'mustn', 'shes', 'cannot', 'do', 'except', 'have', 
            'isn', 'just', 'made', 'various', 'either', 'has', 'this', 
            'everything', 'former', 'thence', 'put', 'their', 'call', 'us', 
            'along', 'on', 'towards', 'below', "shan't", 'since', 'eleven', 
            "n't", 'can', 'few', 'go', 'very', 'before', 'another', 'anyway', 
            'havent', 'else', 'fifteen', 'using', 'an', 'somehow', '’re', 'see',
            'beforehand', "aren't", 'though', 'someone', 'yourselves', 'became',
            'keep', 'herself', 'you', 'when', 'third', 'nowhere', 'seem', 'its',
            'others', 'ours', 'latterly', 'am', 'it', 'why', "hasn't", 'empty', 
            'still', 'whole', 'through', 'what', 'down', 'show', 'beyond', 
            "weren't", 'always', 'much', 'upon', 'didn', 'between', 'had', 
            'there', 'moreover', 'afterwards', 'own', 'among', 'five', 'were', 
            're', 'front', 'hadn', 'hasn', 'almost', 'werent', 'above', 
            'wherein', 'serious', 'indeed', 'every', 'beside', 'most', 
            'together', 'hes', 'my', 'should', 'twelve', 'however', 'perhaps', 
            'ever', 'noone', 'really', 'latter', 'mightnt', 'thru', 'more', 
            'twenty', 'whereafter', "mustn't", 'but', 'formerly', 'both', 
            'mightn', 'dont', 'some', 'whatever', 'would', 'seemed', 'then', 
            'of', 'theirs', 'thatllwas', "'ve", 'already', 'wasn', 'doing', 
            'nothing', "shouldn't", 'bottom', '’d', 'used', 'needn', 'weren', 
            'please', "wouldn't", 'we', 'itself', 'anywhere', 's', 'therefore', 
            'couldn', 'him', 'mostly', 'whom', "'m", 'onto', 'ten', "don't", 
            'don', 'all', '‘m', 'further', "doesn't", 'otherwise', 'did', 
            'under', 'least', 'becomes', 'many', 'give', 'shouldve', 'he', 
            'ca', 'neither', 't', 'she', 'toward', 'thereafter', 'nor', 'wasnt',
            "won't", 'during', 'where', '‘ll', 'part', 'never', 'nine', 
            'whither', 'being', 'wherever', 'about', 'well', 'hasnt', 'the', 
            'those', 'anyhow', 'somewhere', 'cant', 'two', 'having', 
            'elsewhere', "you'd", 'for', 'up', 'across', 'quite', "'re", 
            'first', 'and', 'unless', 'will', 'nobody', '‘s', 've', 'around', 
            'myself', 'not', '’ll', 'take', 'herein', 'again', 'within', 
            'other', 'couldnt', 'although', 'or', 'besides', 'into', 'they', 
            'everywhere', 'amount', 'anything', 'none', 'per', 'side', 'thus', 
            'which', 'whoever', 'isnt', 'whether', 'alone', 'are', '‘re', 
            "that'll", 'meanwhile', 'ourselves', 'less', 'her', 'from', 'was', 
            'been', 'last', 'often', 'shan', 'same', 'than', "mightn't", 
            'forty', 'off', 'once', 'does', 'i', 'hundred', "it's", 'his', 'a',
            'that', 'in', 'them', 'move', 'himself', 'whence', 'haven', 
            'themselves', 'throughout', "haven't", 'namely', 'whereby', 
            'becoming', 'sixty', 'make', 'after', 'whose', 'yet', 'so', 'is', 
            'as', 'even', 'mightve', 'top', '‘d', 'out', "you're", 'll', 
            'something', 'y', 'each', 'any', 'until', 'here', 'mine', 'rather', 
            '’ve', 'thereupon', 'over', 'me', 'our', 'whereupon', "'d", 'yours',
            'hereupon', 'done', 'against', 'get', 'behind', 'several', 'anyone',
            'seeming', "shoulve"}

        if options and options.stop_words is not None:
            self._stop_words = options.stop_words

        self.__calculations = {
            "vocab": TextProfiler._update_vocab,
            "words": TextProfiler._update_words,
        }
        BaseColumnProfiler._filter_properties_w_options(self.__calculations, options)

    def _merge_words(self, other, merged_profile):
        """
        Merges the words of two TextProfiler profiles

        :param self: first profile
        :param other: second profile
        :param merged_profile: merged profile
        :type self: TextProfiler
        :type other: TextProfiler
        :type merged_profile: TextProfiler
        :return:
        """
        if not self._is_case_sensitive:
            merged_profile.word_count = self.word_count.copy()
            additive_words = other.word_count
        else:
            merged_profile.word_count = other.word_count.copy()
            additive_words = self.word_count

        for word in additive_words:
            word_lower = word.lower()
            if word_lower not in self._stop_words:
                if self._is_case_sensitive:
                    merged_profile.word_count[word] += additive_words[word]
                else:
                    merged_profile.word_count[word_lower] += additive_words[word]
    
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
        
        merged_profile._is_case_sensitive = False
        if self._is_case_sensitive and other._is_case_sensitive:
            merged_profile._is_case_sensitive = True
        elif self._is_case_sensitive or other._is_case_sensitive:
            warnings.warn("The merged Text Profile will not be case sensitive "
                          "since there were conflicting values for case "
                          "sensitivity between the two profiles being merged.")

        
        BaseColumnProfiler._merge_calculations(merged_profile.__calculations,
                                 self.__calculations,
                                 other.__calculations)

        if "vocab" in merged_profile.__calculations:
            merged_profile.vocab = self.vocab.copy()
            merged_profile._update_vocab(other.vocab)
            
        if "words" in merged_profile.__calculations:
            self._merge_words(other, merged_profile)

        merged_profile.sample_size = self.sample_size + other.sample_size

        return merged_profile

    @property
    def profile(self):
        """
        Property for profile. Returns the profile of the column.

        :return:
        """
        word_count = sorted(self.word_count.items(),
                            key=lambda x: x[1],
                            reverse=True)
        profile = dict(
            vocab=self.vocab,
            words=list(self.word_count.keys()),
            word_count=dict(word_count),
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
        for row in data:
            for word in re.findall(r'\w+', row):
                word_lower = word.lower()
                if word_lower not in self._stop_words:
                    if not self._is_case_sensitive:
                        word = word_lower
                    self.word_count[word] += 1

    def _update_helper(self, data, profile):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known null parameters of the dataset.

        :param data: df series with nulls removed
        :type data: pandas.core.series.Series
        :param profile: text profile dictionary
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def update(self, data):
        """
        Updates the column profile.

        :param data: df series
        :type data: pandas.core.series.Series
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
