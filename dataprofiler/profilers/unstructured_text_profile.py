from collections import defaultdict
import itertools
import re

from . import utils, BaseColumnProfiler


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

        self._is_case_sensitive = True
        if options:
            self._is_case_sensitive = options.is_case_sensitive

        # these stop words are from nltk
        self._stop_words = {
            'thereafter', 'several', 'whenever', 'round', "she'd", 'shall', 
            'adj', 'front', 'abroad', 'specify', 'namely', 'lest', 'among', 
            'other', "wont", "couldn't", "i've", 'notwithstanding', 'isn', 
            'recently', 'seemed', 'therefore', 'believe', 'twenty', "c'mon", 
            'enough', 'fifth', 'otherwise', 'necessary', "weren't", '‘d', 
            'all', 'thereupon', 'becomes', 'et', 'different', 'top', 'forty', 
            'past', "daren't", 'rd', 'although', 'edu', 'again', 'it', 'three', 
            'hadn', 'my', "you'll", 'makes', 'though', 'fewer', 'four', '’d', 
            'neednt', 'isnt', 'meantime', 'hes', 'qv', 'wasn', 'very', 
            'perhaps', 'who', 'meanwhile', 'sent', 'fifteen', 'has', 'plus', 
            'ago', 'respectively', 'considering', 'provided', 'at', 'few', 
            'elsewhere', 'ten', 'new', 'seven', 'unless', "i'm", 'how', 
            'nothing', 'each', 'hence', 'are', 'some', 'selves', 'already', 
            'behind', 'thanx', 'apart', "he'll", 'in', 'sometime', 'yours', 
            'looks', 'always', 'five', 'with', 'need', 'most', "'ll", "'ve", 
            'com', 'k', 'cause', 'those', 'or', 'toward', 'afterwards', 'ex', 
            'wouldn', 'let', 'out', 'gives', 'there', 'alongside', 'own', 
            'himself', 'mightve', 'her', 'wherein', 'both', 'caption', 'than', 
            'nearly', 'better', 'thereby', 'anyhow', 'itself', 'without', 
            'bottom', "mustn't", 'look', 'shouldn', 'show', 'wasnt', 'mightn', 
            'whose', 'herself', 'hasnt', 'definitely', 'if', 'neverf', 'th', 
            'whence', 'doesn', 'inner', '‘s', 'concerning', 'n’t', 'amid', 
            'ending', '‘m', 'wouldnt', 'appear', 'within', 'greetings', 
            'associated', 'clearly', "shouldn't", 'corresponding', 'o', 'first',
            'novel', 'right', "didn't", "can't", 'non', 'latter', 'whereas', 
            'and', 'under', 'comes', 'mainly', 'these', 've', 'dare', 'towards',
            'yourselves', 'll', 'goes', 'myself', 'whither', 'furthermore', 
            'didn', 'help', 'third', '’ll', 'farther', 'hopefully', 'thanks', 
            'where', 'neverless', 'hereupon', 'anyways', 'done', 'mine', 
            'later', 'about', 'former', "should've", "he's", 'thats', 'sorry', 
            'recent', 'insofar', 'aside', 'changes', 'weren', 'ought', 'ever', 
            'ourselves', 'even', 'specifying', "shan't", 'saw', 'couldn', 
            'whereafter', 'become', 'shan', 'more', "ain't", 'could', "that'll",
            "oughtn't", 'move', 'co', 'many', "'d", 'but', 'however', 'another',
            '’s', 'awfully', 'second', 'fairly', 'thence', "doesn't", 'get', 
            'thus', 'hasn', "you'd", 'see', 'directly', "here's", 'being', 
            'gotten', "it'll", 'seeing', 'now', 'none', 'something', 'begin', 
            'youre', 'dont', 'eighty', 'whatever', 'they', 'secondly', 'ones', 
            'regardless', 'whereby', 'eight', 'she', 'maybe', 'looking', 
            'thatllwas', 'needn', 'sometimes', 'put', 'others', 'amidst', 
            'empty', 'next', 'tell', 'must', 'consider', 'backward', 'any', 
            'into', 'don', 'happens', 'ours', 'side', 'especially', 'follows', 
            'throughout', "'m", 'forth', 'were', "don't", 'much', 'amount', 
            'taken', 'mustn', 'may', 'make', 'miss', 'sup', 'becoming', 'still',
            "a's", 'certain', 'eg', 'never', 'course', 'inside', 'same', 
            'backwards', 'actually', "n't", '‘ve', "needn't", 'contains', 
            'somebody', 'm', 'probably', 'onto', 'just', 'two', 'known', 
            'werent', 'brief', 'getting', "aren't", 'various', 'rather', 'old', 
            'shes', 'while', 'along', 'else', 'sixty', 'particularly', 'hers', 
            'either', 'every', 'alone', 'found', 'placed', 'which', 
            'anyone', 'consequently', 'regards', 'because', 'ltd', 'ain', 'is', 
            'their', 'causes', "i'd", 'immediate', 'somehow', 'gets', 'i', 
            'near', 'lower', 'eleven', 'having', "isn't", 'inc.', 'your', 'so',
            'part', 'ninety', "mightn't", 're', 'available', "it's", 'half', 
            'back', 'over', 'evermore', 'regarding', 'followed', 'entirely', 
            'nonetheless', 's', 'said', 'sub', 'noone', 'soon', 'youve', 
            'seriously', 'keeps', 'you', 'everyone', 'normally', 'away', 'an', 
            'appreciate', 'this', 'across', 'hardly', 'ignored', 'he', 
            'likewise', "that've", 'described', '’re', 'particular', 'its', 
            'hundred', 'sure', 'theirs', 'anyway', 'from', "hadn't", 'says', 
            'like', 'been', 'everywhere', "you've", 'indicate', 'obviously', 
            'saying', 'needs', 'won', 'beforehand', 'whole', 'mrs', 'aren', 
            'twelve', 'often', 'thank', 'used', '’ve', "one's", 'currently', 
            'say', 'on', "'re", 'couldnt', "'s", 'give', 'sensible', 'mean', 
            "i'll", 'taking', 'whom', 'come', 'too', 'gone', "that's", 'became',
            'instead', 'que', 'please', 'since', 'themselves', 'yourself', 
            'haven', 'self', 'them', 'what', 'appropriate', 'liked', 
            'anybody', 'made', 'nobody', "c's", 'call', 'somewhere', 'cant', 
            'for', "won't", 'anything', 'be', 'that', 'the', 'does', 'was', 
            'up', 'reasonably', 'also', 'low', 'whoever', 'asking',  
            'me', 'to', 'inc', 'tends', 'until', 'howbeit', 'havent', 'lately', 
            'someone', 'downwards', 'seems', 'following', 'allows', 'beside', 
            'below', 'thru', 'outside', 'inasmuch', 'would', "let's", 'as', 
            'one', 'hi', 'his', 'nine', 'should', "she'll", 'further', 'when', 
            "hasn't", 'keep', 'according', 'okay', 'had', 'neither', 'might', 
            'nowhere', 'contain', 'y', 'of', 'formerly', 'once', 'mostly', 
            'upon', 'a', 'inward', 'hereafter', 'why', 'whether', 'co.', 
            'herein', 'specified', 'whereupon', 'ok', 'per', 'end', 'around', 
            'relatively', 'hereby', 'd', 'mightnt', 'indicated', "haven't", 
            'certainly', 'indicates', 'except', "wasn't", 'above', 'shouldve', 
            'likely', 'etc', 'presumably', 'beyond', 'no', 'allow', 
            'accordingly', 't', 'ca', 'therein', 'moreover', 'last', 'least', 
            'not', 'we', 'after', 'only', 'together', 'go', 'nevertheless', 
            'do', 'him', 'quite', 'kept', 'everything', 'minus', 'am', 
            'despite', 'six', 'before', "couldn't", 'forever', 'anywhere', 
            'nd', 'little', 'us', 'can', 'containing', 'exactly', 'hither', 
            'seem', 'due', 'someday', 'well', 'during', 'no-one', 'ask', '‘ll',
            'using', 'seen', 'against', 'almost', 'full', 'here', 'example', 
            'fifty', 'did', 'via', 'n‘t', 'everybody', 'mr', 'by', 'off', 
            'indeed', 'such', "she's", 'down', 'ahead', 'far', 'doing', 'less',
            'will', 'came', 'amongst', "it'd", 'our', 'then', 'ma', 'going', 
            'somewhat', 'serious', '’m', 'besides', 'really',
            "wouldn't", 'overall', 'given', "he'd", 'provides', 'got', 'have', 
            "you're", 'possible', 'ie', 'through', "mayn't", 'opposite', 
            'cannot', 'forward', 'oh', 'yet', 'between', 'latterly', 'seeming',
            'best', 'wherever', '‘re', 'merely', 'take', 'nor', 'able'}

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
            
        if merged_profile._is_case_sensitive:
            for word in additive_words:
                if word.lower() not in self._stop_words:
                    merged_profile.word_count[word] += additive_words[word]
        else:
            for word in additive_words:
                if word.lower() not in self._stop_words:
                    merged_profile.word_count[word.lower()] += additive_words[
                            word]
    
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
        if self._is_case_sensitive:
            for row in data:
                for word in re.findall(r'\w+', row):
                    if word.lower() not in self._stop_words:
                            self.word_count[word] += 1
        else:
            for row in data:
                for word in re.findall(r'\w+', row):
                    if word.lower() not in self._stop_words:
                        self.word_count[word.lower()] += 1


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
