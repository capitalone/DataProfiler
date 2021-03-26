from collections import defaultdict

from .base_column_profilers import BaseColumnProfiler
from ..labelers.data_labelers import DataLabeler
from ..labelers.data_processing import CharPostprocessor


class UnstructuredDataLabelerProfile(object):

    def __init__(self, data_labeler_dirpath=None, options=None):
        """
        Initialization of Data Label profiling for unstructured datasets.

        :param data_labeler_dirpath: Directory path to the data labeler
        :type data_labeler_dirpath: String
        :param options: Options for the data labeler column
        :type options: DataLabelerOptions
        """
        # initializing a UnstructuredDataLabeler as well as the entity counts
        # statistic:
        self.options = options
        self._data_labeler = DataLabeler(
            labeler_type='unstructured',
            dirpath=data_labeler_dirpath,
            load_options=None)
        self.entity_counts = dict(
            word_level=defaultdict(int),
            true_char_level=defaultdict(int),
            postprocess_char_level=defaultdict(int))
        self.entity_percentages = dict(
            word_level=defaultdict(int),
            true_char_level=defaultdict(int),
            postprocess_char_level=defaultdict(int))
        self.char_sample_size = 0
        self.word_sample_size = 0
        self.separators = (' ', ',', ';', '"', ':', '\n', '\t', ".", "!", "'")
        self.times = defaultdict(float)

    @property
    def label_encoding(self):
        return self._data_labeler.labels

    @BaseColumnProfiler._timeit(name='data_labeler_predict')
    def _update_helper(self, df_series_clean, profile):
        """
        Method for updating the column profile properties with a cleaned
        dataset and the known profile of the dataset.
        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: profile dictionary
        :type profile: dict
        :return: None
        """
        # this will get char_level predictions as output
        predictions = self._data_labeler.predict(df_series_clean)

        # also store spacy/NER format
        postprocessor = CharPostprocessor(use_word_level_argmax=True,
                                          output_format="NER")
        format_predictions = postprocessor.process(
            df_series_clean, predictions.copy(),
            self._data_labeler.label_mapping)

        # Update counts and percent values
        self._update_word_label_counts(
            df_series_clean, format_predictions['pred'])
        self._update_true_char_label_counts(predictions['pred'])
        self._update_postprocess_char_label_counts(
            df_series_clean, format_predictions['pred'])
        self._update_percentages()

        # This will update the Profiler base properties on NUMBER OF
        # CHARACTERS/WORDS PROCESSED
        self._update_column_base_properties(profile)

    def update(self, df_series):
        if len(df_series) == 0:
            return
        profile = dict(char_sample_size=self.char_sample_size,
                       word_sample_size=self.word_sample_size)
        self._update_helper(df_series, profile)

    @property
    def profile(self):
        profile = {
            "entity_counts": self.entity_counts,
            "times": self.times
        }
        return profile

    def _update_column_base_properties(self, profile):
        """
        Updates the base properties with the base schema.
        :param profile: profile dictionary of data type
        :type profile: dict
        :return: None
        """
        self.metadata = profile

    def _get_percentages(self, level):
        """
        Creates a sorted dictionary of each entity percentages
        :param level: type of percentages returned (either word level or true
            char level or postproceess char level)
        :type level: string
        :return: Dict of entities and percentages
        :rtype: Dict
        """
        if level != 'word_level' and level != 'true_char_level' \
                and level != 'postprocess_char_level':
            return
        total = self.word_sample_size
        if level == 'true_char_level' or level == 'postprocess_char_level':
            total = self.char_sample_size

        percentages = {}
        for entity in self.entity_counts[level]:
            percentages[entity] = self.entity_counts[level][entity] / total
        return percentages

    def _update_percentages(self):
        """
        Helper to update each entity percentage
        :param: None
        :return: None
        """
        self.entity_percentages['word_level'] = self._get_percentages('word_level')
        self.entity_percentages['true_char_level'] = self._get_percentages('true_char_level')
        self.entity_percentages['postprocess_char_level'] = self._get_percentages('postprocess_char_level')

    def _update_true_char_label_counts(self, predictions):
        """
        Updates the true character label counts
        :param predictions: contains array of samples with predictions on the character level
        :type predictions: list
        :return: None
        """
        label_lookup = self._data_labeler.reverse_label_mapping
        char_label_counts = self.entity_counts["true_char_level"]

        for sample in predictions:
            for char_pred in sample:
                curr_label = label_lookup[int(char_pred)]
                char_label_counts[curr_label] += 1
            self.char_sample_size += len(sample)

    def _update_postprocess_char_label_counts(self, df_series_clean,
                                              format_predictions):
        """
        Updates the postprocess character label counts
        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param format_predictions: contains dict of samples with predictions on the character level in congruence with the word level predictions
        :type format_predictions: Dict
        :return: None
        """
        char_label_counts = self.entity_counts["postprocess_char_level"]

        for index, result in enumerate(zip(df_series_clean,
                                           format_predictions)):
            text, entities = result
            index = 0
            for entity in entities:
                char_label_counts['UNKNOWN'] += (entity[0] - index)
                #Add entity char count
                char_label_counts[entity[2]] += (entity[1] - entity[0])
                #Update index
                index = entity[1]

            #Add background from end if there is any
            char_label_counts['UNKNOWN'] += (len(text) - index)

    def _update_word_label_counts(self, df_series_clean, format_predictions):
        """
        Updates the sorted dictionary of each entity count
        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param format_predictions: Dictionary of sample text and entities
        :type format_predictions: dict
        :return: None
        """
        word_label_counts = self.entity_counts["word_level"]

        for index, result in enumerate(zip(df_series_clean,
                                           format_predictions)):
            text, entities = result
            begin_word_idx = -1
            index = 0

            #Find all the background words by searching the non-entity parts
            for entity in entities:
                #Loop through background characters
                while index<entity[0]:
                    if begin_word_idx != -1:
                        #Add background words when separator is found
                        if text[index] in self.separators:
                            word_label_counts['UNKNOWN'] += 1
                            begin_word_idx = -1
                    #Reset index when new word is found
                    elif text[index] not in self.separators:
                        begin_word_idx = index
                    index+=1
                #Update index when entity is reached
                index = entity[1]

            #Same thing as above but after the last entity
            while index < len(text):
                if begin_word_idx != -1:
                    if text[index] in self.separators:
                        word_label_counts['UNKNOWN'] += 1
                        begin_word_idx = -1
                elif text[index] not in self.separators:
                    begin_word_idx = index
                index += 1

        # Add all the entities to the total count
        word_label_counts = self.entity_counts["word_level"]
        for sample_entities in format_predictions:
            for entity in sample_entities:
                word_label_counts[entity[2]] += 1

        self.word_sample_size = 0
        for entity in word_label_counts:
            self.word_sample_size += word_label_counts[entity]