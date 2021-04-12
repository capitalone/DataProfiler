import os
import pkg_resources
import abc
import types
import copy
import json
import inspect
from collections import Counter
import random
import math
import warnings
import copy

import numpy as np

default_labeler_dir = pkg_resources.resource_filename(
    'resources', 'labelers'
)


class AutoSubRegistrationMeta(abc.ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        new_class = super(AutoSubRegistrationMeta, cls).__new__(
            cls, clsname, bases, attrs
        )
        new_class._register_subclass()
        return new_class


class BaseDataProcessor(metaclass=abc.ABCMeta):
    """Abstract Data processing class."""

    processor_type = None
    __subclasses = {}

    def __init__(self, **parameters):

        self._validate_parameters(parameters)
        self._parameters = parameters

    @classmethod
    def _register_subclass(cls):
        """
        Registers a subclass for the class factory.
        """
        if not inspect.isabstract(cls):
            cls._BaseDataProcessor__subclasses[cls.__name__.lower()] = cls

    @classmethod
    def get_class(cls, class_name):
        return cls._BaseDataProcessor__subclasses.get(class_name.lower(), None)

    def __eq__(self, other):
        """
        Checks if two processors are equal with one another.

        :param self: a processor
        :param other: a processor
        :type self: BaseDataProcessor
        :type other: BaseDataProcessor
        :return: Whether or not self and other are equal
        :rtype: bool
        """
        if type(self) != type(other) \
                or self._parameters != other._parameters:
            return False
        return True

    @abc.abstractmethod
    def _validate_parameters(self, parameters):
        """Validates class input parameters for processing"""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """
        raise NotImplementedError()

    def get_parameters(self, param_list=None):
        """
        Returns a dict of parameters from the model given a list.

        :param param_list: list of parameters to retrieve from the model.
        :type param_list: list
        :return: dict of parameters
        """
        if param_list is None:
            return copy.deepcopy(self._parameters)

        param_dict = {}
        for param in param_list:
            if param in self._parameters:
                param_dict[param] = self._parameters.get(param)
            else:
                raise ValueError('`{}` does not exist as a parameter in {}.'.
                                 format(param, self.__class__.__name__))
        return param_dict

    def set_params(self, **kwargs):
        """Given kwargs, set the parameters if they exist."""

        # first check if any parameters are invalid
        self._validate_parameters(kwargs)

        for param in kwargs:
            self._parameters[param] = kwargs[param]

    @abc.abstractmethod
    def process(self, *args):
        """Data processing function."""
        raise NotImplementedError()

    @classmethod
    def load_from_disk(cls, dirpath):
        """Loads a data processor from a given path on disk"""
        with open(os.path.join(dirpath,
                               cls.processor_type + '_parameters.json')) as fp:
            parameters = json.load(fp)

        return cls(**parameters)

    @classmethod
    def load_from_library(cls, name):
        """Loads a data processor from within the library"""
        return cls.load_from_disk(os.path.join(default_labeler_dir, name))

    def _save_processor(self, dirpath):
        """Generic method for saving a data processor"""
        with open(os.path.join(dirpath,
                               self.processor_type + '_parameters.json'),
                  'w') as fp:
            json.dump(self.get_parameters(), fp)

    def save_to_disk(self, dirpath):
        """Saves a data processor to a path on disk."""
        self._save_processor(dirpath)


class BaseDataPreprocessor(BaseDataProcessor):
    """Abstract Data preprocessing class."""

    processor_type = 'preprocessor'
    __metaclass__ = abc.ABCMeta

    def __init__(self, **parameters):
        super(BaseDataPreprocessor, self).__init__(**parameters)

    @abc.abstractmethod
    def process(self, data, labels, label_mapping, batch_size):
        """Data preprocessing function."""
        raise NotImplementedError()


class BaseDataPostprocessor(BaseDataProcessor):
    """Abstract Data postprocessing class."""

    processor_type = 'postprocessor'
    __metaclass__ = abc.ABCMeta

    def __init__(self, **parameters):
        super(BaseDataPostprocessor, self).__init__(**parameters)

    @abc.abstractmethod
    def process(self, data, results, label_mapping):
        """Data postprocessing function."""
        raise NotImplementedError()


class DirectPassPreprocessor(BaseDataPreprocessor,
                             metaclass=AutoSubRegistrationMeta):

    def __init__(self):
        """
        Initialize the DirectPassPreprocessor class
        """
        super(DirectPassPreprocessor, self).__init__()

    def _validate_parameters(self, parameters):
        """
        Validates the parameters set in the processor, raises an error if any
        issues exist.

        :param parameters: parameter dict containing the following parameters:
            N/A
        :type parameters: dict
        :return: None
        :rtype: None
        """
        if parameters:
            raise ValueError('`DirectPassPreprocessor` has no parameters.')

    @classmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """
        help_str = (
            cls.__name__ + "\n\n"
            + "Parameters:\n"
            "    There are no parameters for this processor."
            "\nProcess Input Format:\n"
            "    There is no required format for data or labels parameters for "
            "    this processor. Please refer to the Model for input format.")
        print(help_str)

    def process(self, data, labels=None, label_mapping=None, batch_size=None):
        """Data preprocessing function."""
        if labels is not None:
            return data, labels
        return data


class CharPreprocessor(BaseDataPreprocessor, metaclass=AutoSubRegistrationMeta):

    def __init__(self, max_length=3400, default_label='UNKNOWN',
                 pad_label='PAD', flatten_split=0, flatten_separator=" ",
                 is_separate_at_max_len=False):
        """
        Initialize the CharPreprocessor class

        :param max_length: Maximum char length in a sample.
        :type max_length: int
        :param default_label: Key for label_mapping that is the default label
        :type default_label: string (could be int, char, etc.)
        :param pad_label: Key for label_mapping that is the pad label
        :type pad_label: string (could be int, char, etc.)
        :param flatten_split: approximate output of split between flattened and
            non-flattened characters, value between [0, 1]. When the current
            flattened split becomes more than the `flatten_split` value, any
            leftover sample or subsequent samples will be non-flattened until
            the current flattened split is below the `flatten_split` value
        :type flatten_split: float
        :param flatten_separator: separator used to put between flattened
            samples.
        :type flatten_separator: str
        :param is_separate_at_max_len: if true, separates at max_length,
            otherwise at nearest separator
        :type is_separate_at_max_len: bool
        """
        super().__init__(
            max_length=max_length,
            pad_label=pad_label,
            default_label=default_label,
            flatten_split=flatten_split,
            flatten_separator=flatten_separator,
            is_separate_at_max_len=is_separate_at_max_len,
        )

    def _validate_parameters(self, parameters):
        """
        Validates the parameters set in the processor, raises an error if any
        issues exist.

        :param parameters: parameter dict containing the following parameters:
            max_length: Maximum char length in a sample.
            default_label: Key for label_mapping that is the default label
            pad_label: Key for label_mapping that is the pad label
            flatten_split: Approximate output of split between flattened and
                non-flattened characters, value between [0, 1]. When the current
                flattened split becomes more than the `flatten_split` value,
                any leftover sample or subsequent samples will be non-flattened
                until the current flattened split is below the `flatten_split`
                value
            flatten_separator: Separator used to put between flattened samples.
            is_separate_at_max_len: If true, separates at max_length, otherwise
                at nearest separator
        :type parameters: dict
        :return: None
        :rtype: None
        """
        errors = []
        allowed_parameters = self.__class__.__init__.__code__.co_varnames[
                            1:self.__class__.__init__.__code__.co_argcount]
        for param in parameters:
            value = parameters[param]
            if param == 'max_length' \
                    and (not isinstance(value, int) or value < 1):
                errors.append('`max_length` must be an int > 0')
            elif param in ['default_label', 'pad_label'] and \
                    not isinstance(value, str):
                errors.append("`{}` must be a string.".format(param))
            elif param == 'flatten_split' \
                    and (not isinstance(value, (int, float))
                         or value < 0 or value > 1 or math.isnan(value)):
                errors.append('`flatten_split` must be a float or int '
                              '>= 0 and <= 1')
            elif param == 'flatten_separator' and not isinstance(value, str):
                errors.append('`flatten_separator` must be a str')
            elif param == 'is_separate_at_max_len' \
                    and not isinstance(value, bool):
                errors.append('`{}` must be a bool'.format(param))
            elif param not in allowed_parameters:
                errors.append("{} is not an accepted parameter.".format(param))

        if errors:
            raise ValueError('\n'.join(errors))

    @classmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """
        param_docs = inspect.getdoc(cls._validate_parameters)
        param_start_ind = param_docs.find('parameters:\n') + 12
        param_end_ind = param_docs.find(':type parameters:')

        help_str = (
            cls.__name__ + "\n\n"
            + "Parameters:\n"
            + param_docs[param_start_ind:param_end_ind]
            + "\nProcess Input Format:\n"
            "    data = List of strings ['1st string', 'second string', ...]\n"
            "    labels = [[(<INT>, <INT>, \"<LABEL>\"), "
            "...(num_samples in string)], ...(num strings in data)]")
        print(help_str)

    @staticmethod
    def _find_nearest_sentence_break_before_ind(sentence, start_ind,
                                                min_ind=0,
                                                separators=(' ', '\n', ',',
                                                            '\t', '\r', '\x00',
                                                            '\x01', ';')):
        """
        Find nearest separator before the start_ind and return the index.

        :param sentence: sentence to be split.
        :type sentence: str
        :param start_ind: starting index to begin search.
        :type start_ind: int
        :param min_ind: minimum possible index for break
        :type min_ind: int
        :param separators: list of separators that the sentence can be split
            upon.
        :type separators: tuple(str)
        :return: index at which the sentence can be split, -1 if no index found.
        """
        ind_distance = 0
        separator_dict = dict(zip(separators, [None] * len(separators)))
        while (start_ind - ind_distance) > min_ind:
            if (start_ind - ind_distance) >= min_ind and \
                    sentence[start_ind - ind_distance] in separator_dict:
                return start_ind - ind_distance
            ind_distance += 1
        return min_ind

    def _process_batch_helper(self, data, max_length, default_label, pad_label,
                              labels=None, label_mapping=None, batch_size=32):
        """
        Flatten batches of data

        :param data: List of strings to create embeddings for
        :type data: list(str)
        :param max_length: Maximum char length in a sample.
        :type max_length: int
        :param default_label: Key for label_mapping that is the default label
        :type default_label: str
        :param pad_label: Key for label_mapping that is the pad label
        :type pad_label: str
        :param labels: labels for each input character
        :type labels: list
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: Union[dict, None]
        :param batch_size: Number of samples in the batch of data
        :type batch_size: int
        :return batch_data: A dict containing  samples of size batch_size
        :rtype batch_data: dicts
        """
        # get processor parameters
        flatten_separator = self._parameters['flatten_separator']
        flatten_split = self._parameters['flatten_split']
        is_separate_at_max_len = self._parameters['is_separate_at_max_len']

        flatten_separator_len = len(flatten_separator)
        if flatten_separator_len >= max_length:
            raise ValueError(
                'The `flatten_separator` length cannot be more than ' +
                'or equal to the `max_length`.')
        # Create list of vectors for samples
        flattened_entities = []
        flattened_sample = []
        batch_data = {'samples': []} if labels is None \
            else {'samples': [], 'labels': []}

        flattened_sample_len = 0
        batch_count = 0

        flatten_count = 0
        non_flatten_count = 0
        force_sample = False

        # create None generator if labels is not set
        if labels is None:
            def gen_none():
                """Generates infinite None(s). Must be closed manually."""
                while True:
                    yield None
            labels = gen_none()

        # loop through each sample
        for sample_buffer, label_set in zip(data, labels):
            # ensure data is str
            sample_buffer = str(sample_buffer)
            # buffer is empty, add sample to the buffer.
            sample_len = len(sample_buffer)

            if label_set is not None:
                # Create an entity buffer for sample, assign the default entity
                label_buffer = np.full(
                    sample_len, label_mapping[default_label])

                # Map the entity to the corresponding character
                for start, end, label in label_set:
                    label_index = label_mapping[label]
                    label_buffer[start:end] = label_index
                label_buffer = label_buffer.tolist()

            # loop until the buffer is empty and placed as requested
            buffer_ind = 0
            buffer_len = len(sample_buffer)
            while buffer_ind < buffer_len:
                sample_len = buffer_len - buffer_ind

                # choose whether to flatten the current buffer based on the
                # requested `flatten_ratio` and `curr_flatten_ratio`
                curr_flatten_split = (flatten_count /
                                      max(1, flatten_count + non_flatten_count))
                if curr_flatten_split > flatten_split or not flatten_split:
                    # Don't flatten if the flatten percentage is more than asked

                    separate_ind = max_length + buffer_ind
                    if not is_separate_at_max_len and \
                            separate_ind < buffer_len:
                        # if the sample_buffer will be split, look, for a
                        # separator to split the sample_buffer so as to not
                        # break words.
                        separate_ind = CharPreprocessor.\
                            _find_nearest_sentence_break_before_ind(
                                sample_buffer, separate_ind, min_ind=buffer_ind)

                        # if no separator found, just split on the maximum
                        # possible length.
                        if separate_ind == buffer_ind:
                            # intentionally using flattened_sample_len bc may
                            # have carry over from flattening, if it is 0, it
                            # reverts to max_length which is if splitting a
                            # non-flattened sample at worse case.
                            separate_ind = max_length - flattened_sample_len + \
                                           buffer_ind

                    # padding cases:
                    # 1. if sample is over length: label_buffer[:separate_ind]
                    # 2. if sample is < max_length: label_buffer[:separate_ind]
                    #    but truly becomes label_buffer[:sample_len]

                    # pad the data until fits maximum length
                    pad_len = max(max_length - separate_ind + buffer_ind,
                                  max_length - sample_len)

                    # Only add the buffer up until maximum length
                    batch_data['samples'].append(
                        sample_buffer[buffer_ind:separate_ind])
                    if label_set is not None:
                        batch_data['labels'].append(
                            label_buffer[buffer_ind:separate_ind] +
                            [label_mapping[pad_label]] * pad_len)

                    non_flatten_count += 1
                    batch_count += 1
                    buffer_ind = separate_ind

                else:  # flattening

                    # flattening has 3 states:
                    #   1. if the sample buffer fits into the flattened_sample
                    #      w/o going over the maximum length, add it, but dont
                    #      update the batch.
                    #   2. if the flattened sample will overflow even if the
                    #      separator is added
                    #   3. the sample buffer fits with completely or partially
                    #      when adding the separator

                    # if this is the start of a flattened sample, don't add a
                    # separator
                    if flattened_sample_len > 0:
                        if label_set is not None:
                            flattened_entities.extend(
                                [label_mapping[default_label]] *
                                flatten_separator_len)
                        flattened_sample_len += flatten_separator_len

                    separate_ind = max_length - flattened_sample_len \
                        + buffer_ind
                    if not is_separate_at_max_len and \
                            separate_ind < buffer_len:
                        # if the sample_buffer will be split, look for a
                        # separator to split the sample_buffer so as to not
                        # break words.
                        separate_ind = CharPreprocessor.\
                            _find_nearest_sentence_break_before_ind(
                                sample_buffer, separate_ind, min_ind=buffer_ind)

                        # if no separator found, just split on the maximum
                        # possible length.
                        if separate_ind == buffer_ind \
                                and flattened_sample_len > 0:
                            separate_ind = buffer_ind
                            flattened_sample_len -= flatten_separator_len
                            if label_set is not None:
                                flattened_entities = \
                                    flattened_entities[:-flatten_separator_len]
                            force_sample = True
                        elif separate_ind == buffer_ind:
                            separate_ind = max_length - flattened_sample_len + \
                                           buffer_ind

                    # maximize the amount of the buffer added to the flattened
                    # sample
                    if separate_ind > 0:
                        if buffer_ind != separate_ind:
                            flattened_sample.append(
                                sample_buffer[buffer_ind:separate_ind])

                            # update the length of the flattened sample
                            flattened_sample_len += len(flattened_sample[-1])

                        if label_set is not None:
                            flattened_entities.extend(
                                label_buffer[buffer_ind:separate_ind])

                        buffer_ind = separate_ind

                    # add to flattened sample to the batch if it reaches the
                    # maximum length
                    if flattened_sample_len >= \
                            max_length - flatten_separator_len \
                            or force_sample:
                        force_sample = False
                        batch_sample = flatten_separator.join(
                            flattened_sample)

                        # pad the data until fits maximum length
                        pad_len = max_length - flattened_sample_len
                        if label_set is not None:
                            flattened_entities.extend(
                                [label_mapping[pad_label]] * pad_len)

                        # Add the flattened sample to the batch
                        batch_data['samples'].append(batch_sample)
                        if label_set is not None:
                            batch_data['labels'].append(flattened_entities)

                        # update counts
                        batch_count += 1
                        flatten_count += 1

                        # reset flattened for next samples
                        flattened_sample_len = 0
                        flattened_sample = []
                        flattened_entities = []

                # return the data when the batch size is met
                if batch_count == batch_size:
                    yield batch_data

                    # reset batch data for next iteration
                    batch_data = {'samples': []} if label_set is None \
                        else {'samples': [], 'labels': []}
                    batch_count = 0

        # if anything left in flattened sample, add to batch
        if len(flattened_sample):
            pad_len = max_length - flattened_sample_len
            batch_sample = flatten_separator.join(flattened_sample)
            batch_data['samples'].append(batch_sample)
            if label_set is not None:
                flattened_entities.extend(
                    [label_mapping[pad_label]] * pad_len)
                batch_data['labels'].append(flattened_entities)
            flatten_count += 1

        if isinstance(labels, types.GeneratorType):
            labels.close()
        # only provide data if it exists
        if batch_data['samples']:
            yield batch_data

    def process(self, data, labels=None, label_mapping=None, batch_size=32):
        """
        Flatten batches of data

        :param data: List of strings to create embeddings for
        :type data: numpy.ndarray
        :param labels: labels for each input character
        :type labels: numpy.ndarray
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: Union[None, dict]
        :param batch_size: Number of samples in the batch of data
        :type batch_size: int
        :return batch_data: A dict containing  samples of size batch_size
        :rtype batch_data: dicts
        """
        num_dim = sum([dim > 1 for dim in data.shape])
        if num_dim > 1:
            raise ValueError("Multidimensional data given to "
                             "CharPreprocessor. Consider using a different "
                             "preprocessor or flattening data (and labels)")
        # Flattened data into single dimensional np array, if it was truly 1D
        data = data.reshape(-1)

        if labels is not None:
            if not label_mapping:
                raise ValueError('If `labels` are specified, `label_mapping` '
                                 'must also be specified.')
            if len(data) != len(labels):
                raise ValueError(f"Data and labels given to CharPreprocessor "
                                 f"are different lengths, "
                                 f"{len(data)} != {len(labels)}")

        # Import tensorflow
        import tensorflow as tf

        # get parameters
        max_length = self._parameters['max_length']
        default_label = self._parameters['default_label']
        pad_label = self._parameters['pad_label']

        batch_process_generator = self._process_batch_helper(
            data, max_length, default_label,  pad_label, labels, label_mapping,
            batch_size)

        for batch_data in batch_process_generator:
            # Convert to necessary training data format.
            X_train = np.array(
                [[sentence] for sentence in batch_data['samples']])
            if labels is not None:
                num_classes = max(label_mapping.values()) + 1
                
                Y_train = tf.keras.utils.to_categorical(
                    batch_data['labels'], num_classes)
                yield X_train, Y_train
            else:  # must be an else or will yield this as well
                yield X_train


class CharPostprocessor(BaseDataPostprocessor,
                        metaclass=AutoSubRegistrationMeta):

    def __init__(self,
                 default_label='UNKNOWN', pad_label='PAD',
                 flatten_separator=" ",
                 use_word_level_argmax=False,
                 output_format='character_argmax',
                 separators=(' ', ',', ';', "'", '"', ':', '\n', '\t', "."),
                 word_level_min_percent=0.75):
        """
        Initialize the CharPostprocessor class

        :param default_label: Key for label_mapping that is the default label
        :type default_label: string (could be int, char, etc.)
        :param pad_label: Key for label_mapping that is the pad label
        :type pad_label: string (could be int, char, etc.)
        :param flatten_separator: separator used to put between flattened
            samples.
        :type flatten_separator: str
        :param use_word_level_argmax: whether to require the argmax value of
            each character in a word to determine the word's entity
        :type use_word_level_argmax: bool
        :param output_format: (character_argmax vs NER) where character_argmax
            is a list of encodings for each character in the input text and NER
            is in the dict format which specifies start,end,label for each
            entity in a sentence
        :type output_format: str
        :param separators: list of characters to use for separating words within
            the character predictions
        :type separators: tuple(str)
        :param word_level_min_percent: threshold on generating dominant
            word_level labeling
        :type word_level_min_percent: float
        """
        super().__init__(
            default_label=default_label,
            pad_label=pad_label,
            flatten_separator=flatten_separator,
            use_word_level_argmax=use_word_level_argmax,
            output_format=output_format,
            separators=separators,
            word_level_min_percent=word_level_min_percent,
        )

    def _validate_parameters(self, parameters):
        """
        Validates the parameters set in the processor, raises an error if any
        issues exist.

        :param parameters: parameter dict containing the following parameters:
            default_label: Key for label_mapping that is the default label
            pad_label: Key for label_mapping that is the pad label
            flatten_separator: Separator used to put between flattened samples
                use_word_level_argmax: whether to require the argmax value of
                each character in a word to determine the word's entity
            output_format: (character_argmax vs NER) where character_argmax is a
                list of encodings for each character in the input text and NER
                is in the dict format which specifies start,end,label for each
                entity in a sentence
            separators: List of characters to use for separating words within
                the character predictions
            word_level_min_percent: Threshold on generating dominant word_level
                labeling
        :type parameters: dict
        :return: None
        """
        errors = []
        allowed_parameters = self.__class__.__init__.__code__.co_varnames[
                            1:self.__class__.__init__.__code__.co_argcount]
        for param in parameters:
            value = parameters[param]
            if param in ['default_label', 'pad_label', 'flatten_separator'] \
                    and not isinstance(value, str):
                errors.append("`{}` must be a string.".format(param))
            if param == 'use_word_level_argmax' and not isinstance(value, bool):
                errors.append('`use_word_level_argmax` must be a bool')
            elif param == 'output_format' \
                    and (not isinstance(value, str)
                         or value.lower() not in ['character_argmax', 'ner']):
                errors.append('`output_format` must be a str of value '
                              '`character_argmax` or `ner`')
            elif param == 'separators' \
                    and (not isinstance(value, (list, tuple))
                         or sum(map(lambda x: not isinstance(x, str), value))):
                errors.append('`separators` must be a list of str')
            elif param == 'word_level_min_percent' \
                    and (not isinstance(value, (int, float))
                         or value < 0 or value > 1 or math.isnan(value)):
                errors.append('`word_level_min_percent` must be a float or int '
                              '>= 0 and <= 1')
            elif param not in allowed_parameters:
                errors.append("{} is not an accepted parameter.".format(param))

        if errors:
            raise ValueError('\n'.join(errors))

    @classmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """
        param_docs = inspect.getdoc(cls._validate_parameters)
        param_start_ind = param_docs.find('parameters:\n') + 12
        param_end_ind = param_docs.find(':type parameters:')

        help_str = (
            cls.__name__ + "\n\n"
            + "Parameters:\n"
            + param_docs[param_start_ind:param_end_ind]
            + "\nProcess Output Format:\n"
            "    --character_argmax--\n"
            "    Each character receives a label.\n"
            "    Input String  - 'My String'\n"
            "    Output labels - [[1, 1, 1, 2, 2, 2, 2, 2, 2], "
            "..(num sentences)]\n"
            "\n"
            "    --NER--\n"
            "    List of entities for each sentence in tuple format.\n"
            "    Entity format - (start, end, label)\n"
            "    Input String  - 'My String'\n"
            "    Output labels - [[(0, 2, 'BG'), (3, 9, 'OTHER')], "
            "..(num sentences)]"
        )
        print(help_str)

    def _word_level_argmax(self, data, predictions, label_mapping,
                           default_label):
        """
        Convert char level predictions to word level predictions

        :param data: input text
        :type data: np.ndarray
        :param predictions: character level predictions
        :type predictions: list
        :param label_mapping: labels and corresponding integers
        :type label_mapping: dict
        :param default_label: Key for label_mapping that is the default label
        :type default_label: str
        :return: list containing the word level labels based off of character
                 level predictions
        """
        # get processor parameters
        separators = self._parameters['separators']
        word_level_min_percent = self._parameters['word_level_min_percent']

        # Get word level labelling
        word_level_predictions = []
        background_label = label_mapping[default_label]

        # Iterate over both lists, should be same length
        for sample, char_pred in zip(data, predictions):
            
            # Copy entities_in_sample so can return later
            # changed input param to "predictions" for ease
            # FORMER DEEPCOPY, SHALLOW AS ONLY INTERNAL
            entities_in_sample = list(char_pred)

            # Convert to dict for quick look-up
            separator_dict = {}
            for separator in separators:
                separator_dict[separator] = True

            # Iterate over sample
            start_idx = 0
            label_count = {label_mapping[default_label]: 0}
            for idx in range(len(sample)):

                # Split on separator
                if sample[idx] in separator_dict:

                    # Find sum of labels over entity
                    total_label_count = sum(label_count.values())

                    # If no max is found, set to background
                    dominate_label = label_mapping[default_label]
                    dominate_label_count = 1
                    for label in label_count:
                        label_ratio = float(label_count[label]) / max(
                            float(total_label_count), 1)
                        if label_ratio >= word_level_min_percent and \
                                label_count[label] > dominate_label_count:
                            dominate_label_count = label_count[label]
                            dominate_label = label

                    # Set the dominate label to the entity
                    for i in range(start_idx, idx):
                        entities_in_sample[i] = dominate_label

                    # Set to background if not relabeled
                    if dominate_label == background_label:
                        if start_idx > 0 and entities_in_sample[idx] != \
                                entities_in_sample[start_idx - 1]:
                            entities_in_sample[start_idx - 1] = background_label
                            entities_in_sample[idx] = background_label

                    # Reset for next value
                    start_idx = idx + 1
                    label_count = {background_label: 0}
                    if char_pred[idx] == background_label and \
                            sample[idx] in separator_dict:
                        continue

                # Keep count of labels since start
                label = entities_in_sample[idx]
                if label not in label_count:
                    label_count[label] = 0
                label_count[label] += 1

            word_level_predictions.append(entities_in_sample)

        return word_level_predictions

    @staticmethod
    def convert_to_NER_format(predictions, label_mapping, default_label,
                              pad_label):
        """
        Converts word level predictions to specified format

        :param predictions: predictions
        :type predictions: list
        :param label_mapping: labels and corresponding integers
        :type label_mapping: dict
        :param default_label: default label in label_mapping
        :type default_label: str
        :param pad_label: pad label in label_mapping
        :type pad_label: str
        :return: formatted predictions
        :rtype: list
        """
        # get processor parameters
        output_result = []
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}

        default_ind = label_mapping[default_label]
        pad_ind = label_mapping[pad_label]

        # Loop through character_predictions
        for index, sample in enumerate(predictions):
            sample_output = []
            # Initialize loop vars:
            begin_idx = -1
            curr_idx = -1
            curr_label = -1
            # Loop through each sample
            for curr_idx, curr_label in enumerate(sample):
                # Check if begin index has been set
                if begin_idx != -1:
                    if curr_label != sample[begin_idx]:
                        sample_output.append(
                            (begin_idx, curr_idx,
                             reverse_label_mapping[int(sample[begin_idx])]))
                        # Reset for next iteration
                        begin_idx = -1 if curr_label in [pad_ind, default_ind] \
                            else curr_idx
                # Check if need to set a new begin
                elif curr_label not in [pad_ind, default_ind]:
                    begin_idx = curr_idx
            # Check for properly labeling at end of list
            if begin_idx != -1:
                # Add last sample
                sample_output.append(
                    (begin_idx, curr_idx + 1,
                     reverse_label_mapping[(int(curr_label))]))
            # Add to total output list
            output_result.append(sample_output)

        return output_result

    @staticmethod
    def match_sentence_lengths(data, results, flatten_separator, inplace=True):
        """
        Converts the results from the model into the same ragged data shapes as
        the original data.

        :param data: original input data to the data labeler
        :type data: numpy.ndarray
        :param results: dict of model character level predictions and confs
        :type results: dict
        :param flatten_separator: string which joins to samples together when
                                  flattening
        :type flatten_separator: str
        :param inplace: flag to modify results in place
        :type inplace: bool
        :return: dict(pred=...) or dict(pred=..., conf=...)
        """
        pred_buffer = []
        conf_buffer = []
        result_ind = 0
        buffer_add_inds = np.cumsum(list(map(len, results['pred']))).tolist()
        separator_len = len(flatten_separator)

        if not inplace:
            results = copy.deepcopy(results) 

        if results['pred']:
            pred_buffer = np.concatenate(results['pred'])
        results['pred'] = [np.array([])] * len(data)

        if 'conf' in results:
            if results['conf']:
                conf_buffer = np.concatenate(results['conf'])
            results['conf'] = [np.array([])] * len(data)

        # move out of loop bc faster
        data_lens = list(map(lambda x: len(str(x)), data))


        for data_ind in range(len(data)):

            # add empty results for empty data, accounting for flattening
            if data_lens[data_ind] < 1:
                continue

            # overwrite results with expected data from buffer
            results['pred'][data_ind] = \
                pred_buffer[result_ind:result_ind + data_lens[data_ind]]
            if 'conf' in results:
                results['conf'][data_ind] = \
                    conf_buffer[result_ind:result_ind + data_lens[data_ind]]

            result_ind += data_lens[data_ind]
            # if data in buffer, move to next piece of data with separator
            # distance in mind (no separator if flatten location)
            if buffer_add_inds:

                # loop through the indexes where the results were flattened, if
                # result_ind is more than the index, we have past this
                # flattened index and therefore need to pop it.
                curr_buffer_add_in = buffer_add_inds[0]
                while buffer_add_inds and buffer_add_inds[0] <= result_ind:
                    curr_buffer_add_in = buffer_add_inds.pop(0)

                # we need to check for any cases where the flattened results
                # index matches the end of the data index. If this is the case,
                # a separator was not added. All other cases need a flattened
                # index added.
                if (result_ind > curr_buffer_add_in) \
                        or ((curr_buffer_add_in - result_ind) > separator_len):
                    result_ind += separator_len

        return results

    def process(self, data, results, label_mapping):
        """
        Conducts the processing on the data given the predictions,
        label_mapping, and default_label.

        :param data: original input data to the data labeler
        :type data: np.ndarray
        :param results: dict of model character level predictions and confs
        :type results: dict
        :param label_mapping: labels and corresponding integers
        :type label_mapping: dict
        :return: dict of predictions and if they exist, confidences
        """
        # get processor parameters
        output_format = self._parameters['output_format']
        use_word_level_argmax = self._parameters['use_word_level_argmax']
        flatten_separator = self._parameters['flatten_separator']
        default_label = self._parameters['default_label']
        pad_label = self._parameters['pad_label']

        # Format predictions
        # FORMER DEEPCOPY, SHALLOW AS ONLY INTERNAL
        results = self.match_sentence_lengths(data, dict(results),
                                              flatten_separator)
        
        if use_word_level_argmax:
            results['pred'] = self._word_level_argmax(
                data, results['pred'], label_mapping, default_label)
        if output_format.lower() == "ner":
            results['pred'] = self.convert_to_NER_format(
                results['pred'], label_mapping, default_label, pad_label)

        return results


class StructCharPreprocessor(CharPreprocessor,
                             metaclass=AutoSubRegistrationMeta):

    def __init__(self, max_length=3400, default_label='UNKNOWN',
                 pad_label='PAD', flatten_separator="\x01"*5,
                 is_separate_at_max_len=False):
        """
        Initialize the StructCharPreprocessor class

        :param max_length: Maximum char length in a sample.
        :type max_length: int
        :param default_label: Key for label_mapping that is the default label
        :type default_label: string (could be int, char, etc.)
        :param pad_label: Key for label_mapping that is the pad label
        :type pad_label: string (could be int, char, etc.)
        :param flatten_separator: separator used to put between flattened
            samples.
        :type flatten_separator: str
        :param is_separate_at_max_len: if true, separates at max_length,
            otherwise at nearest separator
        :type is_separate_at_max_len: bool
        """
        super().__init__(
            max_length=max_length,
            pad_label=pad_label,
            default_label=default_label,
            flatten_split=1.0,
            flatten_separator=flatten_separator,
            is_separate_at_max_len=is_separate_at_max_len
        )

    def _validate_parameters(self, parameters):
        """
        Validates the parameters set in the processor, raises an error if any
        issues exist.

        :param parameters: parameter dict containing the following parameters:
            max_length: Maximum char length in a sample.
            default_label: Key for label_mapping that is the default label
            pad_label: Key for label_mapping that is the pad label
            flatten_separator: Separator used to put between flattened samples
            is_separate_at_max_len: If true, separates at max_length, otherwise
                at nearest separator
        :type parameters: dict
        :return: None
        :rtype: None
        """
        # flatten_split is hard coded above hence cannot be passed
        parameters = copy.deepcopy(parameters)
        parameters.pop('flatten_split', None)
        super()._validate_parameters(parameters)

    @classmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for preprocessors.

        :return: None
        """
        param_docs = inspect.getdoc(cls._validate_parameters)
        param_start_ind = param_docs.find('parameters:\n') + 12
        param_end_ind = param_docs.find(':type parameters:')

        help_str = (
            cls.__name__ + "\n\n"
            + "Parameters:\n"
            + param_docs[param_start_ind:param_end_ind]
            + "\nProcess Input Format:\n"
            "    data = List of strings ['1st string', 'second string', ...]\n"
            "    labels = [\"<LABEL>\", \"<LABEL>\", ...(num_samples in data)]")
        print(help_str)

    def get_parameters(self, param_list=None):
        """
        Returns a dict of parameters from the model given a list.

        :param param_list: list of parameters to retrieve from the model.
        :type param_list: list
        :return: dict of parameters
        """
        params = super().get_parameters(param_list)
        params.pop('flatten_split', None)
        return params

    def convert_to_unstructured_format(self, data, labels):
        """
        Converts the list of data samples into the CharPreprocessor
        required input data format.

        :param data: list of strings
        :type data: numpy.ndarray
        :param labels: labels for each input character
        :type labels: list
        :return: data in the following format
                 text="<SAMPLE><SEPARATOR><SAMPLE>...",
                 entities=[(start=<INT>, end=<INT>, label="<LABEL>"),
                                  ...(num_samples in data)])
        """
        separator = self._parameters['flatten_separator']
        default_label = self._parameters['default_label']

        text = separator.join(data.astype(str))
        if labels is None:
            return text, None

        text_len = len(text)
        separator_length = len(separator)

        entities = []
        start = 0
        for sample, entity in zip(data, labels):
            if entity != default_label:
                entities.append((start, start + len(sample), entity))
            start += len(sample) + separator_length
            if start < text_len:
                entities.append((start - separator_length, start, 'PAD'))

        return text, entities

    def process(self, data, labels=None, label_mapping=None, batch_size=32):
        """
        Process structured data for being processed by the
        CharacterLevelCnnModel.

        :param data: List of strings to create embeddings for
        :type data: numpy.ndarray
        :param labels: labels for each input character
        :type labels: numpy.ndarray
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: Union[dict, None]
        :param batch_size: Number of samples in the batch of data
        :type batch_size: int
        :return batch_data: A dict containing  samples of size batch_size
        :rtype batch_data: dict
        """
        if labels is not None:
            if not label_mapping:
                raise ValueError('If `labels` are specified, `label_mapping` '
                                 'must also be specified.')
            if data.shape != labels.shape:
                raise ValueError(f"Data and labels given to "
                                 f"StructCharPreprocessor are of different "
                                 f"shapes, {data.shape} != {labels.shape}")

        num_dim = sum([dim > 1 for dim in data.shape])
        if num_dim > 1:
            warnings.warn("Data given to StructCharPreprocessor was "
                          "multidimensional, it will be flattened for model "
                          "processing. Results may be inaccurate, consider "
                          "reformatting data or changing preprocessor.")
        # Flattened data and labels, confirmed to be same shape
        data = data.reshape(-1)
        if labels is not None:
            labels = labels.reshape(-1)

        # convert structured to unstructured format
        unstructured_data = [[]] * len(data)
        unstructured_labels = None if labels is None else [[]] * len(data)

        # with rework, can be tuned to be batches > size 1
        for ind in range(len(data)):
            batch_data = data[ind:ind+1]
            batch_labels = None if labels is None else labels[ind:ind+1]
            unstructured_text, unstructured_label_set = \
                self.convert_to_unstructured_format(batch_data, batch_labels)
            unstructured_data[ind] = unstructured_text
            if labels is not None:
                unstructured_labels[ind] = unstructured_label_set

        if labels is not None:
            np_unstruct_labels = np.array(unstructured_labels, dtype="object")
        else:
            np_unstruct_labels = None

        return super().process(np.array(unstructured_data), np_unstruct_labels,
                               label_mapping, batch_size)


class StructCharPostprocessor(BaseDataPostprocessor,
                              metaclass=AutoSubRegistrationMeta):

    def __init__(self, default_label='UNKNOWN', pad_label='PAD',
                 flatten_separator="\x01"*5, random_state=None):
        """
        Initialize the StructCharPostprocessor class

        :param default_label: Key for label_mapping that is the default label
        :type default_label: str
        :param pad_label: Key for label_mapping that is the pad label
        :type pad_label: str
        :param flatten_separator: separator used to put between flattened
            samples.
        :type flatten_separator: str
        :param random_state: random state setting to be used for randomly
            selecting a prediction when two labels have equal opportunity for
            a given sample.
        :type random_state: random.Random
        """
        if random_state is None:
            random_state = random.Random()
        elif isinstance(random_state, int):
            random_state = random.Random(random_state)
        elif isinstance(random_state, (list, tuple)) and len(random_state) == 3:
            # tuple required for random state to be set, lists do not work
            if isinstance(random_state[1], list):
                random_state[1] = tuple(random_state[1])
            if isinstance(random_state, list):
                random_state = tuple(random_state)
            temp_random_state = random.Random()
            try:
                temp_random_state.setstate(random_state)
                random_state = temp_random_state
            except (TypeError, ValueError):
                pass  # error will raise in validate parameters

        super().__init__(default_label=default_label,
                         pad_label=pad_label,
                         flatten_separator=flatten_separator,
                         random_state=random_state)


    def __eq__(self, other):
        """
        Checks if two processors are equal with one another.

        :param self: a processor
        :param other: a processor
        :type self: StructCharPostprocessor
        :type other: StructCharPostprocessor
        :return: Whether or not self and other are equal
        :rtype: bool
        """

        if type(self) != type(other) \
                or self._parameters["default_label"] != \
                    other._parameters["default_label"]\
                or self._parameters["pad_label"] != \
                    other._parameters["pad_label"]\
                or self._parameters["flatten_separator"] != \
                    other._parameters["flatten_separator"]:
            return False
        return True

    def _validate_parameters(self, parameters):
        """
        Validates the parameters set in the processor, raises an error if any
        issues exist.

        :param parameters: parameter dict containing the following parameters:
            default_label: Key for label_mapping that is the default label
            pad_label: Key for label_mapping that is the pad label
            flatten_separator: Separator used to put between flattened samples
            random_state: Random state setting to be used for randomly
                selecting a prediction when two labels have equal opportunity
                for a given sample.
        :type parameters: dict
        :return: None
        """
        errors = []
        allowed_parameters = self.__class__.__init__.__code__.co_varnames[
                            1:self.__class__.__init__.__code__.co_argcount]
        for param in parameters:
            value = parameters[param]
            if param in ['default_label', 'pad_label', 'flatten_separator'] \
                    and not isinstance(value, str):
                errors.append("`{}` must be a string.".format(param))
            if param == 'random_state' and not isinstance(value, random.Random):
                errors.append('`{}` must be a random.Random.'.format(param))
            elif param not in allowed_parameters:
                errors.append("{} is not an accepted parameter.".format(param))

        if errors:
            raise ValueError('\n'.join(errors))

    @classmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """
        param_docs = inspect.getdoc(cls._validate_parameters)
        param_start_ind = param_docs.find('parameters:\n') + 12
        param_end_ind = param_docs.find(':type parameters:')

        help_str = (
           cls.__name__ + "\n\n"
            + "Parameters:\n"
            + param_docs[param_start_ind:param_end_ind]
            + "\nProcess Output Format:\n"
              "    Each sample receives a label.\n"
              "    Original data - ['My', 'String', ...]\n"
              "    Output labels - ['<LABEL_1>', '<LABEL_2>', "
              "..(num_samples)]")
        print(help_str)

    @staticmethod
    def match_sentence_lengths(data, results, flatten_separator, inplace=True):
        """
        Converts the results from the model into the same ragged data shapes as
        the original data.

        :param data: original input data to the data labeler
        :type data: np.ndarray
        :param results: dict of model character level predictions and confs
        :type results: dict
        :param flatten_separator: string which joins to samples together when
                                  flattening
        :type flatten_separator: str
        :param inplace: flag to modify results in place
        :type inplace: bool
        :return: dict(pred=...) or dict(pred=..., conf=...)
        """
        pred_buffer = []
        conf_buffer = []
        result_ind = 0
        buffer_add_inds = np.cumsum(list(map(len, results['pred']))).tolist()
        separator_len = len(flatten_separator)

        if not inplace:
            results = copy.deepcopy(results) 

        if results['pred']:
            pred_buffer = np.concatenate(results['pred'])
        results['pred'] = [np.array([])] * len(data)

        if 'conf' in results:
            if results['conf']:
                conf_buffer = np.concatenate(results['conf'])
            results['conf'] = [np.array([])] * len(data)

        # move out of loop bc faster
        data_lens = list(map(lambda x: len(str(x)), data))

        for data_ind in range(len(data)):

            # add empty results for empty data, accounting for flattening
            if data_lens[data_ind] < 1:
                continue

            # overwrite results with expected data from buffer
            results['pred'][data_ind] = \
                pred_buffer[result_ind:result_ind + data_lens[data_ind]]
            if 'conf' in results:
                results['conf'][data_ind] = \
                    conf_buffer[result_ind:result_ind + data_lens[data_ind]]

            result_ind += data_lens[data_ind]
            # if data in buffer, move to next piece of data with separator
            # distance in mind (no separator if flatten location)
            if buffer_add_inds:

                # loop through the indexes where the results were flattened, if
                # result_ind is more than the index, we have past this
                # flattened index and therefore need to pop it.
                curr_buffer_add_in = buffer_add_inds[0]
                while buffer_add_inds and buffer_add_inds[0] <= result_ind:
                    curr_buffer_add_in = buffer_add_inds.pop(0)

                # we need to check for any cases where the flattened results
                # index matches the end of the data index. If this is the case,
                # a separator was not added. All other cases need a flattened
                # index added.
                if (result_ind > curr_buffer_add_in) \
                        or ((curr_buffer_add_in - result_ind) > separator_len):
                    result_ind += separator_len

        return results

    def convert_to_structured_analysis(self, sentences, results, label_mapping,
                                       default_label, pad_label):
        """
        Converts unstructured results to a structured column analysis assuming
        the column was flattened into a single sample. This takes the mode of
        all character predictions except for the separator labels. In cases of
        tie, chose anything but background, otherwise randomly choose between
        the remaining labels.

        :param sentences: samples which were predicted upon
        :type sentences: list(str)
        :param results: character predictions for each sample return from model
        :type results: dict
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: dict
        :param default_label: Key for label_mapping that is the default label
        :type default_label: str
        :param pad_label: Key for label_mapping that is the pad label
        :type pad_label: str
        :return: prediction value for a single column
        """
        default_ind = label_mapping[default_label]
        ignore_value = label_mapping[pad_label]
        num_labels = max(label_mapping.values()) + 1

        labels_out = np.ones((len(results['pred']),))
        if 'conf' in results:
            confs_out = np.zeros((len(results['pred']), num_labels))

        for i, label_sentences in enumerate(zip(results['pred'], sentences)):
            column_labels, sentence = label_sentences

            # get count of all labels in prediction
            column_label_counter = Counter(column_labels[:len(sentence)])
            column_label_counter.pop(ignore_value, None)
            modes = [entity_id for entity_id, count in
                     column_label_counter.most_common() if
                     column_label_counter.most_common(1)[0][1] == count]
            if len(modes) == 0:
                labels_out[i] = None
            elif len(modes) == 1:
                labels_out[i] = modes[0]
            else:
                # always choose other than background, otherwise randomly choose
                if default_ind in modes:
                    modes.remove(default_ind)
                labels_out[i] = self._parameters['random_state'].choice(modes)
            if 'conf' in results:
                sample_count = sum(column_label_counter.values())
                for label_id, count in column_label_counter.items():
                    confs_out[i, int(label_id)] = count / sample_count

        results['pred'] = labels_out
        if 'conf' in results:
            results['conf'] = confs_out
        return results

    def process(self, data, results, label_mapping):
        """
        Postprocessing of CharacterLevelCnnModel results when given structured
        data processed by StructCharPreprocessor.

        :param data: original input data to the data labeler
        :type data: Union[numpy.ndarray, pandas.DataFrame]
        :param results: dict of model character level predictions and confs
        :param results: dict
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: dict
        :return: dict of predictions and if they exist, confidences
        :rtype: dict
        """
        # get processor parameters
        flatten_separator = self._parameters['flatten_separator']
        default_label = self._parameters['default_label']
        pad_label = self._parameters['pad_label']

        # Format predictions
        # FORMER DEEPCOPY, SHALLOW AS ONLY INTERNAL
        results = self.match_sentence_lengths(data, dict(results),
                                              flatten_separator)
        results = self.convert_to_structured_analysis(
            data, results,
            label_mapping=label_mapping,
            default_label=default_label,
            pad_label=pad_label)

        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        rev_label_map_vec_func = np.vectorize(
            lambda x: reverse_label_mapping.get(x, None))

        results['pred'] = rev_label_map_vec_func(results['pred'])
        return results

    def _save_processor(self, dirpath):
        """
        Saves the data processor

        :param dirpath: directory to save the processor
        :type dirpath: str
        :return:
        """
        params = copy.deepcopy(self._parameters)
        params['random_state'] = params['random_state'].getstate()
        with open(os.path.join(dirpath,
                               self.processor_type + '_parameters.json'),
                  'w') as fp:
            json.dump(params, fp)


class RegexPostProcessor(BaseDataPostprocessor,
                         metaclass=AutoSubRegistrationMeta):

    def __init__(self, aggregation_func='split', priority_order=None,
                 random_state=None):
        """
        Initialize the RegexPostProcessor class

        :param aggregation_func: aggregation function to apply to regex model
                output (split, random, priority)
        :type aggregation_func: str
        :param priority_order: if priority is set as the aggregation function,
            the order in which entities are given priority must be set
        :type priority_order: Union[list, numpy.ndarray]
        :param random_state: random state setting to be used for randomly
            selecting a prediction when two labels have equal opportunity for
            a given sample.
        :type random_state: random.Random
        """
        if random_state is None:
            random_state = random.Random()
        elif isinstance(random_state, int):
            random_state = random.Random(random_state)
        elif isinstance(random_state, (list, tuple)) and len(random_state) == 3:
            # tuple required for random state to be set, lists do not work
            if isinstance(random_state[1], list):
                random_state[1] = tuple(random_state[1])
            if isinstance(random_state, list):
                random_state = tuple(random_state)
            temp_random_state = random.Random()
            try:
                temp_random_state.setstate(random_state)
                random_state = temp_random_state
            except (TypeError, ValueError):
                pass  # error will raise in validate parameters

        parameters = {
            "aggregation_func": aggregation_func,
            "priority_order": priority_order,
            "random_state": random_state
        }

        super().__init__(**parameters)

    def _validate_parameters(self, parameters):
        """
        Validates the parameters set in the processor, raises an error if any
        issues exist.

        :param parameters: parameter dict containing the following parameters:
            aggregation_func: aggregation function to apply to regex model
                output (split, random, priority)
            priority_order: if priority is set as the aggregation function,
                the order in which entities are given priority must be set
            random_state: Random state setting to be used for randomly
                selecting a prediction when two labels have equal opportunity
                for a given sample.
        :type parameters: dict
        :return: None
        """
        errors = []
        allowed_parameters = self.__class__.__init__.__code__.co_varnames[
                             1:self.__class__.__init__.__code__.co_argcount]
        for param in parameters:
            value = parameters[param]
            if param == 'aggregation_func':
                if not isinstance(value, str):
                    errors.append("`{}` must be a string.".format(param))
                elif value.lower() not in ['split', 'priority', 'random']:
                    errors.append("`{}` must be a one of ['split', 'priority', "
                                  "'random'].".format(param))
            elif param == 'priority_order':
                # if aggregation function is being set to priority, or is not
                # being changed and is already set
                aggregtation_func = parameters.get(
                    'aggregation_func',
                    self._parameters.get('aggregation_func')
                    if hasattr(self, '_parameters') else None)
                if value is None and aggregtation_func == 'priority':
                    errors.append(
                        '`{}` cannot be None if `aggregtation_func` == '
                        'priority.'.format(param))
                elif value is not None \
                        and not isinstance(value, (list, np.ndarray)):
                    errors.append(
                        '`{}` must be a list or numpy.ndarray.'.format(param))
            elif param == 'random_state' \
                    and not isinstance(value, random.Random):
                errors.append('`{}` must be a random.Random.'.format(param))
            elif param not in allowed_parameters:
                errors.append(
                    "{} is not an accepted parameter.".format(param))

        if errors:
            raise ValueError('\n'.join(errors))

    @classmethod
    def help(cls):
        """
        Help function describing alterable parameters, input data formats
        for preprocessors, and output data formats for postprocessors.

        :return: None
        """
        param_docs = inspect.getdoc(cls._validate_parameters)
        param_start_ind = param_docs.find('parameters:\n') + 12
        param_end_ind = param_docs.find(':type parameters:')

        help_str = (
                cls.__name__ + "\n\n"
                + "Parameters:\n"
                + param_docs[param_start_ind:param_end_ind]
                + "\nProcess Output Format:\n"
                  "    Each sample receives a label.\n"
                  "    Original data - ['My', 'String', ...]\n"
                  "    Output labels - ['<LABEL_1>', '<LABEL_2>', "
                  "..(num samples)]")
        print(help_str)

    @staticmethod
    def priority_prediction(results, entity_priority_order):
        """
        Aggregation function using priority of regex to give entity
        determination.

        :param results: regex from model in format: dict(pred=..., conf=...)
        :type results: dict
        :param entity_priority_order: list of entity priorities (lowest has
            higher priority)
        :type entity_priority_order: np.ndarray
        :return: aggregated predictions
        """
        # default aggregation function which selects the first predicted label
        # with the lowest priority of integer.
        for i, pred in enumerate(results['pred']):
            results['pred'][i] = entity_priority_order[
                np.argmax(pred[:, entity_priority_order], axis=1)
            ]

    @staticmethod
    def split_prediction(results):
        """
        Splits the prediction across votes.
        :param results: regex from model in format: dict(pred=..., conf=...)
        :type results: dict
        :return: aggregated predictions
        """
        for i, pred in enumerate(results['pred']):
            results['pred'][i] = \
                pred / np.linalg.norm(pred, axis=1, ord=1, keepdims=True)

    def process(self, data, labels=None, label_mapping=None, batch_size=None):
        """Data preprocessing function."""

        aggregation_func = self._parameters['aggregation_func']
        aggregation_func = aggregation_func.lower()

        results = copy.deepcopy(labels)
        
        if aggregation_func == 'split':
            self.split_prediction(results)
        elif aggregation_func == 'priority':
            self.priority_prediction(
                results, np.array(self._parameters['priority_order']))
        elif aggregation_func == 'random':
            num_labels = max(label_mapping.values()) + 1
            random_state = self._parameters['random_state']
            priority_order = np.array(list(range(num_labels)))
            random_state.shuffle(priority_order)
            self.priority_prediction(results, priority_order)
        else:
            raise ValueError('`{}` is not a valid aggregation function'.format(
                aggregation_func))

        return results

    def _save_processor(self, dirpath):
        """
        Saves the data processor

        :param dirpath: directory to save the processor
        :type dirpath: str
        :return:
        """
        params = copy.deepcopy(self._parameters)
        params['random_state'] = params['random_state'].getstate()
        with open(os.path.join(dirpath,
                               self.processor_type + '_parameters.json'),
                  'w') as fp:
            json.dump(params, fp)
