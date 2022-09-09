"""Contains class for regex data labeling model."""
import copy
import json
from operator import neg
import os
import re
import sys

import numpy as np
from rapidfuzz import process, fuzz

from .. import dp_logging
from .base_model import AutoSubRegistrationMeta, BaseModel

logger = dp_logging.get_child_logger(__name__)


class ColumnNameModel(BaseModel, metaclass=AutoSubRegistrationMeta):
    """Class for column name data labeling model."""

    def __init__(self, label_mapping=None, parameters=None):
        r"""
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: dict
        :param parameters: Contains all the appropriate parameters for the model.
            Possible parameters are:
                max_length, max_num_chars, dim_embed
        :type parameters: dict
        :return: None
        """
        # parameter initialization
        if not parameters:
            parameters = {}
        parameters.setdefault('negative_dataframe', )
        parameters.setdefault('positive_dataframe', )

        # initialize class
        self.set_label_mapping(label_mapping)
        self._validate_parameters(parameters)
        self._parameters = parameters

    def _validate_parameters(self, parameters):
        r"""
        Validate the parameters sent in.

        Raise error if invalid parameters are present.

        :param parameters: parameter dict containing the following parameters:
            regex_patterns: patterns associated with each label_mapping
                Example regex_patterns:
                    regex_patterns = {
                        "LABEL_1": [
                            "LABEL_1_pattern_1",
                            "LABEL_1_pattern_2",
                            ...
                        ],
                        "LABEL_2": [
                            "LABEL_2_pattern_1",
                            "LABEL_2_pattern_2",
                            ...
                        ],
                        ...
                    }
            encapsulators: regex to add to start and end of each regex
                (used to capture entities inside of text).
                Example encapsulators:
                    encapsulators = {
                        'start': r'(?<![\w.\$\%\-])',
                        'end': r'(?:(?=(\b|[ ]))|(?=[^\w\%\$]([^\w]|$))|$)',
                    }
            ignore_case: whether or not to set the regex ignore case flag
            default_label: default label to assign when no regex found
        :type parameters: dict
        :return: None
        """
        raise NotImplementedError()

    def _compare_negative(self, list_of_column_names, check_dataframe, negative_threshold):
        """Filter out column name examples that are false positives"""
        scores = self._model(
                list_of_column_names,
                check_dataframe,
                self._make_lower_case(),
                fuzz.token_sort_ratio,
                include_label=True)

        list_of_column_names_filtered = []
        for i in range(len(list_of_column_names)):
            if scores[i][0] < negative_threshold:
                list_of_column_names_filtered.append(list_of_column_names[i])

        return list_of_column_names_filtered

    def _compare_positive(self, list_of_column_names, check_dataframe, positive_threshold, include_label):
        """Run the model on examples that are true positives"""
        
        scores = self._model(
            list_of_column_names,
            check_dataframe,
            self._make_lower_case(),
            fuzz.token_sort_ratio,
            include_label=include_label,
        )

        output_dictionary = {}
        for i in range(len(list_of_column_names)):
            if scores[i][0] > positive_threshold:
                output_dictionary[list_of_column_names[i]] = {}
                output_dictionary[list_of_column_names[i]]['prediction'] = \
                    check_dataframe['label'].iloc[scores[i][1]]
                output_dictionary[list_of_column_names[i]]['Similarity Score'] = scores[i][0]

        return output_dictionary

    def _construct_model(self):
        pass

    def _reconstruct_model(self):
        pass
    
    def _make_lower_case(str, **kwargs):
        return str.lower()

    def _model(list_of_column_names, sensitive_terms_df, processor, scorer, include_label=True):
        scores = []
        model_outputs = process.cdist(list_of_column_names,
                            sensitive_terms_df['attribute'],
                            processor=processor,
                            scorer=scorer)

        for iter_value, ngram_match_results in enumerate(model_outputs):
            column_result = [np.max(ngram_match_results)]
            if include_label:
                index_max_result = ngram_match_results.argmax(axis=0)
                column_result.append(index_max_result)
            scores.append(column_result)
        return scores

    def _need_to_reconstruct_model(self):
        pass

    def reset_weights(self):
        """Reset weights."""
        pass

    def predict(self, data, batch_size=None, show_confidences=False, verbose=True, include_label=True):
        """
        Apply the regex patterns (within regex_model) to the input_string.

        Create predictions for all matching patterns. Each pattern has an
        associated entity and the predictions of each character within the
        string are given a True or False identification for each entity. All
        characters not identified by ANY of the regex patterns in the
        pattern_dict are considered background characters, and are replaced with
        the default_label value.

        :param data: list of strings to predict upon
        :type data: iterator
        :param batch_size: does not impact this model and should be fixed to not
            be required.
        :type batch_size: N/A
        :param show_confidences: whether user wants prediction confidences
        :type show_confidences:
        :param verbose: Flag to determine whether to print status or not
        :type verbose: bool
        :return: char level predictions and confidences
        :rtype: dict
        """
        negative_df = self._parameters['negative_dataframe']
        if negative_df:
            data = self._compare_negative(data, negative_df, negative_threshold=50, include_label=False)
        output = self._compare_positive(
                                        data,
                                        self._parameters['positive_dataframe'],
                                        positive_threshold=85,
                                        include_label=True)

        if show_confidences:
            return {"pred": predictions, "conf": conf}
        return {"pred": predictions}

    @classmethod
    def load_from_disk(cls, dirpath):
        """
        Load whole model from disk with weights.

        :param dirpath: directory path where you want to load the model from
        :type dirpath: str
        :return: None
        """
        # load parameters
        model_param_dirpath = os.path.join(dirpath, "model_parameters.json")
        with open(model_param_dirpath, "r") as fp:
            parameters = json.load(fp)

        # load label_mapping
        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath, "r") as fp:
            label_mapping = json.load(fp)

        loaded_model = cls(label_mapping, parameters)
        return loaded_model

    def save_to_disk(self, dirpath):
        """
        Save whole model to disk with weights.

        :param dirpath: directory path where you want to save the model to
        :type dirpath: str
        :return: None
        """
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        model_param_dirpath = os.path.join(dirpath, "model_parameters.json")
        with open(model_param_dirpath, "w") as fp:
            json.dump(self._parameters, fp)

        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath, "w") as fp:
            json.dump(self.label_mapping, fp)
