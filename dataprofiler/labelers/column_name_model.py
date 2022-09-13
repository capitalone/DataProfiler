"""Contains class for regex data labeling model."""
import copy
import json
from operator import neg
import os
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from .. import dp_logging
from .base_model import AutoSubRegistrationMeta, BaseModel

logger = dp_logging.get_child_logger(__name__)


class ColumnNameModel(BaseModel, metaclass=AutoSubRegistrationMeta):
    """Class for column name data labeling model."""

    def __init__(self, parameters=None):
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
        parameters.setdefault('false_positive_dict', None)
        parameters.setdefault('true_positive_dict', None)

        # initialize class
        self._validate_parameters(parameters)
        self._parameters = parameters

    def _validate_parameters(self, parameters):
        r"""
        Validate the parameters sent in.

        Raise error if invalid parameters are present.

        :param parameters: parameter dict containing the following parameters:
            true_positive_dict
            false_positive_dict
        :type parameters: dict
        :return: None
        """
        errors = []

        list_of_accepted_parameters = [
            "true_positive_dict",
            "false_positive_dict",
        ]

        for param in parameters:
            value = parameters[param]
            if param == "false_positive_dict" and value != None and ( 
                not isinstance(value, list)
                or 'attribute' not in value[0].keys()
            ):
                errors.append(
                    "`{}` must be a list of dictionaries with at least 'attribute' as the key".format(param)
                )
            elif param == "true_positive_dict" and (
                not isinstance(value, list)
                or not isinstance(value[0], dict)
            ):
                errors.append(
                    """`{}` must be a list of dictionaries each with the following
                    two keys: 'attribute' and 'label'""".format(param)
                )
            elif param not in list_of_accepted_parameters:
                errors.append("`{}` is not an accepted parameter.".format(param))
        if errors:
            raise ValueError("\n".join(errors))

    @staticmethod
    def _make_lower_case(str, **kwargs):
        return str.lower()

    def _compare_negative(self, list_of_column_names, check_values_dict, negative_threshold):
        """Filter out column name examples that are false positives"""
        scores = self._model(
                list_of_column_names,
                check_values_dict,
                self._make_lower_case,
                fuzz.token_sort_ratio)

        list_of_column_names_filtered = []
        for i in range(len(list_of_column_names)):
            if scores[i][0] < negative_threshold:
                list_of_column_names_filtered.append(list_of_column_names[i])

        return list_of_column_names_filtered

    def _compare_positive(self, list_of_column_names, check_values_dict, positive_threshold, include_label, show_confidences):
        """Calculate similarity scores between list of column names and true positive examples"""

        scores = self._model(
            list_of_column_names,
            check_values_dict,
            self._make_lower_case,
            fuzz.token_sort_ratio,
            include_label=include_label,
        )

        output_dictionary = {}
        for i in range(len(list_of_column_names)):
            if scores[i][0] > positive_threshold:
                output_dictionary[list_of_column_names[i]] = {}
                output_dictionary[list_of_column_names[i]]['pred'] = \
                    check_values_dict[scores[i][1]]['label']
                if show_confidences:
                    output_dictionary[list_of_column_names[i]]['conf'] = scores[i][0]

        return output_dictionary

    def _construct_model(self):
        pass

    def _reconstruct_model(self):
        pass

    def _need_to_reconstruct_model(self):
        pass

    def reset_weights(self):
        pass

    def _model(self, list_of_column_names, check_values_dict, processor, scorer, include_label=False):
        scores = []

        check_values_list = [dict['attribute'] for dict in check_values_dict]

        model_outputs = process.cdist(list_of_column_names,
                            check_values_list,
                            processor=processor,
                            scorer=scorer)

        for iter_value, ngram_match_results in enumerate(model_outputs):
            column_result = [np.max(ngram_match_results)]
            if include_label:
                index_max_result = ngram_match_results.argmax(axis=0)
                column_result.append(index_max_result)
            scores.append(column_result)
        
        return scores

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
        false_positive_dict = self._parameters['false_positive_dict']
        if false_positive_dict:
            data = self._compare_negative(data, false_positive_dict, negative_threshold=50)
            if verbose:
                logger.info("compare_negative process complete")
        
        output = self._compare_positive(
                                        data,
                                        self._parameters['true_positive_dict'],
                                        positive_threshold=85,
                                        include_label=True,
                                        show_confidences=show_confidences)
        if verbose:
            logger.info("compare_positive process complete")

        return output

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

        loaded_model = cls(parameters)
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
