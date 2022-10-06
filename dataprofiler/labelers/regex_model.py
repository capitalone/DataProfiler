"""Contains class for regex data labeling model."""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from typing import Dict

import numpy as np

from dataprofiler._typing import DataArray

from .. import dp_logging
from .base_model import AutoSubRegistrationMeta, BaseModel

logger = dp_logging.get_child_logger(__name__)


class RegexModel(BaseModel, metaclass=AutoSubRegistrationMeta):
    """Class for regex data labeling model."""

    def __init__(self, label_mapping: Dict[str, int], parameters: Dict = None) -> None:
        r"""
        Initialize Regex Model.

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

        Example encapsulators:
            encapsulators = {
                'start': r'(?<![\w.\$\%\-])',
                'end': r'(?:(?=(\b|[ ]))|(?=[^\w\%\$]([^\w]|$))|$)',
            }

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
        parameters.setdefault("regex_patterns", {})
        parameters.setdefault("encapsulators", {"start": "", "end": ""})
        parameters.setdefault("ignore_case", True)
        parameters.setdefault("default_label", "UNKNOWN")
        self._epoch_id = 0

        # initialize class
        self.set_label_mapping(label_mapping)
        self._validate_parameters(parameters)
        self._parameters = parameters

    def _validate_parameters(self, parameters: Dict) -> None:
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
        _retype = type(re.compile("pattern for py 3.6 & 3.7"))

        errors = []

        list_of_necessary_params = [
            "encapsulators",
            "regex_patterns",
            "ignore_case",
            "default_label",
        ]
        # Make sure the necessary parameters are present and valid.
        for param in parameters:
            value = parameters[param]
            if param == "encapsulators" and (
                not isinstance(value, dict)
                or "start" not in value
                or "end" not in value
            ):
                errors.append(
                    "`{}` must be a dict with keys 'start' and 'end'".format(param)
                )
            elif param == "regex_patterns":
                if not isinstance(value, dict):
                    errors.append(
                        "`{}` must be a dict of regex pattern lists.".format(param)
                    )
                    continue
                for key in value:
                    if key not in self.label_mapping:
                        errors.append(
                            "`{}` was a regex pattern not found in the "
                            "label_mapping".format(key)
                        )
                    elif not isinstance(value[key], list):
                        errors.append(
                            "`{}` must be a list of regex patterns, i.e."
                            "[pattern_1, pattern_2, ...]".format(key)
                        )
                    else:
                        for i in range(len(value[key])):
                            if not isinstance(value[key][i], (_retype, str)):
                                errors.append(
                                    "`{}`, pattern `{}' was not a valid regex "
                                    "pattern (re.Pattern, str)".format(key, i)
                                )
                            elif isinstance(value[key][i], str):
                                try:
                                    re.compile(value[key][i])
                                except re.error as e:
                                    errors.append(
                                        "`{}`, pattern {} was not a valid regex"
                                        " pattern: {}".format(key, i, str(e))
                                    )
            elif param == "ignore_case" and not isinstance(parameters[param], bool):
                errors.append("`{}` must be a bool.".format(param))
            elif param == "default_label" and not isinstance(parameters[param], str):
                errors.append("`{}` must be a string.".format(param))
            elif param not in list_of_necessary_params:
                errors.append("`{}` is not an accepted parameter.".format(param))
        if errors:
            raise ValueError("\n".join(errors))

    def _construct_model(self) -> None:
        pass

    def _reconstruct_model(self) -> None:
        pass

    def _need_to_reconstruct_model(self) -> bool:
        pass

    def reset_weights(self) -> None:
        """Reset weights."""
        pass

    def predict(
        self,
        data: DataArray,
        batch_size: int = None,
        show_confidences: bool = False,
        verbose: bool = True,
    ) -> Dict:
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
        start_pattern = ""
        end_pattern = ""
        regex_patterns = self._parameters["regex_patterns"]
        default_ind = self.label_mapping[self._parameters["default_label"]]
        encapsulators = self._parameters["encapsulators"]
        re_flags = re.IGNORECASE if self._parameters["ignore_case"] else 0

        if encapsulators:
            start_pattern = encapsulators["start"]
            end_pattern = encapsulators["end"]

        pre_compiled_patterns = copy.deepcopy(regex_patterns)
        for entity_label, entity_patterns in pre_compiled_patterns.items():
            for i in range(len(entity_patterns)):
                pattern = (
                    start_pattern + pre_compiled_patterns[entity_label][i] + end_pattern
                )
                pre_compiled_patterns[entity_label][i] = re.compile(
                    pattern, flags=re_flags
                )

        # Construct array initial regex predictions where background is
        # predicted.
        predictions = [np.empty((0,))] * 100
        i = 0
        for i, input_string in enumerate(data):

            # Double array size
            if len(predictions) <= i:
                predictions.extend([np.empty((0,))] * len(predictions))

            pred = np.zeros((len(input_string), self.num_labels), dtype=int)
            pred[:, default_ind] = 1

            for entity_label, entity_patterns in pre_compiled_patterns.items():
                entity_id = self.label_mapping[entity_label]
                for re_pattern in entity_patterns:

                    for each_find in re_pattern.finditer(input_string):
                        indices = each_find.span(0)
                        pred[indices[0] : indices[1], default_ind] = 0
                        pred[indices[0] : indices[1], entity_id] = 1
            if verbose:
                sys.stdout.flush()
                sys.stdout.write("\rData Samples Processed: {:d}   ".format(i + 1))
            predictions[i] = pred

        if verbose:
            logger.info("\rData Samples Processed: {:d}   ".format(i + 1))

        # Trim array size to number of samples
        if len(predictions) > i + 1:
            del predictions[i + 1 :]

        if show_confidences:
            conf = copy.deepcopy(predictions)
            for i in range(len(conf)):
                conf[i] = conf[i] / np.linalg.norm(
                    conf[i], axis=1, ord=1, keepdims=True
                )
            return {"pred": predictions, "conf": conf}
        return {"pred": predictions}

    @classmethod
    def load_from_disk(cls, dirpath: str) -> RegexModel:
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

    def save_to_disk(self, dirpath: str) -> None:
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
