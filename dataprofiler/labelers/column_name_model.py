"""Contains class for column name data labeling model."""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

import numpy as np

from dataprofiler._typing import DataArray

try:
    import rapidfuzz
except ImportError:
    # not required when using `ColumnNameModel`, but
    # will below recommend to install if not installed'
    # when running `predict()`
    pass

from .. import dp_logging
from .base_model import AutoSubRegistrationMeta, BaseModel
from .utils import require_module

logger = dp_logging.get_child_logger(__name__)


class ColumnNameModel(BaseModel, metaclass=AutoSubRegistrationMeta):
    """Class for column name data labeling model."""

    def __init__(self, label_mapping: Dict[str, int], parameters: Dict = None) -> None:
        """Initialize function for ColumnNameModel.

        :param parameters: Contains all the appropriate parameters for the model.
            Possible parameters are:
                max_length, max_num_chars, dim_embed
        :type parameters: dict
        :return: None
        """
        # parameter initialization
        if not parameters:
            parameters = {}
        parameters.setdefault("false_positive_dict", None)
        parameters.setdefault("true_positive_dict", None)
        parameters.setdefault("include_label", True)
        parameters.setdefault("negative_threshold_config", None)
        parameters.setdefault("positive_threshold_config", None)

        # validate and set parameters
        self.set_label_mapping(label_mapping)
        self._validate_parameters(parameters)
        self._parameters = parameters

    def _validate_parameters(self, parameters: Dict) -> None:
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

        required_parameters = [
            "true_positive_dict",
        ]

        optional_parameters = [
            "false_positive_dict",
            "include_label",
            "negative_threshold_config",
            "positive_threshold_config",
        ]

        list_of_accepted_parameters = optional_parameters + required_parameters

        if parameters["true_positive_dict"]:
            label_map_dict_keys = set(self.label_mapping.keys())
            true_positive_unique_labels = set(
                parameters["true_positive_dict"][0].values()
            )

            # if not a subset that is less than or equal to
            # label mapping dict
            if true_positive_unique_labels > label_map_dict_keys:
                errors.append(
                    """`true_positive_dict` must be a subset
                        of the `label_mapping` values()"""
                )

        for param in parameters:
            value = parameters[param]
            if (
                param == "false_positive_dict"
                and value is not None
                and (not isinstance(value, list) or "attribute" not in value[0].keys())
                and parameters["negative_threshold_config"] is not None
            ):
                errors.append(
                    """`{}` must be a list of dictionaries with at
                    least 'attribute' as the key""".format(
                        param
                    )
                )
            elif param == "true_positive_dict" and (
                not isinstance(value, list)
                or not isinstance(value[0], dict)
                and "attribute" not in value[0].keys()
                and "label" not in value[0].keys()
            ):
                errors.append(
                    """`{}` is a required parameters that must be a list
                    of dictionaries each with the following
                    two keys: 'attribute' and 'label'""".format(
                        param
                    )
                )
            elif param == "include_label" and not isinstance(value, bool):
                errors.append(
                    "`{}` is a required parameter that must be a boolean.".format(param)
                )
            elif param == "negative_threshold_config" and (
                not isinstance(value, int)
                and parameters["false_positive_dict"] is not None
            ):
                errors.append(
                    "`{}` is an optional parameter that must be a boolean.".format(
                        param
                    )
                )
            elif param == "positive_threshold_config" and (
                value is None or not isinstance(value, int)
            ):
                errors.append(
                    "`{}` is a required parameter that must be a boolean.".format(param)
                )
            elif param not in list_of_accepted_parameters:
                errors.append("`{}` is not an accepted parameter.".format(param))

        if errors:
            raise ValueError("\n".join(errors))

    @staticmethod
    def _make_lower_case(str: str, **kwargs: Any) -> str:
        return str.lower()

    def _compare_negative(
        self,
        list_of_column_names: DataArray,
        check_values_dict: Dict,
        negative_threshold: float,
    ) -> DataArray:
        """Filter out column name examples that are false positives."""
        scores = self._model(
            list_of_column_names,
            check_values_dict,
            self._make_lower_case,
            rapidfuzz.fuzz.token_sort_ratio,
        )

        list_of_column_names_filtered = []
        for i in range(len(list_of_column_names)):
            if scores[i][0] < negative_threshold:
                list_of_column_names_filtered.append(list_of_column_names[i])

        return list_of_column_names_filtered

    def _construct_model(self) -> None:
        pass

    def _reconstruct_model(self) -> None:
        pass

    def _need_to_reconstruct_model(self) -> bool:
        pass

    def reset_weights(self) -> None:
        """Reset weights function."""
        pass

    @require_module(["rapidfuzz"])
    def _model(
        self,
        list_of_column_names: List[str],
        check_values_dict: List[Dict],
        processor: Callable,
        scorer: Callable,
        include_label: bool = False,
    ) -> List:
        scores = []

        check_values_list = [dict["attribute"] for dict in check_values_dict]

        model_outputs = rapidfuzz.process.cdist(
            list_of_column_names, check_values_list, processor=processor, scorer=scorer
        )

        for iter_value, ngram_match_results in enumerate(model_outputs):
            column_result = [np.max(ngram_match_results)]
            if include_label:
                index_max_result = ngram_match_results.argmax(axis=0)
                column_result.append(index_max_result)
            scores.append(column_result)

        return scores

    def predict(
        self,
        data: DataArray,
        batch_size: int = None,
        show_confidences: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Apply the `process.cdist` for similarity score on input list of strings.

        :param data: list of strings to predict upon
        :type data: iterator
        :param batch_size: does not impact this model and should be fixed to not
            be required.
        :type batch_size: N/A
        :param show_confidences: Parameter disabled. Confidence values returned
            by default.
        :type show_confidences:
        :param verbose: Flag to determine whether to print status or not
        :type verbose: bool
        :return: char level predictions and confidences
        :rtype: dict
        """
        false_positive_dict = self._parameters["false_positive_dict"]
        if false_positive_dict:
            data = self._compare_negative(
                data,
                false_positive_dict,
                negative_threshold=self._parameters["negative_threshold_config"],
            )
            if verbose:
                logger.info("compare_negative process complete")

        # old compare_positive
        output = self._model(
            data,
            self._parameters["true_positive_dict"],
            processor=self._make_lower_case,
            scorer=rapidfuzz.fuzz.token_sort_ratio,
            include_label=self._parameters["include_label"],
        )

        predictions = np.array([])
        confidences = np.array([])

        # `data` at this point is either filtered or not filtered
        # list of column names on which we are predicting
        for iter_value, value in enumerate(data):

            if output[iter_value][0] > self._parameters["positive_threshold_config"]:
                predictions = np.append(
                    predictions,
                    self._parameters["true_positive_dict"][output[iter_value][1]][
                        "label"
                    ],
                )

                if show_confidences:
                    confidences = np.append(confidences, output[iter_value][0])

        if verbose:
            logger.info("compare_positive process complete")

        if show_confidences:
            return {"pred": predictions, "conf": confidences}
        return {"pred": predictions}

    @classmethod
    def load_from_disk(cls, dirpath: str) -> ColumnNameModel:
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
