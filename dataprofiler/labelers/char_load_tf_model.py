"""Contains class for training data labeler model."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import cast

import numpy as np
import tensorflow as tf

from dataprofiler._typing import DataArray

from .. import dp_logging
from . import labeler_utils
from .base_model import AutoSubRegistrationMeta, BaseModel, BaseTrainableModel

_file_dir = os.path.dirname(os.path.abspath(__file__))

logger = dp_logging.get_child_logger(__name__)
labeler_utils.hide_tf_logger_warnings()


class CharLoadTFModel(BaseTrainableModel, metaclass=AutoSubRegistrationMeta):
    """For training data labeler model."""

    # boolean if the label mapping requires the mapping for index 0 reserved
    requires_zero_mapping = False

    def __init__(
        self, model_path: str, label_mapping: dict[str, int], parameters: dict = None
    ) -> None:
        """
        Initialize Loadable TF Model.

        :param model_path: path to model to load
        :type model_path: str
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: dict
        :param parameters: Contains all the appropriate parameters for the
            model. Must contain num_labels. Other possible parameters are:
                max_length, max_char_encoding_id, dim_embed, size_fc
                dropout, size_conv, num_fil, optimizer, default_label
        :type parameters: dict
        :return: None
        """
        # parameter initialization
        if not parameters:
            parameters = {}
        parameters.setdefault("default_label", "UNKNOWN")
        parameters["model_path"] = model_path
        parameters["pad_label"] = "PAD"
        self._epoch_id = 0

        # reconstruct flags for model
        self._model_num_labels = 0
        self._model_default_ind = -1

        BaseModel.__init__(self, label_mapping, parameters)

    def __eq__(self, other: object) -> bool:
        """
        Check if two models are equal with one another.

        May only check important variables, i.e. may not check model itself.

        :param self: a model
        :param other: a model
        :type self: BaseModel
        :type other: BaseModel
        :return: Whether or not self and other are equal
        :rtype: bool
        """
        if (
            not isinstance(other, BaseModel)
            or self._parameters != other._parameters
            or self._label_mapping != other._label_mapping
        ):
            return False
        return True

    def _validate_parameters(self, parameters: dict) -> None:
        """
        Validate the parameters sent in.

        Raise error if invalid parameters are present.

        :param parameters: parameter dict containing the following parameters:
            max_length: Maximum char length in a sample
            max_char_encoding_id: Maximum integer value for encoding the input
            dim_embed: Number of embedded dimensions
            size_fc: Size of each fully connected layers
            dropout: Ratio of dropout in the model
            size_conv: Convolution kernel size
            default_label: Key for label_mapping that is the default label
            pad_label: Key for entities_dict that is the pad label
            num_fil: Number of filters in each convolution layer
        :type parameters: dict
        :return: None
        """
        errors = []
        list_of_necessary_params = ["model_path", "default_label", "pad_label"]

        # Make sure the necessary parameters are present and valid.
        for param in parameters:
            if param in list_of_necessary_params:
                if not isinstance(parameters[param], str):
                    error = str(param) + " must be a string."
                    errors.append(error)

        # Error if there are extra parameters thrown in
        for param in parameters:
            if param not in list_of_necessary_params:
                errors.append(param + " is not an accepted parameter.")
        if errors:
            raise ValueError("\n".join(errors))

    def set_label_mapping(self, label_mapping: list[str] | dict[str, int]) -> None:
        """
        Set the labels for the model.

        :param label_mapping: label mapping of the model
        :type label_mapping: dict
        :return: None
        """
        if not isinstance(label_mapping, (list, dict)):
            raise TypeError(
                "Labels must either be a non-empty encoding dict "
                "which maps labels to index encodings or a list."
            )

        label_mapping = copy.deepcopy(label_mapping)
        if "PAD" not in label_mapping:
            if isinstance(label_mapping, list):  # if list missing PAD
                label_mapping = ["PAD"] + label_mapping
            elif 0 not in label_mapping.values():  # if dict missing PAD and 0
                label_mapping.update({"PAD": 0})
            else:
                label_mapping.update({"PAD": max(list(label_mapping.values())) + 1})
        if self._parameters["default_label"] not in label_mapping:
            raise ValueError(
                "The `default_label` of {} must exist in the "
                "label mapping.".format(self._parameters["default_label"])
            )
        super().set_label_mapping(label_mapping)

    def _need_to_reconstruct_model(self) -> bool:
        """
        Determine whether or not the model needs to be reconstructed.

        :return: bool of whether or not the model needs to reconstruct.
        """
        if not self._model:
            return False
        default_ind = self.label_mapping[self._parameters["default_label"]]
        return (
            self.num_labels != self._model_num_labels
            or default_ind != self._model_default_ind
        )

    def save_to_disk(self, dirpath: str) -> None:
        """
        Save whole model to disk with weights.

        :param dirpath: directory path where you want to save the model to
        :type dirpath: str
        :return: None
        """
        if not self._model:
            self._construct_model()
        elif self._need_to_reconstruct_model():
            self._reconstruct_model()

        model_param_dirpath = os.path.join(dirpath, "model_parameters.json")
        model_parameters = self._parameters.copy()
        model_parameters.pop("model_path")
        with open(model_param_dirpath, "w") as fp:
            json.dump(model_parameters, fp)
        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath, "w") as fp:
            json.dump(self.label_mapping, fp)
        self._model.save(dirpath)

    @classmethod
    def load_from_disk(cls, dirpath: str) -> CharLoadTFModel:
        """
        Load whole model from disk with weights.

        :param dirpath: directory path where you want to load the model from
        :type dirpath: str
        :return: loaded CharLoadTFModel
        :rtype: CharLoadTFModel
        """
        # load parameters
        model_param_dirpath = os.path.join(dirpath, "model_parameters.json")
        with open(model_param_dirpath) as fp:
            parameters = json.load(fp)

        # load label_mapping
        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath) as fp:
            label_mapping = json.load(fp)

        # use f1 score metric
        custom_objects = {
            "F1Score": labeler_utils.F1Score(
                num_classes=max(label_mapping.values()) + 1, average="micro"
            ),
            "CharLoadTFModel": cls,
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            tf_model = tf.keras.models.load_model(dirpath)

        loaded_model = cls(dirpath, label_mapping, parameters)
        loaded_model._model = tf_model

        # load self
        loaded_model._model_num_labels = loaded_model.num_labels
        loaded_model._model_default_ind = loaded_model.label_mapping[
            loaded_model._parameters["default_label"]
        ]
        return loaded_model

    def _construct_model(self) -> None:
        """
        Model constructor for the data labeler.

        This also serves as a weight reset.

        :return: None
        """
        num_labels = self.num_labels
        default_ind = self.label_mapping[self._parameters["default_label"]]
        model_loc = self._parameters["model_path"]

        self._model: tf.keras.Model = tf.keras.models.load_model(model_loc)
        self._model = tf.keras.Model(self._model.inputs, self._model.outputs)
        softmax_output_layer_name = self._model.output_names[0]
        softmax_layer_ind = cast(
            int,
            labeler_utils.get_tf_layer_index_from_name(
                self._model, softmax_output_layer_name
            ),
        )
        softmax_layer = self._model.get_layer(softmax_output_layer_name)

        new_softmax_layer = softmax_layer.output
        if softmax_layer.weights[0].shape[-1] != num_labels:
            new_softmax_layer = tf.keras.layers.Dense(
                num_labels, activation="softmax", name="softmax_output"
            )(self._model.layers[softmax_layer_ind - 1].output)

        # Add argmax layer to get labels directly as an output
        argmax_layer = tf.keras.ops.argmax(new_softmax_layer, axis=2)

        argmax_outputs = [new_softmax_layer, argmax_layer]
        self._model = tf.keras.Model(self._model.inputs, argmax_outputs)
        self._model = tf.keras.Model(self._model.inputs, self._model.outputs)

        # Compile the model w/ metrics
        softmax_output_layer_name = self._model.output_names[0]
        losses = {softmax_output_layer_name: "categorical_crossentropy"}

        # use f1 score metric
        f1_score_training = labeler_utils.F1Score(
            num_classes=num_labels, average="micro"
        )
        metrics = {
            softmax_output_layer_name: [
                "categorical_crossentropy",
                "acc",
                f1_score_training,
            ]
        }

        self._model.compile(loss=losses, optimizer="adam", metrics=metrics)

        self._epoch_id = 0
        self._model_num_labels = num_labels
        self._model_default_ind = default_ind

    def reset_weights(self) -> None:
        """
        Reset the weights of the model.

        :return: None
        """
        self._construct_model()

    def _reconstruct_model(self) -> None:
        """
        Reconstruct appropriate layers if number of number of labels is altered.

        :return: None
        """
        # Reset model
        tf.keras.backend.clear_session()

        num_labels = self.num_labels
        default_ind = self.label_mapping[self._parameters["default_label"]]

        # Add the final Softmax layer to the previous spot
        # self._model.layers[-2] to skip: original softmax
        final_softmax_layer = tf.keras.layers.Dense(
            num_labels, activation="softmax", name="softmax_output"
        )(self._model.layers[-2].output)

        # Add argmax layer to get labels directly as an output
        argmax_layer = tf.keras.ops.argmax(final_softmax_layer, axis=2)

        argmax_outputs = [final_softmax_layer, argmax_layer]
        self._model = tf.keras.Model(self._model.inputs, argmax_outputs)

        # Compile the model
        softmax_output_layer_name = self._model.output_names[0]
        losses = {softmax_output_layer_name: "categorical_crossentropy"}

        # use f1 score metric
        f1_score_training = labeler_utils.F1Score(
            num_classes=num_labels, average="micro"
        )
        metrics = {
            softmax_output_layer_name: [
                "categorical_crossentropy",
                "acc",
                f1_score_training,
            ]
        }

        self._model.compile(loss=losses, optimizer="adam", metrics=metrics)

        self._epoch_id = 0
        self._model_num_labels = num_labels
        self._model_default_ind = default_ind

    def fit(
        self,
        train_data: DataArray,
        val_data: DataArray = None,
        batch_size: int = None,
        epochs: int = None,
        label_mapping: dict[str, int] = None,
        reset_weights: bool = False,
        verbose: bool = True,
    ) -> tuple[dict, float | None, dict]:
        """
        Train the current model with the training data and validation data.

        :param train_data: Training data used to train model
        :type train_data: Union[list, np.ndarray]
        :param val_data: Validation data used to validate the training
        :type val_data: Union[list, np.ndarray]
        :param batch_size: Used to determine number of samples in each batch
        :type batch_size: int
        :param label_mapping: maps labels to their encoded integers
        :type label_mapping: Union[dict, None]
        :param reset_weights: Flag to determine whether to reset the weights or
            not
        :type reset_weights: bool
        :param verbose: Flag to determine whether to print status or not
        :type verbose: bool
        :return: history, f1, f1_report
        :rtype: Tuple[dict, float, dict]
        """
        if label_mapping is not None:
            self.set_label_mapping(label_mapping)

        if not self._model:
            self._construct_model()
        else:
            if self._need_to_reconstruct_model():
                self._reconstruct_model()
            if reset_weights:
                self.reset_weights()

        history: dict = defaultdict()
        f1: float | None = None
        f1_report: dict = {}

        self._model.reset_metrics()
        softmax_output_layer_name = self._model.output_names[0]

        start_time = time.time()
        batch_id = 0
        for x_train, y_train in train_data:
            model_results = self._model.train_on_batch(
                x_train, {softmax_output_layer_name: y_train}
            )
            sys.stdout.flush()
            if verbose:
                sys.stdout.write(
                    "\rEPOCH %d, batch_id %d: loss: %f - acc: %f - "
                    "f1_score %f" % (self._epoch_id, batch_id, *model_results[1:])
                )
            batch_id += 1

        for i, metric_label in enumerate(self._model.metrics_names):
            history[metric_label] = model_results[i]

        if val_data:
            f1, f1_report = self._validate_training(val_data)  # type: ignore
            history["f1_report"] = f1_report

            val_f1 = f1_report["weighted avg"]["f1-score"] if f1_report else np.NAN
            val_precision = (
                f1_report["weighted avg"]["precision"] if f1_report else np.NAN
            )
            val_recall = f1_report["weighted avg"]["recall"] if f1_report else np.NAN
            epoch_time = time.time() - start_time
            logger.info(
                "\rEPOCH %d (%ds), loss: %f - acc: %f - f1_score %f -- "
                "val_f1: %f - val_precision: %f - val_recall %f"
                % (
                    self._epoch_id,
                    epoch_time,
                    *model_results[1:],
                    val_f1,
                    val_precision,
                    val_recall,
                )
            )

        self._epoch_id += 1

        return history, f1, f1_report

    def _validate_training(
        self,
        val_data: DataArray,
        batch_size_test: int = 32,
        verbose_log: bool = True,
        verbose_keras: bool = False,
    ) -> tuple[float, dict] | tuple[None, None]:
        """
        Validate the model on the test set and return the evaluation metrics.

        :param val_data: data generator for the validation
        :type val_data: iterator
        :param batch_size_test: Number of samples to process in testing
        :type batch_size_test: int
        :param verbose_log: whether or not to print out scores for training,
            etc.
        :type verbose_log: bool
        :param verbose_keras: whether or not to print out scores for training,
            from keras.
        :type verbose_keras: bool
        return (f1-score, f1 report).
        """
        f1 = None
        f1_report = None

        if val_data is None:
            return f1, f1_report

        # Predict on the test set
        batch_id = 0
        y_val_pred = []
        y_val_test = []
        for x_val, y_val in val_data:
            y_val_pred.append(
                self._model.predict(
                    x_val, batch_size=batch_size_test, verbose=verbose_keras
                )[1]
            )
            y_val_test.append(np.argmax(y_val, axis=-1))
            batch_id += 1
            sys.stdout.flush()
            if verbose_log:
                sys.stdout.write(
                    "\rEPOCH %g, validation_batch_id %d" % (self._epoch_id, batch_id)
                )

        tf.keras.backend.set_floatx("float32")
        # Clean the predicted entities and the actual entities
        f1, f1_report = labeler_utils.evaluate_accuracy(
            np.concatenate(y_val_pred, axis=0),
            np.concatenate(y_val_test, axis=0),
            self.num_labels,
            self.reverse_label_mapping,
            verbose=verbose_keras,
        )

        return f1, f1_report

    def predict(
        self,
        data: DataArray,
        batch_size: int = 32,
        show_confidences: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Run model and get predictions.

        :param data: text input
        :type data: Union[list, numpy.ndarray]
        :param batch_size: number of samples in the batch of data
        :type batch_size: int
        :param show_confidences: whether user wants prediction confidences
        :type show_confidences:
        :param verbose: Flag to determine whether to print status or not
        :type verbose: bool
        :return: char level predictions and confidences
        :rtype: dict
        """
        if not self._model:
            self._construct_model()
        elif self._need_to_reconstruct_model():
            raise RuntimeError(
                "The model label mapping definitions have been "
                "altered without additional training. Please "
                "train the model or reset the label mapping to "
                "predict."
            )
        # Pre-allocate space for predictions
        confidences: list = []
        predictions: list = []

        # Run model with batching
        allocation_index = 0
        for batch_id, batch_data in enumerate(data):
            model_output = self._model(tf.convert_to_tensor(batch_data))

            # Count number of samples in batch to prevent array mismatch
            num_samples_in_batch = len(batch_data)

            # Double array size
            if len(predictions) <= allocation_index:
                predictions += predictions
                if show_confidences:
                    confidences += confidences

            if show_confidences:
                confidences[
                    allocation_index : allocation_index + num_samples_in_batch
                ] = model_output[0].numpy()
            predictions[
                allocation_index : allocation_index + num_samples_in_batch
            ] = model_output[1].numpy()

            allocation_index += num_samples_in_batch

        # Convert predictions, confidences to lists from numpy
        predictions = [predictions[i].tolist() for i in range(allocation_index)]
        if show_confidences:
            confidences = [confidences[i].tolist() for i in range(0, allocation_index)]

        if show_confidences:
            return {"pred": predictions, "conf": confidences}
        return {"pred": predictions}

    def details(self) -> None:
        """
        Print the relevant details of the model.

        Details include summary, parameters, label mapping.
        """
        if not self._model:
            self._construct_model()

        print("\n###### Model Details ######\n")
        self._model.summary()
        print("\nModel Parameters:")
        for key, value in self._parameters.items():
            print(f"{key}: {value}")
        print("\nModel Label Mapping:")
        for key, value in self.label_mapping.items():
            print(f"{key}: {value}")
