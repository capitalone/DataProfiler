"""Contains classes for char data labeling."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn import decomposition

from dataprofiler._typing import DataArray

from .. import dp_logging
from . import labeler_utils
from .base_model import AutoSubRegistrationMeta, BaseModel, BaseTrainableModel

_file_dir = os.path.dirname(os.path.abspath(__file__))

logger = dp_logging.get_child_logger(__name__)
labeler_utils.hide_tf_logger_warnings()


def build_embd_dictionary(filename: str) -> Dict[str, np.ndarray]:
    """
    Return a numpy embedding dictionary from embed file with GloVe-like format.

    :param filename: Path to the embed file for loading
    :type filename: str
    """
    embd_table = dict()
    with open(filename, "r") as embds:
        for line in embds:
            line = line.strip().split()  # type: ignore[assignment]
            embd_table[line[0]] = np.asarray(line[1:])

    return embd_table


def create_glove_char(n_dims: int, source_file: str = None) -> None:
    """
    Embed GloVe chars embeddings from source file to n_dims principal components.

    Embed in a new file.

    :param n_dims: Final number of principal component dims of the embeddings
    :type n_dims: int
    :param source_file: Location of original embeddings to factor down
    :type source_file: str
    """
    if source_file is None:
        source_file = os.path.join(_file_dir, "embeddings/glove.840B.300d-char.txt")
    # get embedding table first and vectors as array
    embd_table = build_embd_dictionary(source_file)
    embd_words: List[str]
    embd_matrix: List[np.ndarray]
    embd_words, embd_matrix = [  # type: ignore
        np.asarray(ls) if i > 0 else list(ls)  # type: ignore
        for i, ls in enumerate(zip(*embd_table.items()))
    ]

    # get PCA embedder
    pca = decomposition.PCA(n_components=n_dims)
    reduced_embds: np.ndarray = pca.fit_transform(embd_matrix)

    # write to file
    dir_name = os.path.dirname(source_file)
    embd_file_name = os.path.join(dir_name, "glove-reduced-{}D.txt".format(n_dims))
    with open(embd_file_name, "w") as file:
        for word, embd in zip(embd_words, reduced_embds):
            file.write(word + " " + " ".join(str(num) for num in embd) + "\n")


class CharacterLevelCnnModel(BaseTrainableModel, metaclass=AutoSubRegistrationMeta):
    """Class for training char data labeler."""

    # boolean if the label mapping requires the mapping for index 0 reserved
    requires_zero_mapping: bool = True

    def __init__(self, label_mapping: Dict[str, int], parameters: Dict = None) -> None:
        """
        Initialize CNN Model.

        Initialize epoch_id.

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
        parameters.setdefault("max_length", 3400)
        parameters.setdefault("max_char_encoding_id", 127)
        parameters.setdefault("dim_embed", 64)
        parameters.setdefault("size_fc", [96, 96])
        parameters.setdefault("dropout", 0.073)
        parameters.setdefault("size_conv", 13)
        parameters.setdefault("default_label", "UNKNOWN")
        parameters.setdefault("num_fil", [48 for _ in range(4)])
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

    def _validate_parameters(self, parameters: Dict) -> None:
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
        list_of_necessary_params = [
            "max_length",
            "max_char_encoding_id",
            "dim_embed",
            "size_fc",
            "dropout",
            "size_conv",
            "default_label",
            "pad_label",
            "num_fil",
        ]
        # Make sure the necessary parameters are present and valid.
        for param in parameters:
            if param in [
                "max_length",
                "max_char_encoding_id",
                "dim_embed",
                "size_conv",
            ]:
                if (
                    not isinstance(parameters[param], (int, float))
                    or parameters[param] < 0
                ):
                    errors.append(
                        param + " must be a valid integer or float " "greater than 0."
                    )
            elif param == "dropout":
                if (
                    not isinstance(parameters[param], (int, float))
                    or parameters[param] < 0
                    or parameters[param] > 1
                ):
                    errors.append(
                        param + " must be a valid integer or float " "from 0 to 1."
                    )
            elif param == "size_fc" or param == "num_fil":
                if (
                    not isinstance(parameters[param], list)
                    or len(parameters[param]) == 0
                ):
                    errors.append(param + " must be a non-empty list of " "integers.")
                else:
                    for item in parameters[param]:
                        if not isinstance(item, int):
                            errors.append(
                                param + " must be a non-empty " "list of integers."
                            )
                            break
            elif param == "default_label":
                if not isinstance(parameters[param], str):
                    error = str(param) + " must be a string."
                    errors.append(error)

        # Error if there are extra parameters thrown in
        for param in parameters:
            if param not in list_of_necessary_params:
                errors.append(param + " is not an accepted parameter.")
        if errors:
            raise ValueError("\n".join(errors))

    def set_label_mapping(
        self, label_mapping: Union[List[str], Dict[str, int]]
    ) -> None:
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
        if (
            isinstance(label_mapping, dict) and label_mapping.get("PAD", None) != 0
        ):  # dict with bad PAD
            raise ValueError("`PAD` must map to index zero.")
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
        with open(model_param_dirpath, "w") as fp:
            json.dump(self._parameters, fp)
        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath, "w") as fp:
            json.dump(self.label_mapping, fp)
        self._model.save(os.path.join(dirpath))

    @classmethod
    def load_from_disk(cls, dirpath: str) -> CharacterLevelCnnModel:
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

        # use f1 score metric
        custom_objects = {
            "F1Score": labeler_utils.F1Score(
                num_classes=max(label_mapping.values()) + 1, average="micro"
            ),
            "CharacterLevelCnnModel": cls,
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            tf_model = tf.keras.models.load_model(dirpath)

        loaded_model = cls(label_mapping, parameters)
        loaded_model._model = tf_model

        # Tensorflow v1 Model weights need to be transferred.
        if not callable(tf_model):
            loaded_model._construct_model()
            tf1_weights = []
            for var in tf_model.variables:
                if "training" not in var.name:
                    tf1_weights.append(var.value())

            loaded_model._construct_model()
            tf1_weights.append(loaded_model._model.weights[-1].value())
            loaded_model._model.set_weights(tf1_weights)

        # load self
        loaded_model._model_num_labels = loaded_model.num_labels
        loaded_model._model_default_ind = loaded_model.label_mapping[
            loaded_model._parameters["default_label"]
        ]
        return loaded_model

    @staticmethod
    def _char_encoding_layer(
        input_str_tensor: tf.Tensor, max_char_encoding_id: int, max_len: int
    ) -> tf.Tensor:
        """
        Encode characters for the list of sentences.

        :param input_str_tensor: input list of sentences converted to tensor
        :type input_str_tensor: tf.tensor
        :param max_char_encoding_id: Maximum integer value for encoding the
            input
        :type max_char_encoding_id: int
        :param max_len: Maximum char length in a sample
        :type max_len: int
        :return : tensor containing encoded list of input sentences
        :rtype: tf.Tensor
        """
        # convert characters to indices
        input_str_flatten = tf.reshape(input_str_tensor, [-1])
        sentences_encode = tf.strings.unicode_decode(
            input_str_flatten, input_encoding="UTF-8"
        )
        sentences_encode = tf.add(tf.cast(1, tf.int32), sentences_encode)
        sentences_encode = tf.math.minimum(sentences_encode, max_char_encoding_id + 1)

        # padding
        sentences_encode_pad = sentences_encode.to_tensor(shape=[None, max_len])
        return sentences_encode_pad

    @staticmethod
    def _argmax_threshold_layer(
        num_labels: int, threshold: float = 0.0, default_ind: int = 1
    ) -> tf.keras.layers.Layer:
        """
        Add an argmax threshold layer to the model.

        This layer's output will be the argmax value if the
        confidence for that argmax meets the threshold
        for its label, otherwise it will be the default label index.

        :param num_labels: number of entities
        :type num_labels: int
        :param threshold: default set to 0 so all confidences pass.
        :type threshold: float
        :param default_ind: default index
        :type default_ind: int
        :return: final argmax threshold layer for the model
        """
        # Initialize the thresholds vector variable and create the threshold
        # matrix.
        class ThreshArgMaxLayer(tf.keras.layers.Layer):
            def __init__(self, threshold_: float, num_labels_: int) -> None:
                super(ThreshArgMaxLayer, self).__init__()
                thresh_init = tf.constant_initializer(threshold_)
                self.thresh_vec = tf.Variable(
                    name="ThreshVec",
                    initial_value=thresh_init(shape=[num_labels_]),
                    trainable=False,
                )

            def call(
                self, argmax_layer: tf.Tensor, confidence_layer: tf.Tensor
            ) -> tf.Tensor:
                threshold_at_argmax = tf.gather(self.thresh_vec, argmax_layer)

                confidence_max_layer = tf.keras.backend.max(confidence_layer, axis=2)

                # Check if the confidences meet the threshold minimum.
                argmax_mask = tf.keras.backend.cast(
                    tf.keras.backend.greater_equal(
                        confidence_max_layer, threshold_at_argmax
                    ),
                    dtype=argmax_layer.dtype,
                )

                # Create a vector the same size as the batch_size which
                # represents the background label
                bg_label_tf = tf.keras.backend.constant(
                    default_ind, dtype=argmax_layer.dtype
                )

                # Generate the final predicted output using the function:
                final_predicted_layer = tf.add(
                    bg_label_tf,
                    tf.multiply(tf.subtract(argmax_layer, bg_label_tf), argmax_mask),
                    name="ThreshArgMax",
                )

                return final_predicted_layer

        return ThreshArgMaxLayer(threshold, num_labels)

    def _construct_model(self) -> None:
        """
        Construct model for the data labeler.

        This also serves as a weight reset.

        :return: None
        """
        num_labels = self.num_labels
        default_ind = self.label_mapping[self._parameters["default_label"]]

        # Reset model
        tf.keras.backend.clear_session()

        # generate glove embedding
        create_glove_char(self._parameters["dim_embed"])

        # generate model
        self._model = tf.keras.models.Sequential()

        # default parameters
        max_length = self._parameters["max_length"]
        max_char_encoding_id = self._parameters["max_char_encoding_id"]

        # Encoding layer
        def encoding_function(input_str: tf.Tensor) -> tf.Tensor:
            char_in_vector = CharacterLevelCnnModel._char_encoding_layer(
                input_str, max_char_encoding_id, max_length
            )
            return char_in_vector

        self._model.add(tf.keras.layers.Input(shape=(None,), dtype=tf.string))

        self._model.add(
            tf.keras.layers.Lambda(encoding_function, output_shape=tuple([max_length]))
        )

        # Create a pre-trained weight matrix
        # character encoding indices range from 0 to max_char_encoding_id,
        # we add one extra index for out-of-vocabulary character
        embed_file = os.path.join(
            _file_dir,
            "embeddings/glove-reduced-{}D.txt".format(self._parameters["dim_embed"]),
        )
        embedding_matrix = np.zeros(
            (max_char_encoding_id + 2, self._parameters["dim_embed"])
        )
        embedding_dict = build_embd_dictionary(embed_file)

        input_shape = tuple([max_length])
        # Fill in the weight matrix: let pad and space be 0s
        for ascii_num in range(max_char_encoding_id):
            if chr(ascii_num) in embedding_dict:
                embedding_matrix[ascii_num + 1] = embedding_dict[chr(ascii_num)]

        self._model.add(
            tf.keras.layers.Embedding(
                max_char_encoding_id + 2,
                self._parameters["dim_embed"],
                weights=[embedding_matrix],
                input_length=input_shape[0],
                trainable=True,
            )
        )

        # Add the convolutional layers
        for fil in self._parameters["num_fil"]:
            self._model.add(
                tf.keras.layers.Conv1D(
                    filters=fil,
                    kernel_size=self._parameters["size_conv"],
                    activation="relu",
                    padding="same",
                )
            )
            if self._parameters["dropout"]:
                self._model.add(tf.keras.layers.Dropout(self._parameters["dropout"]))
            # Add batch normalization, set fused = True for compactness
            self._model.add(tf.keras.layers.BatchNormalization(fused=False, scale=True))

        # Add the fully connected layers
        for size in self._parameters["size_fc"]:
            self._model.add(tf.keras.layers.Dense(units=size, activation="relu"))
            if self._parameters["dropout"]:
                self._model.add(tf.keras.layers.Dropout(self._parameters["dropout"]))

        # Add the final Softmax layer
        self._model.add(tf.keras.layers.Dense(num_labels, activation="softmax"))

        # Output the model into a .pb file for TensorFlow
        argmax_layer = tf.keras.backend.argmax(self._model.output)

        # Create confidence layers
        final_predicted_layer = CharacterLevelCnnModel._argmax_threshold_layer(
            num_labels, threshold=0.0, default_ind=default_ind
        )

        argmax_outputs = self._model.outputs + [
            argmax_layer,
            final_predicted_layer(argmax_layer, self._model.output),
        ]
        self._model = tf.keras.Model(self._model.inputs, argmax_outputs)

        # Compile the model
        softmax_output_layer_name = self._model.outputs[0].name.split("/")[0]
        losses = {softmax_output_layer_name: "categorical_crossentropy"}

        # use f1 score metric
        f1_score_training = labeler_utils.F1Score(
            num_classes=num_labels, average="micro"
        )
        metrics = {softmax_output_layer_name: ["acc", f1_score_training]}

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
        Reconstruct appropriate layers if number of labels is altered.

        :return: None
        """
        # Reset model
        tf.keras.backend.clear_session()

        num_labels = self.num_labels
        default_ind = self.label_mapping[self._parameters["default_label"]]

        # Remove the 3 output layers (dense_2', 'tf_op_layer_ArgMax',
        #                             'thresh_arg_max_layer')
        for _ in range(3):
            self._model.layers.pop()

        # Add the final Softmax layer to the previous spot
        final_softmax_layer = tf.keras.layers.Dense(
            num_labels, activation="softmax", name="dense_2"
        )(self._model.layers[-4].output)

        # Output the model into a .pb file for TensorFlow
        argmax_layer = tf.keras.backend.argmax(final_softmax_layer)

        # Create confidence layers
        final_predicted_layer = CharacterLevelCnnModel._argmax_threshold_layer(
            num_labels, threshold=0.0, default_ind=default_ind
        )

        argmax_outputs = [final_softmax_layer] + [
            argmax_layer,
            final_predicted_layer(argmax_layer, final_softmax_layer),
        ]
        self._model = tf.keras.Model(self._model.inputs, argmax_outputs)

        # Compile the model
        softmax_output_layer_name = self._model.outputs[0].name.split("/")[0]
        losses = {softmax_output_layer_name: "categorical_crossentropy"}

        # use f1 score metric
        f1_score_training = labeler_utils.F1Score(
            num_classes=num_labels, average="micro"
        )
        metrics = {softmax_output_layer_name: ["acc", f1_score_training]}

        self._model.compile(loss=losses, optimizer="adam", metrics=metrics)
        self._epoch_id = 0
        self._model_num_labels = num_labels
        self._model_default_ind = default_ind

    def fit(
        self,
        train_data: DataArray,
        val_data: Optional[DataArray] = None,
        batch_size: int = None,
        epochs: int = None,
        label_mapping: Dict[str, int] = None,
        reset_weights: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict, Optional[float], Dict]:
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

        history: Dict = defaultdict()
        f1: Optional[float] = None
        f1_report: Dict = {}

        self._model.reset_metrics()
        softmax_output_layer_name = self._model.outputs[0].name.split("/")[0]

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
    ) -> Union[Tuple[float, Dict], Tuple[None, None]]:
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
    ) -> Dict:
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
            raise ValueError(
                "You are trying to predict without a model. "
                "Construct/Load a model before predicting."
            )
        elif self._need_to_reconstruct_model():
            raise RuntimeError(
                "The model label mapping definitions have been "
                "altered without additional training. Please "
                "train the model or reset the label mapping to "
                "predict."
            )
        # Pre-allocate space for predictions
        confidences: Union[List, np.ndarray] = []
        sentence_lengths = np.zeros((batch_size,), dtype=int)
        predictions = np.zeros((batch_size, self._parameters["max_length"]))
        if show_confidences:
            confidences = np.zeros(
                (batch_size, self._parameters["max_length"], self.num_labels)
            )

        # Run model with batching
        allocation_index = 0
        for batch_id, batch_data in enumerate(data):
            model_output = self._model(tf.convert_to_tensor(batch_data))

            # Count number of samples in batch to prevent array mismatch
            num_samples_in_batch = len(batch_data)
            allocation_index = batch_id * batch_size

            # Double array size
            if len(predictions) <= allocation_index:
                predictions = np.pad(
                    predictions, ((0, len(predictions)), (0, 0)), mode="constant"
                )
                sentence_lengths = np.pad(
                    sentence_lengths,
                    pad_width=((0, len(sentence_lengths)),),
                    mode="constant",
                )
                if show_confidences:
                    confidences = np.pad(
                        confidences,
                        ((0, len(predictions)), (0, 0), (0, 0)),
                        mode="constant",
                    )

            if show_confidences:
                confidences[
                    allocation_index : allocation_index + num_samples_in_batch
                ] = model_output[0].numpy()
            predictions[
                allocation_index : allocation_index + num_samples_in_batch
            ] = model_output[1].numpy()
            sentence_lengths[
                allocation_index : allocation_index + num_samples_in_batch
            ] = list(map(lambda x: len(x[0]), batch_data))

            allocation_index += num_samples_in_batch

        # Convert predictions, confidences to lists from numpy
        predictions_list: List = [i for i in range(0, allocation_index)]
        confidences_list: List
        if show_confidences:
            confidences_list = [i for i in range(0, allocation_index)]

        # Append slices of predictions to return prediction & confidence matrices
        for index, sentence_length in enumerate(sentence_lengths[:allocation_index]):
            predictions_list[index] = list(predictions[index][:sentence_length])
            if show_confidences:
                confidences_list[index] = list(confidences[index][:sentence_length])

        if show_confidences:
            return {"pred": predictions_list, "conf": confidences_list}
        return {"pred": predictions_list}

    def details(self) -> None:
        """
        Print the relevant details of the model.

        Details include summary, parameters, and label mapping.
        """
        print("\n###### Model Details ######\n")
        self._model.summary()
        print("\nModel Parameters:")
        for key, value in self._parameters.items():
            print("{}: {}".format(key, value))
        print("\nModel Label Mapping:")
        for key, value in self.label_mapping.items():
            print("{}: {}".format(key, value))
