import json
import copy
import os
import sys
import time
import logging
from collections import defaultdict
import functools

import tensorflow as tf
import numpy as np
from sklearn import decomposition

from . import labeler_utils
from .base_model import BaseModel, BaseTrainableModel
from .base_model import AutoSubRegistrationMeta
from .. import dp_logging

_file_dir = os.path.dirname(os.path.abspath(__file__))

logger = dp_logging.get_child_logger(__name__)


class NoV1ResourceMessageFilter(logging.Filter):
    """Removes TF2 warning for using TF1 model which has resources."""
    def filter(self, record):
        msg = 'is a problem, consider rebuilding the SavedModel after ' + \
            'running tf.compat.v1.enable_resource_variables()'
        return msg not in record.getMessage()


tf_logger = logging.getLogger('tensorflow')
tf_logger.addFilter(NoV1ResourceMessageFilter())


def protected_register_keras_serializable(package='Custom', name=None):
    """
    Protects against already registered keras serializable layers. This
    ensures that if it was already registered, it will not try to register it
    again.
    """
    def decorator(arg):
        """Protects against double registration of a keras layer."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + '>' + class_name
        if tf.keras.utils.get_registered_object(registered_name) is None:
            tf.keras.utils.register_keras_serializable(package, name)(arg)
        return arg
    return decorator


@protected_register_keras_serializable()
class FBetaScore(tf.keras.metrics.Metric):
    r"""Computes F-Beta score.
    Adapted and slightly modified from https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/metrics/f_scores.py#L211-L283

    # Copyright 2019 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #     https://github.com/tensorflow/addons/blob/v0.12.0/LICENSE
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    It is the weighted harmonic mean of precision
    and recall. Output range is `[0, 1]`. Works for
    both multi-class and multi-label classification.
    $$
    F_{\beta} = (1 + \beta^2) * \frac{\textrm{precision} * \textrm{precision}}{(\beta^2 \cdot \textrm{precision}) + \textrm{recall}}
    $$
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    Returns:
        F-Beta Score: float.
    """

    # Modification: remove the run-time type checking for functions
    def __init__(self, num_classes, average=None, beta=1.0, threshold=None,
                 name="fbeta_score", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, 'micro', 'macro', 'weighted']"
            )

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError("The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
        tf.keras.backend.batch_set_value([(v, reset_value) for v in self.variables])


@protected_register_keras_serializable()
class F1Score(FBetaScore):
    r"""Computes F-1 Score.

    # Copyright 2019 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #     https://github.com/tensorflow/addons/blob/v0.12.0/LICENSE
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    It is the harmonic mean of precision and recall.
    Output range is `[0, 1]`. Works for both multi-class
    and multi-label classification.
    $$
    F_1 = 2 \cdot \frac{\textrm{precision} \cdot \textrm{recall}}{\textrm{precision} + \textrm{recall}}
    $$
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    Returns:
        F-1 Score: float.
    """

    # Modification: remove the run-time type checking for functions
    def __init__(self, num_classes, average=None, threshold=None,
                 name="f1_score", dtype=None):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


class PreEncodedCharCnnModel(BaseTrainableModel,
                             metaclass=AutoSubRegistrationMeta):

    # boolean if the label mapping requires the mapping for index 0 reserved
    requires_zero_mapping = True

    def __init__(self, label_mapping=None, parameters=None):
        """
        CNN Model Initializer. initialize epoch_id

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
        parameters.setdefault('max_length', 1014)
        parameters.setdefault('alphabet_size', 69)
        parameters.setdefault('dim_embed', 32)
        parameters.setdefault('conv_layers', [
          [256, 7,  1],
          [256, 7,  1],
          [256, 3, -1],
          [256, 3, -1],
          [256, 3, -1],
          [256, 3,  1]
        ])
        parameters.setdefault('size_fc', [512, 512])
        parameters.setdefault('dropout', 0.5)
        parameters.setdefault('threshold', 1e-6)
        parameters.setdefault('default_label', "UNKNOWN")
        parameters['pad_label'] = 'PAD'
        self._epoch_id = 0

        # reconstruct flags for model
        self._model_num_labels = 0
        self._model_default_ind = -1

        BaseModel.__init__(self, label_mapping, parameters)

    def __eq__(self, other):
        """
        Checks if two models are equal with one another, may only check
        important variables, i.e. may not check model itself.

        :param self: a model
        :param other: a model
        :type self: BaseModel
        :type other: BaseModel
        :return: Whether or not self and other are equal
        :rtype: bool
        """
        if self._parameters != other._parameters \
                or self._label_mapping != other._label_mapping:
            return False
        return True

    def _validate_parameters(self, parameters):
        """
        Validate the parameters sent in. Raise error if invalid parameters are
        present.

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
        list_of_necessary_params = ['max_length', 'alphabet_size',
                                    'dim_embed', 'size_fc', 'dropout',
                                    'threshold', 'conv_layers', 'default_label',
                                     'pad_label']
        # Make sure the necessary parameters are present and valid.
        for param in parameters:
            if param in ['max_length', 'alphabet_size', 'dim_embed']:
                if not isinstance(parameters[param], (int, float)) \
                        or parameters[param] < 0:
                    errors.append(param + " must be a valid integer or float "
                                          "greater than 0.")
            elif param in ['dropout', 'threshold']:
                if not isinstance(parameters[param], (int, float)) \
                        or parameters[param] < 0 or parameters[param] > 1:
                    errors.append(param + " must be a valid integer or float "
                                          "from 0 to 1.")
            elif param == 'size_fc':
                if not isinstance(parameters[param], list) \
                        or len(parameters[param]) == 0:
                    errors.append(param + " must be a non-empty list of "
                                          "integers.")
                else:
                    for item in parameters[param]:
                        if not isinstance(item, int):
                            errors.append(param + " must be a non-empty "
                                                  "list of integers.")
                            break
            elif param == 'conv_layers':
                is_bad_conv_layers = True
                if isinstance(parameters[param], list):
                    is_bad_conv_layers = False
                    for layer in parameters[param]:
                        if (not isinstance(layer, list) or len(layer) != 3
                                or any([not isinstance(x, int) for x in  layer])):
                            is_bad_conv_layers = True
                if is_bad_conv_layers:
                    errors.append(param + " must be a non-empty list of "
                                          "tuples containing 3 integers.")
            elif param == 'default_label':
                if not isinstance(parameters[param], str):
                    error = str(param) + " must be a string."
                    errors.append(error)

        # Error if there are extra parameters thrown in
        for param in parameters:
            if param not in list_of_necessary_params:
                errors.append(param + " is not an accepted parameter.")
        if errors:
            raise ValueError('\n'.join(errors))

    def set_label_mapping(self, label_mapping):
        """
        Sets the labels for the model

        :param label_mapping: label mapping of the model
        :type label_mapping: dict
        :return: None
        """
        if not isinstance(label_mapping, (list, dict)):
            raise TypeError("Labels must either be a non-empty encoding dict "
                            "which maps labels to index encodings or a list.")

        label_mapping = copy.deepcopy(label_mapping)
        if 'PAD' not in label_mapping:
            if isinstance(label_mapping, list):  # if list missing PAD
                label_mapping = ['PAD'] + label_mapping
            elif 0 not in label_mapping.values():  # if dict missing PAD and 0
                label_mapping.update({'PAD': 0})
        if (isinstance(label_mapping, dict)
                and label_mapping.get('PAD', None) != 0):  # dict with bad PAD
            raise ValueError("`PAD` must map to index zero.")
        if self._parameters['default_label'] not in label_mapping:
            raise ValueError("The `default_label` of {} must exist in the "
                             "label mapping.".format(
                                self._parameters['default_label']))
        super().set_label_mapping(label_mapping)

    def _need_to_reconstruct_model(self):
        """
        Determines whether or not the model needs to be reconstructed.

        :return: bool of whether or not the model needs to reconstruct.
        """
        if not self._model:
            return False
        default_ind = self.label_mapping[self._parameters['default_label']]
        return self.num_labels != self._model_num_labels or \
            default_ind != self._model_default_ind

    def save_to_disk(self, dirpath):
        """
        Saves whole model to disk with weights

        :param dirpath: directory path where you want to save the model to
        :type dirpath: str
        :return: None
        """
        if not self._model:
            self._construct_model()
        elif self._need_to_reconstruct_model():
            self._reconstruct_model()

        model_param_dirpath = os.path.join(dirpath, "model_parameters.json")
        with open(model_param_dirpath, 'w') as fp:
            json.dump(self._parameters, fp)
        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath, 'w') as fp:
            json.dump(self.label_mapping, fp)
        self._model.save(os.path.join(dirpath))

    @classmethod
    def load_from_disk(cls, dirpath):
        """
        Loads whole model from disk with weights

        :param dirpath: directory path where you want to load the model from
        :type dirpath: str
        :return: None
        """

        # load parameters
        model_param_dirpath = os.path.join(dirpath, "model_parameters.json")
        with open(model_param_dirpath, 'r') as fp:
            parameters = json.load(fp)

        # load label_mapping
        labels_dirpath = os.path.join(dirpath, "label_mapping.json")
        with open(labels_dirpath, 'r') as fp:
            label_mapping = json.load(fp)

        # use f1 score metric
        custom_objects = {
            "F1Score": F1Score(
                num_classes=max(label_mapping.values()) + 1,
                average='micro'),
            "CharacterLevelCnnModel": cls,
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            tf_model = tf.keras.models.load_model(dirpath)

        loaded_model = cls(label_mapping, parameters)
        loaded_model._model = tf_model
        #
        # # Tensorflow v1 Model weights need to be transferred.
        # if not callable(tf_model):
        #     loaded_model._construct_model()
        #     tf1_weights = []
        #     for var in tf_model.variables:
        #         if 'training' not in var.name:
        #             tf1_weights.append(var.value())
        #
        #     loaded_model._construct_model()
        #     tf1_weights.append(loaded_model._model.weights[-1].value())
        #     loaded_model._model.set_weights(tf1_weights)

        # load self
        loaded_model._model_num_labels = loaded_model.num_labels
        loaded_model._model_default_ind = loaded_model.label_mapping[
            loaded_model._parameters['default_label']
        ]
        return loaded_model

    def _construct_model(self):
        """
        Model constructor for the data labeler. This also serves as a weight
        reset.

        :return: None
        """
        num_labels = self.num_labels
        default_ind = self.label_mapping[self._parameters['default_label']]

        # default parameters
        max_length = self._parameters['max_length']
        alphabet_size = self._parameters['alphabet_size']
        dim_embed = self._parameters['dim_embed']
        conv_layers = self._parameters['conv_layers']
        size_fc = self._parameters['size_fc']
        threshold = self._parameters['threshold']
        dropout = self._parameters['dropout']

        # Reset model
        tf.keras.backend.clear_session()

         # Input layer
        inputs = tf.keras.layers.Input(
            shape=(None,), name='sent_input', dtype='int64')
        # Embedding layers
        x_embedding = tf.keras.layers.Embedding(
            alphabet_size + 1, dim_embed, input_length=max_length)(inputs)

        # Convolution layers
        x = x_embedding
        for cl in conv_layers:
            x = tf.keras.layers.Convolution1D(cl[0], cl[1], padding='same')(x)
            x = tf.keras.layers.ThresholdedReLU(threshold)(x)
            if cl[2] != -1:
                x = tf.keras.layers.MaxPooling1D(cl[2])(x)
        # x = tf.keras.layers.Flatten()(x)

        # Fully connected layers
        for fl in size_fc:
            x_dense = tf.keras.layers.Dense(fl)(x)
            x = tf.keras.layers.ThresholdedReLU(threshold)(x_dense)
            x = tf.keras.layers.Dropout(dropout)(x)

        # Output layer
        predictions = tf.keras.layers.Dense(
            num_labels, activation='softmax', name='softmax_output')(x)
        # argmax layer
        argmax_layer = tf.keras.backend.argmax(predictions)

        # Build and compile model
        self._model = tf.keras.models.Model(
            inputs=inputs, outputs=[predictions, argmax_layer])

        # Compile the model w/ metrics
        softmax_output_layer_name = self._model.outputs[0].name.split('/')[0]
        losses = {softmax_output_layer_name: "categorical_crossentropy"}

        # use f1 score metric
        f1_score_training = F1Score(num_classes=num_labels, average='micro')
        metrics = {softmax_output_layer_name: ['acc', f1_score_training]}

        self._model.compile(loss=losses, optimizer="adam", metrics=metrics)

        self._epoch_id = 0
        self._model_num_labels = num_labels
        self._model_default_ind = default_ind

    def reset_weights(self):
        """
        Reset the weights of the model.

        :return: None
        """
        self._construct_model()

    def _reconstruct_model(self):
        """
        Reconstruct the appropriate layers if the number of number of labels is
        altered

        :return: None
        """

        # Reset model
        tf.keras.backend.clear_session()

        num_labels = self.num_labels
        default_ind = self.label_mapping[self._parameters['default_label']]

        # Remove the 2 output layers ('softmax', 'tf_op_layer_ArgMax')
        for _ in range(2):
            self._model.layers.pop()

        # Add the final Softmax layer to the previous spot
        final_softmax_layer = tf.keras.layers.Dense(
            num_labels, activation='softmax', name="softmax_output")(
            self._model.layers[-4].output)

        # Output the model into a .pb file for TensorFlow
        argmax_layer = tf.keras.backend.argmax(final_softmax_layer)


        argmax_outputs = [final_softmax_layer, argmax_layer]
        self._model = tf.keras.Model(self._model.inputs, argmax_outputs)

        # Compile the model
        softmax_output_layer_name = self._model.outputs[0].name.split('/')[0]
        losses = {softmax_output_layer_name: "categorical_crossentropy"}

        # use f1 score metric
        f1_score_training = F1Score(num_classes=num_labels, average='micro')
        metrics = {softmax_output_layer_name: ['acc', f1_score_training]}

        self._model.compile(loss=losses, optimizer="adam", metrics=metrics)

        self._epoch_id = 0
        self._model_num_labels = num_labels
        self._model_default_ind = default_ind

    def fit(self, train_data, val_data=None, batch_size=32, label_mapping=None,
            reset_weights=False, verbose=True):
        """
        Train the current model with the training data and validation data

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
        :return: None
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

        history = defaultdict()
        f1 = None
        f1_report = []

        self._model.reset_metrics()
        softmax_output_layer_name = self._model.outputs[0].name.split('/')[0]

        start_time = time.time()
        batch_id = 0
        for x_train, y_train in train_data:
            model_results = self._model.train_on_batch(
                x_train, {softmax_output_layer_name: y_train})
            sys.stdout.flush()
            if verbose:
                sys.stdout.write(
                    "\rEPOCH %d, batch_id %d: loss: %f - acc: %f - "
                    "f1_score %f" %
                    (self._epoch_id, batch_id, *model_results[1:]))
            batch_id += 1

        for i, metric_label in enumerate(self._model.metrics_names):
            history[metric_label] = model_results[i]

        if val_data:
            f1, f1_report = self._validate_training(val_data)
            history['f1_report'] = f1_report

            val_f1 = f1_report['weighted avg']['f1-score'] \
                if f1_report else np.NAN
            val_precision = f1_report['weighted avg']['precision'] \
                if f1_report else np.NAN
            val_recall = f1_report['weighted avg']['recall'] \
                if f1_report else np.NAN
            epoch_time = time.time() - start_time
            logger.info("\rEPOCH %d (%ds), loss: %f - acc: %f - f1_score %f -- "
                        "val_f1: %f - val_precision: %f - val_recall %f" %
                        (self._epoch_id, epoch_time, *model_results[1:],
                         val_f1, val_precision, val_recall))

        self._epoch_id += 1

        return history, f1, f1_report

    def _validate_training(self, val_data, batch_size_test=32,
                           verbose_log=True, verbose_keras=False):
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
            y_val_pred.append(self._model.predict(
                x_val, batch_size=batch_size_test, verbose=verbose_keras)[1])
            y_val_test.append(np.argmax(y_val, axis=-1))
            batch_id += 1
            sys.stdout.flush()
            if verbose_log:
                sys.stdout.write("\rEPOCH %g, validation_batch_id %d" %
                                 (self._epoch_id, batch_id))

        tf.keras.backend.set_floatx('float32')
        # Clean the predicted entities and the actual entities
        f1, f1_report = labeler_utils.evaluate_accuracy(
            np.concatenate(y_val_pred, axis=0),
            np.concatenate(y_val_test, axis=0),
            self.num_labels,
            self.reverse_label_mapping,
            verbose=verbose_keras)

        return f1, f1_report

    def predict(self, data, batch_size=32, show_confidences=False,
                verbose=True):
        """
        Run model and get predictions

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
            raise ValueError("You are trying to predict without a model. "
                             "Construct/Load a model before predicting.")
        elif self._need_to_reconstruct_model():
            raise RuntimeError("The model label mapping definitions have been "
                               "altered without additional training. Please "
                               "train the model or reset the label mapping to "
                               "predict.")
        # Pre-allocate space for predictions
        confidences = []
        sentence_lengths = np.zeros((batch_size,), dtype=int)
        predictions = np.zeros((batch_size, self._parameters['max_length']))
        if show_confidences:
            confidences = np.zeros((batch_size,
                                    self._parameters['max_length'],
                                    self.num_labels))

        # Run model with batching
        allocation_index = 0
        for batch_id, batch_data in enumerate(data):
            model_output = self._model(
                tf.convert_to_tensor(batch_data)
            )

            # Count number of samples in batch to prevent array mismatch
            num_samples_in_batch = len(batch_data)
            allocation_index = batch_id * batch_size

            # Double array size
            if len(predictions) <= allocation_index:
                predictions = np.pad(predictions, ((0, len(predictions)),
                                                   (0, 0)), mode='constant')
                sentence_lengths = np.pad(
                    sentence_lengths, pad_width=((0, len(sentence_lengths)),),
                    mode='constant')
                if show_confidences:
                    confidences = np.pad(confidences,
                                         ((0, len(predictions)),
                                          (0, 0), (0, 0)), mode='constant')

            if show_confidences:
                confidences[allocation_index:allocation_index + num_samples_in_batch] = model_output[0].numpy()
            predictions[allocation_index:allocation_index + num_samples_in_batch] = model_output[1].numpy()
            sentence_lengths[allocation_index:allocation_index + num_samples_in_batch] = list(map(lambda x: len(x), batch_data))

            allocation_index += num_samples_in_batch

        # Convert predictions, confidences to lists from numpy
        predictions_list = [i for i in range(0, allocation_index)]
        confidences_list = None
        if show_confidences:
            confidences_list = [i for i in range(0, allocation_index)]

        # Append slices of predictions to return prediction & confidence matrices
        for index, sentence_length \
                in enumerate(sentence_lengths[:allocation_index]):
            predictions_list[index] = list(predictions[index][:sentence_length])
            if show_confidences:
                confidences_list[index] = list(confidences[index][:sentence_length])

        if show_confidences:
            return {'pred': predictions_list, 'conf': confidences_list}
        return {'pred': predictions_list}

    def details(self):
        """
        Prints the relevant details of the model (summary, parameters, label
        mapping)
        """
        print("\n###### Model Details ######\n")
        self._model.summary()
        print("\nModel Parameters:")
        for key, value in self._parameters.items():
            print("{}: {}".format(key, value))
        print("\nModel Label Mapping:")
        for key, value in self.label_mapping.items():
            print("{}: {}".format(key, value))
