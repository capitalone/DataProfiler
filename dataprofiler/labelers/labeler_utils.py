"""Contains functions for the data labeler."""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, Dict, cast

import numpy as np
import scipy
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning

from .. import dp_logging
from .classification_report_utils import classification_report

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = dp_logging.get_child_logger(__name__)


def f1_report_dict_to_str(f1_report: dict, label_names: list[str]) -> str:
    """
    Return the report string from the f1_report dict.

    Example Output:
                      precision    recall  f1-score   support

         class 0       0.00      0.00      0.00         1
         class 1       1.00      0.67      0.80         3

       micro avg       0.67      0.50      0.57         4
       macro avg       0.50      0.33      0.40         4
    weighted avg       0.75      0.50      0.60         4

    Note: this is generally taken from the `classification_report` function
    inside sklearn.
    :param f1_report: f1 report dictionary from sklearn
    :type f1_report: dict
    :param label_names: names of labels included in the report
    :type label_names: list(str)
    :return: string representing f1_report printout
    :rtype: str
    """
    sig_figs = 2
    headers = ["precision", "recall", "f1-score", "support"]

    longest_last_line_heading = "weighted avg"
    name_width = max(len(name) for name in label_names)
    width = max(name_width, len(longest_last_line_heading), sig_figs)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    report_end = "\n"
    row_fmt = (
        "{:>{width}s} "
        + (" {{{}:>9.{{sig_figs}}f}}" * 3).format(*headers[:-1])
        + " {support:>9}\n"
    )
    for key, row in f1_report.items():
        if key not in ["accuracy", "macro avg", "weighted avg", "micro avg"]:
            report += row_fmt.format(key, **row, width=width, sig_figs=sig_figs)
        else:
            if key == "accuracy":
                row_fmt_accuracy = (
                    "{:>{width}s} "
                    + " {:>9.{sig_figs}}" * 2
                    + " {:>9.{sig_figs}f}"
                    + " {:>9}\n"
                )
                report_end += row_fmt_accuracy.format(
                    key, "", "", row, "", width=width, sig_figs=sig_figs
                )
            else:
                report_end += row_fmt.format(key, **row, width=width, sig_figs=sig_figs)
    report += report_end
    return report


def evaluate_accuracy(
    predicted_entities_in_index: list[list[int]],
    true_entities_in_index: list[list[int]],
    num_labels: int,
    entity_rev_dict: dict[int, str],
    verbose: bool = True,
    omitted_labels: tuple[str, ...] = ("PAD", "UNKNOWN"),
    confusion_matrix_file: str | None = None,
) -> tuple[float, dict]:
    """
    Evaluate accuracy from comparing predicted labels with true labels.

    :param predicted_entities_in_index: predicted encoded labels for input
        sentences
    :type predicted_entities_in_index: list(array(int))
    :param true_entities_in_index: true encoded labels for input sentences
    :type true_entities_in_index: list(array(int))
    :param entity_rev_dict: dictionary to convert indices to entities
    :type entity_rev_dict: dict([index, entity])
    :param verbose: print additional information for debugging
    :type verbose: boolean
    :param omitted_labels: labels to omit from the accuracy evaluation
    :type omitted_labels: list() of text labels
    :param confusion_matrix_file: File name (and dir) for confusion matrix
    :type confusion_matrix_file: str
    :return : f1-score
    :rtype: float
    """
    label_names = None
    label_indexes = None
    if entity_rev_dict:
        label_names = [
            str(x[1])
            for x in sorted(entity_rev_dict.items(), key=lambda x: x[0])
            if x[1] not in omitted_labels
        ]
        label_indexes = [
            x[0]
            for x in sorted(entity_rev_dict.items(), key=lambda x: x[0])
            if x[1] not in omitted_labels
        ]

    max_len = len(predicted_entities_in_index[0])
    true_labels_padded = np.zeros((len(true_entities_in_index), max_len))
    for i, true_labels_row in enumerate(true_entities_in_index):
        true_labels_padded[i][: len(true_labels_row)] = true_labels_row

    true_labels_flatten = np.hstack(true_labels_padded)  # type: ignore
    predicted_labels_flatten = np.hstack(predicted_entities_in_index)

    all_labels: list[str] = []
    if entity_rev_dict:
        all_labels = [entity_rev_dict[key] for key in sorted(entity_rev_dict.keys())]

    # From sklearn, description of the confusion matrix:
    # By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    # is equal to the number of observations known to be in group :math:`i` but
    # predicted to be in group :math:`j`.
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.int64)
    batch_size = min(2**20, len(true_labels_flatten))
    for batch_ind in range(len(true_labels_flatten) // batch_size + 1):
        true_label_batch = true_labels_flatten[
            batch_size * batch_ind : (batch_ind + 1) * batch_size
        ]
        pred_label_batch = predicted_labels_flatten[
            batch_size * batch_ind : (batch_ind + 1) * batch_size
        ]
        conf_mat += scipy.sparse.coo_matrix(
            (np.ones((len(pred_label_batch),)), (true_label_batch, pred_label_batch)),
            shape=(num_labels, num_labels),
            dtype=np.int64,
        ).toarray()

    # Only write confusion matrix if file exists
    if confusion_matrix_file and entity_rev_dict:
        import pandas as pd

        conf_mat_pd = pd.DataFrame(
            conf_mat,
            columns=list(map(lambda x: "pred:" + x, all_labels)),
            index=list(map(lambda x: "true:" + x, all_labels)),
        )

        # Make directory, if required
        if os.path.dirname(confusion_matrix_file) and not os.path.isdir(
            os.path.dirname(confusion_matrix_file)
        ):
            os.makedirs(os.path.dirname(confusion_matrix_file))

        conf_mat_pd.to_csv(confusion_matrix_file)

    f1_report: dict = cast(
        Dict,
        classification_report(
            conf_mat, labels=label_indexes, target_names=label_names, output_dict=True
        ),
    )

    # adjust macro average to be updated only on positive support labels
    # note: in sklearn, support is number of occurrences of each label in
    # true_labels_flatten
    num_labels_with_positive_support = 0
    for key, values in f1_report.items():
        if key not in ["accuracy", "macro avg", "weighted avg", "micro avg"]:
            if values["support"]:
                num_labels_with_positive_support += 1

    # bc sklearn does not remove 0.0 f1 score for 0 support in macro avg.
    for metric in f1_report["macro avg"].keys():
        if metric != "support":
            if not num_labels_with_positive_support:
                f1_report["macro avg"][metric] = np.nan
            else:
                if not label_names:
                    f1_report["macro avg"][metric] = 0
                else:
                    f1_report["macro avg"][metric] *= (
                        float(len(label_names)) / num_labels_with_positive_support
                    )

    if "macro avg" in f1_report:
        f1: float = f1_report["macro avg"]["f1-score"]  # this is micro for the report
    else:
        # this is the only remaining option for the report
        f1 = f1_report["accuracy"]

    if verbose:
        if not label_names:
            label_names = [""]

        f1_report_str = f1_report_dict_to_str(f1_report, label_names)
        logger.info(f"(After removing non-entity tokens)\n{f1_report_str}")
        logger.info(f"F1 Score: {f1}")

    return f1, f1_report


def get_tf_layer_index_from_name(model: tf.keras.Model, layer_name: str) -> int | None:
    """
    Return the index of the layer given the layer name within a tf model.

    :param model: tf keras model to search
    :param layer_name: name of the layer to find
    :return: layer index if it exists or None
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return idx
    return None


def hide_tf_logger_warnings() -> None:
    """Filter out a set of warnings from the tf logger."""

    class NoV1ResourceMessageFilter(logging.Filter):
        """Removes TF2 warning for using TF1 model which has resources."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Remove warning."""
            msg = (
                "is a problem, consider rebuilding the SavedModel after "
                + "running tf.compat.v1.enable_resource_variables()"
            )
            return msg not in record.getMessage()

    tf_logger = logging.getLogger("tensorflow")
    tf_logger.addFilter(NoV1ResourceMessageFilter())


def protected_register_keras_serializable(
    package: str = "Custom", name: str | None = None
) -> Callable:
    """
    Protect against already registered keras serializable layers.

    Ensures that if it was already registered, it will not try to
    register it again.
    """

    def decorator(arg: Any) -> Any:
        """Protect against double registration of a keras layer."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + ">" + class_name
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
    F_{\beta} = (1 + \beta^2) * \frac{\textrm{precision} *
    \textrm{precision}}{(\beta^2 \cdot \textrm{precision}) + \textrm{recall}}
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
    """  # NOQA: E501

    # Modification: remove the run-time type checking for functions
    def __init__(
        self,
        num_classes: int,
        average: str | None = None,
        beta: float = 1.0,
        threshold: float | None = None,
        name: str = "fbeta_score",
        dtype: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FBetaScore class."""
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

        def _zero_wt_init(name: str) -> tf.Variable:
            return self.add_weight(
                name=name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Update state."""
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val: tf.Tensor, sample_weight: tf.Tensor | None) -> tf.Tensor:
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

    def result(self) -> tf.Tensor:
        """Return f1 score."""
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

    def get_config(self) -> dict:
        """Return the serializable config of the metric."""
        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}


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
    F_1 = 2 \cdot \frac{\textrm{precision}
    \cdot \textrm{recall}}{\textrm{precision} + \textrm{recall}}
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
    def __init__(
        self,
        num_classes: int,
        average: str | None = None,
        threshold: float | None = None,
        name: str = "f1_score",
        dtype: str | None = None,
    ) -> None:
        """Initialize F1Score object."""
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def get_config(self) -> dict:
        """Get configuration."""
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
