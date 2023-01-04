"""Contains functions for classification."""
from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import sklearn.metrics._classification


def convert_confusion_matrix_to_MCM(conf_matrix: list | np.ndarray) -> np.ndarray:
    """
    Convert a confusion matrix into the MCM format.

    Format for precision/recall/fscore/
    support computation by sklearn.

    The format is as specified by sklearn below:
    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.
    Note: this utilizes code/ideology from sklearn.

    :param conf_matrix: confusion matrix, which is a square matrix describing
        false positives and false negatives, true positives and true negatives
        for classification
    :type conf_matrix: Union[list, np.ndarray]
    :return: MCM format for readability by sklearn confusion reports.
    :rtype: np.ndarray
    """
    if not isinstance(conf_matrix, np.ndarray):
        conf_matrix = np.array(conf_matrix)
    num_labels = conf_matrix.shape[0]
    num_samples = np.sum(conf_matrix)
    MCM = np.zeros((num_labels, 2, 2), dtype=np.int64)

    # True Positives
    MCM[:, 1, 1] = np.sum(conf_matrix * np.eye(num_labels), axis=1)

    # False Negatives
    MCM[:, 1, 0] = np.sum(
        conf_matrix * (np.ones(num_labels) - np.eye(num_labels)), axis=1
    )

    # False Positives
    MCM[:, 0, 1] = np.sum(
        conf_matrix.T * (np.ones(num_labels) - np.eye(num_labels)), axis=1
    )

    # True Negatives
    MCM[:, 0, 0] = num_samples - MCM[:, 1, 0] - MCM[:, 0, 1] - MCM[:, 1, 1]

    return MCM


def precision_recall_fscore_support(
    MCM: np.ndarray,
    beta: float = 1.0,
    labels: np.ndarray | None = None,
    pos_label: str | int = 1,
    average: str | None = None,
    warn_for: tuple[str, ...] | set[str] = ("precision", "recall", "f-score"),
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Perform same functionality as recision_recall_fscore_support function.

    Copy of the precision_recall_fscore_support function from sklearn.metrics
    with the update to receiving the MCM instead of calculating each time it is
    called.

    Parameters
    ----------
    MCM : array, shape (n_outputs, 2, 2)
        Multi-classification confusion matrix as referenced by the sklearn
        metrics module. A 2x2 confusion matrix corresponding to each output in
        the input. In multilabel confusion matrix :math:`MCM`, the count of
        true negatives is :math:`MCM_{:,0,0}`, false negatives is
        :math:`MCM_{:,1,0}`, true positives is :math:`MCM_{:,1,1}` and false
        positives is :math:`MCM_{:,0,1}`.

    beta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None (default), 'binary', 'micro', 'macro', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    recall : float (if average is not None) or array of float, , shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined;
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, the metric will be set to 0, as will f-score, and
    ``UndefinedMetricWarning`` will be raised.
    """
    if beta <= 0:
        raise ValueError("beta should be >0 in the F-beta score")

    # ALTERATION: want to still validate average, labels, pos_label, but
    # requires y_true, y_pred, so passed in a mock version `[0]`
    labels = sklearn.metrics._classification._check_set_wise_labels(
        [0], [0], average, labels, pos_label
    )

    # ALTERATION: remove weighted as an option since we don't allow that for
    # passing in the MCM.
    if average == "samples":
        average_options = (None, "micro", "macro", "weighted")
        raise ValueError("average has to be one of " + str(average_options))

    # ALTERATION: Reduce MCM to only labels desired if not all desired.
    if labels is not None:
        MCM = np.take(MCM, labels, axis=0)

    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta**2

    # Divide, and on zero-division, set scores to 0 and warn:

    precision = sklearn.metrics._classification._prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for
    )
    recall = sklearn.metrics._classification._prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for
    )
    # Don't need to warn for F: either P or R warned, or tp == 0 where pos
    # and true are nonzero, in which case, F is well-defined and zero
    denom = beta2 * precision + recall
    denom[denom == 0.0] = 1  # avoid division by 0
    f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            return np.array([0.0]), np.array([0.0]), np.array([0.0]), None
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def classification_report(
    conf_matrix: np.ndarray,
    labels: list | np.ndarray | None = None,
    target_names: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
    digits: int = 2,
    output_dict: bool = False,
) -> str | dict:
    """
    Build a text report showing the main classification metrics.

    Copy of the classification_report function from sklearn.metrics
    with the update to receiving the conf_matrix instead of calculating each
    time it is called.

    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    conf_matrix : array, shape = [n_labels, n_labels]
        confusion matrix, which is a square matrix describing
        false positives and false negatives, true positives and true negatives
        for classification.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    digits : int
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.

    output_dict : bool (default = False)
        If True, return output as dict

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::

            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }

        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), sample average (only for multilabel classification) and
        micro average (averaging the total true positives, false negatives and
        false positives) it is only shown for multi-label or multi-class
        with a subset of classes because it is accuracy otherwise.
        See also:func:`precision_recall_fscore_support` for more details
        on averages.

        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".

    See also
    --------
    precision_recall_fscore_support, confusion_matrix,
    multilabel_confusion_matrix
    """
    # ALTERATION: replaced the _check_targets with this if statement since
    # no y_true, y_pred
    y_type = "multiclass" if conf_matrix.shape[0] > 2 else "binary"

    labels_given = True
    if labels is None:
        # ALTERATION: replaced the label determination of unique_labels
        # since no y_true, y_pred
        labels = np.array(list(range(len(conf_matrix))))
        labels_given = False
    else:
        labels = np.asarray(labels)

    # ALTERATION: replaced the label determination of unique_labels
    # since no y_true, y_pred
    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary") and (
        not labels_given or (set(labels) == set(list(range(len(conf_matrix)))))
    )

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                f"labels size, {len(labels)}, does not match size of "
                f"target_names, {len(target_names)}"
            )
        else:
            raise ValueError(
                f"Number of classes, {len(labels)}, does not match size of "
                f"target_names, {len(target_names)}. Try specifying the labels "
                "parameter"
            )
    if target_names is None:
        target_names = ["%s" % label for label in labels]

    headers = ["precision", "recall", "f1-score", "support"]

    # ALTERATION: instead of passing y_true,y_pred to
    # `precision_recall_fscore_support`, create the MCM from the confusion
    # matrix and pass it instead.
    MCM = convert_confusion_matrix_to_MCM(conf_matrix)

    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(
        MCM, labels=labels, average=None, sample_weight=sample_weight
    )
    rows = zip(target_names, p, r, f1, cast(np.ndarray, s))

    if y_type.startswith("multilabel"):
        average_options: tuple[str, ...] = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    if output_dict:
        report_dict: dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [i.item() for i in scores]))
    else:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        # ALTERATION: instead of passing y_true,y_pred to
        # `precision_recall_fscore_support`, create the MCM from the confusion
        # matrix and pass it instead.
        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            MCM, labels=labels, average=average, sample_weight=sample_weight
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(cast(np.ndarray, s))]

        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [i.item() for i in avg]))
        else:
            if line_heading == "accuracy":
                row_fmt_accuracy = (
                    "{:>{width}s} "
                    + " {:>9.{digits}}" * 2
                    + " {:>9.{digits}f}"
                    + " {:>9}\n"
                )
                report += row_fmt_accuracy.format(
                    line_heading, "", "", *avg[2:], width=width, digits=digits
                )
            else:
                report += row_fmt.format(line_heading, *avg, width=width, digits=digits)

    if output_dict:
        if "accuracy" in report_dict.keys():
            report_dict["accuracy"] = report_dict["accuracy"]["precision"]
        return report_dict
    else:
        return report
