import os
import warnings

import scipy
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from .classification_report_utils import classification_report
from .. import dp_logging

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = dp_logging.get_child_logger(__name__)


def f1_report_dict_to_str(f1_report, label_names):
    """
    Returns the report string from the f1_report dict.

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

    longest_last_line_heading = 'weighted avg'
    name_width = max(len(name) for name in label_names)
    width = max(name_width, len(longest_last_line_heading), sig_figs)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    report_end = '\n'
    row_fmt = '{:>{width}s} ' + (' {{{}:>9.{{sig_figs}}f}}' * 3).format(
        *headers[:-1]) + ' {support:>9}\n'
    for key, row in f1_report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            report += row_fmt.format(key, **row, width=width, sig_figs=sig_figs)
        else:
            if key == 'accuracy':
                row_fmt_accuracy = '{:>{width}s} ' + \
                                   ' {:>9.{sig_figs}}' * 2 + ' {:>9.{sig_figs}f}' + \
                                   ' {:>9}\n'
                report_end += row_fmt_accuracy.format(key, '', '', row, '',
                                                      width=width, sig_figs=sig_figs)
            else:
                report_end += row_fmt.format(key, **row,
                                             width=width, sig_figs=sig_figs)
    report += report_end
    return report


def evaluate_accuracy(predicted_entities_in_index, true_entities_in_index,
                      num_labels, entity_rev_dict, verbose=True,
                      omitted_labels=('PAD', 'UNKNOWN'),
                      confusion_matrix_file=None):
    """
    Evaluate the accuracy from comparing the predicted labels with true labels

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
        label_names = [str(x[1]) for x in
                       sorted(entity_rev_dict.items(), key=lambda x: x[0]) if
                       x[1] not in omitted_labels]
        label_indexes = [x[0] for x in
                         sorted(entity_rev_dict.items(), key=lambda x: x[0]) if
                         x[1] not in omitted_labels]

    max_len = len(predicted_entities_in_index[0])
    true_labels_padded = np.zeros((len(true_entities_in_index), max_len))
    for i, true_labels_row in enumerate(true_entities_in_index):
        true_labels_padded[i][:len(true_labels_row)] = true_labels_row

    true_labels_flatten = np.hstack(true_labels_padded)
    predicted_labels_flatten = np.hstack(predicted_entities_in_index)

    if entity_rev_dict:
        all_labels = [entity_rev_dict[key] for key in
                      sorted(entity_rev_dict.keys())]

    # From sklearn, description of the confusion matrix:
    # By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    # is equal to the number of observations known to be in group :math:`i` but
    # predicted to be in group :math:`j`.
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.int64)
    batch_size = min(2**20, len(true_labels_flatten))
    for batch_ind in range(len(true_labels_flatten)//batch_size + 1):
        true_label_batch = true_labels_flatten[batch_size*batch_ind:(batch_ind + 1) * batch_size]
        pred_label_batch = predicted_labels_flatten[batch_size * batch_ind:(batch_ind + 1) * batch_size]
        conf_mat += scipy.sparse.coo_matrix(
            (
                np.ones((len(pred_label_batch),)),
                (true_label_batch, pred_label_batch)
            ),
            shape=(num_labels, num_labels),
            dtype=np.int64).toarray()

    # Only write confusion matrix if file exists
    if confusion_matrix_file and entity_rev_dict:
        import pandas as pd
        conf_mat_pd = pd.DataFrame(
            conf_mat,
            columns=list(map(lambda x: 'pred:' + x, all_labels)),
            index=list(map(lambda x: 'true:' + x, all_labels)))
        
        # Make directory, if required
        if os.path.dirname(confusion_matrix_file) \
                and not os.path.isdir(os.path.dirname(confusion_matrix_file)):
            os.makedirs(os.path.dirname(confusion_matrix_file))
            
        conf_mat_pd.to_csv(confusion_matrix_file)

    f1_report = classification_report(
        conf_mat,
        labels=label_indexes,
        target_names=label_names, output_dict=True)

    # adjust macro average to be updated only on positive support labels
    # note: in sklearn, support is number of occurrences of each label in
    # true_labels_flatten
    num_labels_with_positive_support = 0
    for key, values in f1_report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            if values['support']:
                num_labels_with_positive_support += 1

    # bc sklearn does not remove 0.0 f1 score for 0 support in macro avg.
    for metric in f1_report['macro avg'].keys():
        if metric != 'support':
            if not num_labels_with_positive_support:
                f1_report['macro avg'][metric] = np.nan
            else:
                f1_report['macro avg'][metric] *= float(
                    len(label_names)) / num_labels_with_positive_support
            
    if 'macro avg' in f1_report:
        f1 = f1_report['macro avg']['f1-score']  # this is micro for the report
    else:
        # this is the only remaining option for the report
        f1 = f1_report['accuracy']

    if verbose:
        f1_report_str = f1_report_dict_to_str(f1_report, label_names)
        logger.info(f"(After removing non-entity tokens)\n{f1_report_str}")
        logger.info(f"F1 Score: {f1}")
    
    return f1, f1_report
