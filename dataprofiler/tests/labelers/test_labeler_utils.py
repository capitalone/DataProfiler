import logging
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import tensorflow as tf

from dataprofiler.labelers import labeler_utils


class TestEvaluateAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_labels = 3
        cls.reverse_label_mapping = {
            0: "PAD",
            1: "UNKNOWN",
            2: "OTHER",
        }
        cls.y_true = np.array([[0, 1, 2, 2, 2, 0]]).T.tolist()
        cls.y_pred = np.array([[0, 0, 2, 2, 1, 2]]).T.tolist()

    def test_no_omit_class(self):

        expected_output = {
            "PAD": {
                "precision": 1 / 2,
                "recall": 1 / 2,
                "f1-score": 1 / 2,
                "support": 2,
            },
            "UNKNOWN": {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 1,
            },
            "OTHER": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "accuracy": 0.5,
            "macro avg": {
                "precision": (1 / 2 + 2 / 3) / 3,
                "recall": (1 / 2 + 2 / 3) / 3,
                "f1-score": (1 / 2 + 2 / 3) / 3,
                "support": 6,
            },
            "weighted avg": {
                "precision": 1 / 2,
                "recall": 1 / 2,
                "f1-score": 1 / 2,
                "support": 6,
            },
        }

        f1, f1_report = labeler_utils.evaluate_accuracy(
            self.y_pred,
            self.y_true,
            self.num_labels,
            self.reverse_label_mapping,
            omitted_labels=[],
            verbose=False,
        )

        self.assertEqual((1 / 2 + 2 / 3) / 3, f1)
        self.assertDictEqual(expected_output, f1_report)

    def test_omit_1_class(self):

        expected_output = {
            "UNKNOWN": {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 1,
            },
            "OTHER": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "micro avg": {
                "precision": 1 / 2,
                "recall": 1 / 2,
                "f1-score": 1 / 2,
                "support": 4,
            },
            "macro avg": {
                "precision": 1 / 3,
                "recall": 1 / 3,
                "f1-score": 1 / 3,
                "support": 4,
            },
            "weighted avg": {
                "precision": 1 / 2,
                "recall": 1 / 2,
                "f1-score": 1 / 2,
                "support": 4,
            },
        }

        f1, f1_report = labeler_utils.evaluate_accuracy(
            self.y_pred,
            self.y_true,
            self.num_labels,
            self.reverse_label_mapping,
            omitted_labels=["PAD"],
            verbose=False,
        )

        self.assertEqual(1 / 3, f1)
        self.assertDictEqual(expected_output, f1_report)

    def test_omit_2_classes(self):

        expected_output = {
            "OTHER": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "micro avg": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "macro avg": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "weighted avg": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
        }

        f1, f1_report = labeler_utils.evaluate_accuracy(
            self.y_pred,
            self.y_true,
            self.num_labels,
            self.reverse_label_mapping,
            verbose=False,
        )

        self.assertEqual(2 / 3, f1)
        self.assertDictEqual(expected_output, f1_report)

    def test_no_support_classes(self):

        expected_output = {
            "OTHER": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "NO_SUPPORT": {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 0,
            },
            "NO_SUPPORT2": {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 0,
            },
            "micro avg": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "macro avg": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
            "weighted avg": {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1-score": 2 / 3,
                "support": 3,
            },
        }

        reverse_label_mapping = self.reverse_label_mapping.copy()
        reverse_label_mapping[3] = "NO_SUPPORT"
        reverse_label_mapping[4] = "NO_SUPPORT2"

        f1, f1_report = labeler_utils.evaluate_accuracy(
            self.y_pred,
            self.y_true,
            self.num_labels + 2,
            reverse_label_mapping,
            verbose=False,
        )

        self.assertEqual(2 / 3, f1)
        self.assertDictEqual(expected_output, f1_report)

    def test_verbose(self):
        with self.assertLogs("DataProfiler.labelers.labeler_utils", level="INFO") as cm:
            f1, f1_report = labeler_utils.evaluate_accuracy(
                self.y_pred,
                self.y_true,
                self.num_labels,
                self.reverse_label_mapping,
                omitted_labels=[],
                verbose=True,
            )

        log_output = "".join(cm.output)

        self.assertIn("PAD", log_output)
        self.assertIn("UNKNOWN", log_output)
        self.assertIn("OTHER", log_output)
        self.assertIn("weighted avg", log_output)
        self.assertIn("accuracy", log_output)
        self.assertIn("macro avg", log_output)
        self.assertIn("support", log_output)
        self.assertIn("f1-score ", log_output)
        self.assertIn("F1 Score: ", log_output)

    @mock.patch("pandas.DataFrame")
    def test_save_conf_mat(self, mock_dataframe):

        # ideally mock out the actual contents written to file, but
        # would be difficult to get this completely worked out.
        expected_conf_mat = np.array(
            [
                [1, 0, 1],
                [1, 0, 0],
                [0, 1, 2],
            ]
        )
        expected_row_col_names = dict(
            columns=["pred:PAD", "pred:UNKNOWN", "pred:OTHER"],
            index=["true:PAD", "true:UNKNOWN", "true:OTHER"],
        )
        mock_instance_df = mock.Mock(spec=pd.DataFrame)()
        mock_dataframe.return_value = mock_instance_df

        # still omit bc confusion mat should include all despite omit
        f1, f1_report = labeler_utils.evaluate_accuracy(
            self.y_pred,
            self.y_true,
            self.num_labels,
            self.reverse_label_mapping,
            omitted_labels=["PAD"],
            verbose=False,
            confusion_matrix_file="test.csv",
        )

        self.assertTrue((mock_dataframe.call_args[0][0] == expected_conf_mat).all())
        self.assertDictEqual(expected_row_col_names, mock_dataframe.call_args[1])

        mock_instance_df.to_csv.assert_called()


class TestTFFunctions(unittest.TestCase):
    def test_get_tf_layer_index_from_name(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input((1, 2), name="input"))
        model.add(tf.keras.layers.Dense(units=4, name="dense0"))
        model.add(tf.keras.layers.Dense(units=3, name="dense1"))

        ind = labeler_utils.get_tf_layer_index_from_name(model, "not a layer")
        self.assertIsNone(ind)

        # input is not counted in the layer
        ind = labeler_utils.get_tf_layer_index_from_name(model, "input")
        self.assertIsNone(ind)

        ind = labeler_utils.get_tf_layer_index_from_name(model, "dense1")
        self.assertEqual(1, ind)

        ind = labeler_utils.get_tf_layer_index_from_name(model, "dense0")
        self.assertEqual(0, ind)

    def test_hide_tf_logger_warnings(self):
        logger = logging.getLogger("tensorflow")
        num_loggers = len(logger.filters)

        # make change and validate updated filter
        labeler_utils.hide_tf_logger_warnings()
        self.assertEqual(1 + num_loggers, len(logger.filters))
