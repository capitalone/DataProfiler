import unittest

import numpy as np

from dataprofiler.labelers import classification_report_utils


class TestClassificationReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reverse_label_mapping = {
            0: "PAD",
            1: "UNKNOWN",
            2: "OTHER",
        }

    def test_classification_report(self):
        conf_matrix = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 2]])

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

        report = classification_report_utils.classification_report(
            conf_matrix,
            labels=list(self.reverse_label_mapping.keys()),
            target_names=list(self.reverse_label_mapping.values()),
            output_dict=True,
        )
        self.assertEqual(expected_output, report)

    def test_print_classification_report(self):
        conf_matrix = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 2]])

        report = classification_report_utils.classification_report(
            conf_matrix,
            labels=list(self.reverse_label_mapping.keys()),
            target_names=list(self.reverse_label_mapping.values()),
        )

        self.assertIn("PAD", report)
        self.assertIn("UNKNOWN", report)
        self.assertIn("OTHER", report)
        self.assertIn("weighted avg", report)
        self.assertIn("accuracy", report)
        self.assertIn("macro avg", report)
        self.assertIn("support", report)
        self.assertIn("f1-score ", report)

    def test_convert_confusion_matrix_to_MCM(self):
        conf_matrix = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 2]])

        expected_MCM = np.array([[[3, 1], [1, 1]], [[4, 1], [1, 0]], [[2, 1], [1, 2]]])

        # np.ndarray format
        output_MCM = classification_report_utils.convert_confusion_matrix_to_MCM(
            conf_matrix
        )

        self.assertTrue(np.array_equal(expected_MCM, output_MCM))

        # also check cant take in list format
        output_MCM = classification_report_utils.convert_confusion_matrix_to_MCM(
            conf_matrix.tolist()
        )

        self.assertTrue(np.array_equal(expected_MCM, output_MCM))
