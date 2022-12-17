import unittest

import numpy as np
import pkg_resources

import dataprofiler as dp
from dataprofiler.labelers.column_name_model import ColumnNameModel
from dataprofiler.labelers.data_labelers import BaseDataLabeler
from dataprofiler.labelers.data_processing import (
    ColumnNameModelPostprocessor,
    DirectPassPreprocessor,
)

default_labeler_dir = pkg_resources.resource_filename("resources", "labelers")


class TestColumnNameDataLabeler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.one_data = ["ssn"]
        cls.two_data = ["ssn", "failing_fail_fail"]

        cls.parameters = {
            "true_positive_dict": [
                {"attribute": "ssn", "label": "ssn"},
                {"attribute": "suffix", "label": "name"},
                {"attribute": "my_home_address", "label": "address"},
            ],
            "false_positive_dict": [
                {
                    "attribute": "contract_number",
                    "label": "ssn",
                },
                {
                    "attribute": "role",
                    "label": "name",
                },
                {
                    "attribute": "send_address",
                    "label": "address",
                },
            ],
            "negative_threshold_config": 50,
            "positive_threshold_config": 85,
            "include_label": True,
        }

        cls.label_mapping = [
            label["label"] for label in cls.parameters["true_positive_dict"]
        ]

        preprocessor = DirectPassPreprocessor()
        model = ColumnNameModel(
            label_mapping=cls.label_mapping, parameters=cls.parameters
        )
        postprocessor = ColumnNameModelPostprocessor()

        cls.data_labeler = BaseDataLabeler.load_with_components(
            preprocessor=preprocessor, model=model, postprocessor=postprocessor
        )

    def test_default_model(self):
        """simple test of the DataLabeler's predict"""

        # get prediction from labeler
        labeler_predictions = self.data_labeler.predict(self.one_data)

        # for now just checking that it's not empty
        # and that let of output is the same as len of
        # input values for the model to predict
        self.assertIsNotNone(labeler_predictions)
        self.assertEqual(len(self.one_data), len(labeler_predictions))

    def test_results_filtering(self):
        """test where false negative doesn't exist
        and true positive is filtered
        """

        self.parameters.pop("false_positive_dict")
        model = ColumnNameModel(
            label_mapping=self.label_mapping, parameters=self.parameters
        )

        labeler_predictions = self.data_labeler.predict(self.two_data)

        self.assertIsNotNone(labeler_predictions)
        self.assertEqual(1, len(labeler_predictions))

    def test_load_from_library(self):
        """test successful load from model library"""
        dp.DataLabeler.load_from_library("column_name_labeler")


if __name__ == "__main__":
    unittest.main()
