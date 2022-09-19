import os
import unittest

import numpy as np
import pkg_resources

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
        cls.data = ["ssn"]

        parameters = {
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
            "include_label": True,
        }

        preprocessor = DirectPassPreprocessor()
        model = ColumnNameModel(parameters=parameters)
        postprocessor = ColumnNameModelPostprocessor(
            true_positive_dict=parameters["true_positive_dict"],
            positive_threshold_config=85,
        )

        cls.data_labeler = BaseDataLabeler.load_with_components(
            preprocessor=preprocessor, model=model, postprocessor=postprocessor
        )

    def test_default_model(self):
        """simple test of predict"""

        # get prediction from labeler
        labeler_predictions = self.data_labeler.predict(self.data)

        # for now just checking that it's not empty
        self.assertIsNotNone(labeler_predictions)
        self.assertEqual(len(self.data), len(labeler_predictions))


if __name__ == "__main__":
    unittest.main()
