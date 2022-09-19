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
        cls.data = ["test_social_security_number"]

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

        model = ColumnNameModel(parameters=parameters)

        cls.data_labeler = BaseDataLabeler.load_with_components(
            preprocessor=DirectPassPreprocessor(),
            model=model,
            postprocessor=ColumnNameModelPostprocessor(),
        )

    def test_default_model(self):
        """simple test of predict"""

        data_labeler = self.data_labeler

        # get char-level predictions on default model
        model_predictions = data_labeler.predict(self.data)
        final_results = model_predictions["pred"]

        # for now just checking that it's not empty
        self.assertIsNotNone(final_results)
        self.assertEqual(len(self.data), len(final_results))

    def test_default_confidences(self):
        """tests confidence scores output"""

        data_labeler = self.data_labeler

        # get char-level predictions/confidence scores on default model
        results = data_labeler.predict(
            self.data, predict_options=dict(show_confidences=True)
        )
        model_predictions_char_level, model_confidences_char_level = (
            results["pred"],
            results["conf"],
        )

        # for now just checking that it's not empty
        self.assertIsNotNone(model_confidences_char_level)


if __name__ == "__main__":
    unittest.main()
