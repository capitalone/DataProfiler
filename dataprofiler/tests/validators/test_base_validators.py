import os
import unittest

from dask import dataframe as dd

from dataprofiler.data_readers.data import Data
from dataprofiler.validators.base_validators import Validator

# go up one folder
MODULE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDataValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(MODULE_PATH, "data")
        cls.input_file_names = [
            dict(
                path=os.path.join(test_dir, "csv/aws_honeypot_marx_geo.csv"), type="csv"
            ),
        ]

        file = cls.input_file_names[0]
        data = Data(file["path"])
        sub_data = data.data.head(100)
        cls.sub_data = sub_data
        cls.dask_data = dd.from_pandas(sub_data, npartitions=2)

        cls.correct_pandas_config = {
            "df_type": "pandas",
            "known_anomaly_validation": {
                "int_col": {"range": {"start": 3000, "end": 4000}, "list": [2192]}
            },
        }

        cls.correct_dask_config = {
            "df_type": "dask",
            "known_anomaly_validation": {
                "int_col": {"range": {"start": 3000, "end": 4000}, "list": [2192]}
            },
        }

        cls.wrong_config = {
            "df_type": "pandas",
            "known_anomaly_validation": {
                "int_col": {"rng": {"start": 3000, "end": 4000}}
            },
        }

        cls.empty_config = {
            "df_type": "pandas",
            "known_anomaly_validation": {"int_col": {}},
        }

        cls.true_results = {
            "df_type": "pandas",
            "known_anomaly_validation": {
                "int_col": {"range": [1, 2, 5, 13, 24, 45, 52, 75, 98], "list": [99]}
            },
        }

    def test_data_validation(self):
        """
        Testing to ensure that the validate method does not return None.
        """
        validator = Validator()
        validator.validate(data=self.sub_data, config=self.correct_pandas_config)
        self.assertIsNotNone(validator.validation_report)

    def test_dask_data_validation(self):
        """
        Testing to ensure that the validate method does not return None.
        """
        validator = Validator()
        validator.validate(data=self.dask_data, config=self.correct_dask_config)
        self.assertIsNotNone(validator.validation_report)

    def test_data_validation_output(self):
        """
        Test that the validation runs and returns the correct output.
        """
        validator = Validator()
        validator.validate(data=self.sub_data, config=self.correct_pandas_config)
        self.assertEqual(
            validator.validation_report["int_col"]["range"],
            self.true_results["known_anomaly_validation"]["int_col"]["range"],
        )
        self.assertEqual(
            validator.validation_report["int_col"]["list"],
            self.true_results["known_anomaly_validation"]["int_col"]["list"],
        )

    def test_data_validation_wrong_config(self):
        """
        Test that the validation method raises exceptions when provided
        a wrong configuration dictionary.
        """
        validator = Validator()
        with self.assertRaises(TypeError):
            validator.validate(data=self.sub_data, config=self.wrong_config)

    def test_data_validation_empty_config(self):
        """
        Test that the validation method raises exceptions when provided
        a empty configuration dictionary.
        """
        validator = Validator()
        with self.assertRaises(Warning):
            validator.validate(data=self.sub_data, config=self.empty_config)

    def test_data_validation_no_run_with_get(self):
        """
        Test that the validation method raises exceptions when provided
        a empty configuration dictionary.
        """
        validator = Validator()
        with self.assertRaises(Warning):
            validator.get()


if __name__ == "__main__":
    unittest.main()
