import os
import unittest

import numpy as np
import pandas as pd

import dataprofiler as dp


class TestV2StructuredDataLabeler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        num_repeat_data = 50
        cls.data = np.array(
            [['123 Fake St.', '1/2/2020', 'nice.'],
             ['4/3/22', 'abc', '333-44-2341']] * num_repeat_data
        ).reshape((-1,))
        cls.labels = np.array(
            [['ADDRESS', 'DATETIME', 'BACKGROUND'],
             ['DATETIME', 'BACKGROUND', 'SSN']] * num_repeat_data
        ).reshape((-1,))
        cls.df = pd.DataFrame([cls.data, cls.labels]).T

    # simple test for new default TF model + predict()
    def test_fit_with_default_model(self):
        """test fitting with data labeler"""
        # constructing default StructuredDataLabeler()
        dirpath = os.path.join(
            dp.labelers.base_data_labeler.default_labeler_dir,
            dp.labelers.StructuredDataLabeler._default_model_loc)
        default = dp.labelers.TrainableDataLabeler(dirpath=dirpath)

        # validate epoch id
        self.assertEqual(0, default.model._epoch_id)

        # with labels process

        # get char-level predictions on default model
        model_predictions = default.fit(x=self.df[0], y=self.df[1])
        self.assertEqual(1, len(model_predictions))  # 1 epoch only, so 1 result
        self.assertEqual(3, len(model_predictions[0]))  # history, f1, f1_report
        self.assertIsInstance(model_predictions[0][0], dict)  # history
        self.assertIsInstance(model_predictions[0][1], float)  # float
        self.assertIsInstance(model_predictions[0][2], dict)  # f1_report

        # validate epoch id
        self.assertEqual(1, default.model._epoch_id)

        # no bg, pad, but includes micro, macro, weighted
        # default labels + micro, macro, weighted - bg, pad
        self.assertEqual(len(default.labels)+1, len(model_predictions[0][2].keys()))

        # test default no validation
        model_predictions = default.fit(
            x=self.df[0], y=self.df[1], validation_split=0)
        self.assertEqual(1, len(model_predictions))  # 1 epoch only, so 1 result
        self.assertEqual(3, len(model_predictions[0]))  # history, f1, f1_report
        self.assertIsInstance(model_predictions[0][0], dict)  # history
        self.assertIsNone(model_predictions[0][1])  # no f1 since no validation
        self.assertListEqual(model_predictions[0][2], [])  # empty f1_report

        # validate epoch id
        self.assertEqual(2, default.model._epoch_id)

    def test_data_labeler_change_labels(self):
        """test changing labels of data labeler with fitting data"""
        # constructing default StructuredDataLabeler()
        dirpath = os.path.join(
            dp.labelers.base_data_labeler.default_labeler_dir,
            dp.labelers.StructuredDataLabeler._default_model_loc)
        default = dp.labelers.TrainableDataLabeler(dirpath=dirpath)

        # get char-level predictions on default model
        expected_label_mapping = dict(list(zip(
            ['PAD', 'BACKGROUND', 'ADDRESS', 'DATETIME', 'SSN'],
            [0, 1, 2, 3, 4]
        )))
        model_predictions = default.fit(
            x=self.df[0], y=self.df[1],
            labels=['BACKGROUND', 'ADDRESS', 'DATETIME', 'SSN'])
        self.assertEqual(1, len(model_predictions))
        self.assertEqual(3, len(model_predictions[0]))  # history, f1, f1_report
        self.assertIsInstance(model_predictions[0][0], dict)  # history
        self.assertIsInstance(model_predictions[0][1], float)  # f1
        self.assertIsInstance(model_predictions[0][2], dict)  # f1_report
        self.assertDictEqual(expected_label_mapping, default.label_mapping)

        # no bg, pad, but includes micro, macro, weighted
        self.assertEqual(6, len(model_predictions[0][2].keys()))

        # ensure as long as label in label_mapping, will work.
        # get char-level predictions on default model
        try:
            model_predictions = default.fit(
                x=self.df[0], y=self.df[1],
                labels=['BACKGROUND', 'ADDRESS', 'DATETIME', 'SSN',
                        'CREDIT_CARD'])
        except Exception as e:
            self.fail(str(e))

        # failure occurs if label in data frame ont in labels
        with self.assertRaises(KeyError):
            model_predictions = default.fit(
                x=self.df[0], y=self.df[1],
                labels=['BACKGROUND', 'ADDRESS', 'DATETIME'])

    def test_data_labeler_extend_labels(self):
        """test extending labels of data labeler with fitting data"""
        # constructing default StructuredDataLabeler()
        dirpath = os.path.join(
            dp.labelers.base_data_labeler.default_labeler_dir,
            dp.labelers.StructuredDataLabeler._default_model_loc)
        data_labeler = dp.labelers.TrainableDataLabeler(dirpath=dirpath)

        data_labeler.add_label('NEW_LABEL')

        # validate raises error if not trained before fit
        with self.assertRaisesRegex(RuntimeError,
                                    "The model label mapping definitions have "
                                    "been altered without additional training. "
                                    "Please train the model or reset the "
                                    "label mapping to predict."):
            model_predictions = data_labeler.predict(data=self.df[0])


        # get char-level predictions on default model
        expected_label_mapping = {
            "PAD": 0, "CITY": 1, "BACKGROUND": 1, "ADDRESS": 2, "BAN": 3,
            "CREDIT_CARD": 4, "EMAIL_ADDRESS": 5, "UUID": 6, "HASH_OR_KEY": 7,
            "IPV4": 8, "IPV6": 9, "MAC_ADDRESS": 10, "NAME": 11, "PERSON": 11,
            "PHONE_NUMBER": 12, "SSN": 13, "URL": 14, "DATETIME": 15,
            "INTEGER_BIG": 16, "INTEGER": 16, "FLOAT": 17, "QUANTITY": 18,
            "ORDINAL": 19, "NEW_LABEL": 20}
        model_predictions = data_labeler.fit(x=self.df[0], y=self.df[1])

        self.assertEqual(1, len(model_predictions))
        self.assertEqual(3, len(model_predictions[0]))  # history, f1, f1_report
        self.assertIsInstance(model_predictions[0][0], dict)  # history
        self.assertIsInstance(model_predictions[0][1], float)  # f1
        self.assertIsInstance(model_predictions[0][2], dict)  # f1_report
        self.assertDictEqual(expected_label_mapping, data_labeler.label_mapping)

        # no bg, pad, but includes micro, macro, weighted
        self.assertEqual(22, len(model_predictions[0][2].keys()))

    def test_default_tf_model(self):
        """simple test for new default TF model + predict()"""

        # constructing default StructuredDataLabeler()
        default = dp.labelers.StructuredDataLabeler()

        # get char-level predictions on default model
        model_predictions = default.predict(self.data)
        final_results = model_predictions["pred"]

        # for now just checking that it's not empty
        self.assertIsNotNone(final_results)
        self.assertEqual(len(self.data), len(final_results))

    def test_default_confidences(self):
        """tests confidence scores output"""

        # constructing default StructuredDataLabeler()
        default = dp.labelers.StructuredDataLabeler()

        # get char-level predictions/confidence scores on default model
        results = default.predict(self.data,
                                  predict_options=dict(show_confidences=True))
        model_predictions_char_level, model_confidences_char_level = \
            results["pred"], results["conf"]

        # for now just checking that it's not empty
        self.assertIsNotNone(model_confidences_char_level)

    def test_default_edge_cases(self):
        """more complicated test for edge cases for the default model"""
        sample = ["1234567890", "!@#$%&^*$)*#%)#*%-=+~.,/?{}[]|`",
                  "\n \n \n \t \t"]

        # constructing default StructuredDataLabeler()
        default = dp.labelers.StructuredDataLabeler()

        # get char-level predictions on default model
        model_predictions = default.predict(sample)
        final_results = model_predictions["pred"]

        # for now just checking that it's not empty
        self.assertIsNotNone(final_results)

    def test_default_special_cases(self):
        """
        tests for empty string (returns none) and mixed samples cases
        (throws error) w/ default labeler
        """
        # first test multiple empty strings in sample:
        sample1 = ["", "", ""]

        # constructing default StructuredDataLabeler()
        default = dp.labelers.StructuredDataLabeler()

        # get char-level predictions on default model
        results = default.predict(
            sample1, predict_options=dict(show_confidences=True))
        model_predictions_char_level = results["pred"]
        model_confidences_char_level = results["conf"]

        # test that we get empty list for predictions/confidences:
        self.assertEqual(model_predictions_char_level.tolist(),
                         [None, None, None])
        self.assertTrue((model_confidences_char_level == 0.0).all())

        # Now we test mixed samples case:
        sample2 = ["", "abc", "\t", ""]

        # this can change if model changes
        expected_output = {'pred': [None, 'BACKGROUND', 'BACKGROUND', None]}
        output = default.predict(sample2)
        output['pred'] = output['pred'].tolist()
        self.assertDictEqual(expected_output, output)

    # simple test for new default TF model + predict()
    def test_fit_with_reset_weights(self):
        """test fitting with data labeler while resetting the weights"""
        # constructing default StructuredDataLabeler()
        dirpath = os.path.join(
            dp.labelers.base_data_labeler.default_labeler_dir,
            dp.labelers.StructuredDataLabeler._default_model_loc)
        default = dp.labelers.TrainableDataLabeler(dirpath=dirpath)

        # fit on default model with reset weights  
        model_predictions = default.fit(
            x=self.df[0], y=self.df[1], reset_weights=True)

        # assert appropriate results
        self.assertEqual(1, len(model_predictions))  # 1 epoch only, so 1 result
        self.assertEqual(3, len(model_predictions[0]))  # history, f1, f1_report
        self.assertIsInstance(model_predictions[0][0], dict)  # history
        self.assertIsInstance(model_predictions[0][1], float)  # float
        self.assertIsInstance(model_predictions[0][2], dict)  # f1_report

        # validate epoch id
        self.assertEqual(1, default.model._epoch_id)

        # test subsequent weight reset
        model_predictions = default.fit(
            x=self.df[0], y=self.df[1], reset_weights=True)

        # validate epoch id
        self.assertEqual(1, default.model._epoch_id)


if __name__ == '__main__':
    unittest.main()
