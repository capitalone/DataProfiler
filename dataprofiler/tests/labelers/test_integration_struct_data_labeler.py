import os
import unittest

import numpy as np
import pandas as pd

import dataprofiler as dp
from dataprofiler.labelers.character_level_cnn_model import \
    CharacterLevelCnnModel


class TestStructuredDataLabeler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        num_repeat_data = 50
        cls.data = np.array(
            [['123 Fake St.', '1/2/2020', 'nice.'],
             ['4/3/22', 'abc', '333-44-2341']] * num_repeat_data
        ).reshape((-1,))
        cls.labels = np.array(
            [['ADDRESS', 'DATETIME', 'UNKNOWN'],
             ['DATETIME', 'UNKNOWN', 'SSN']] * num_repeat_data
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
            ['PAD', 'UNKNOWN', 'ADDRESS', 'DATETIME', 'SSN'],
            [0, 1, 2, 3, 4]
        )))
        model_predictions = default.fit(
            x=self.df[0], y=self.df[1],
            labels=['UNKNOWN', 'ADDRESS', 'DATETIME', 'SSN'])
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
                labels=['UNKNOWN', 'ADDRESS', 'DATETIME', 'SSN',
                        'CREDIT_CARD'])
        except Exception as e:
            self.fail(str(e))

        # failure occurs if label in data frame ont in labels
        with self.assertRaises(KeyError):
            model_predictions = default.fit(
                x=self.df[0], y=self.df[1],
                labels=['UNKNOWN', 'ADDRESS', 'DATETIME'])

    def test_data_labeler_extend_labels(self):
        """test extending labels of data labeler with fitting data"""
        # constructing default StructuredDataLabeler()
        dirpath = os.path.join(
            dp.labelers.base_data_labeler.default_labeler_dir,
            dp.labelers.StructuredDataLabeler._default_model_loc)
        data_labeler = dp.labelers.TrainableDataLabeler(dirpath=dirpath)

        original_label_mapping = data_labeler.label_mapping.copy()
        original_max_label = data_labeler.label_mapping[
            max(data_labeler.label_mapping, key=data_labeler.label_mapping.get)]

        new_label = 'NEW_LABEL'
        data_labeler.add_label(new_label)

        new_max_label = data_labeler.label_mapping[
            max(data_labeler.label_mapping, key=data_labeler.label_mapping.get)]

        expected_label_mapping = original_label_mapping
        expected_label_mapping[new_label] = new_max_label

        new_label_count = len(data_labeler.label_mapping)

        # validate raises error if not trained before fit
        with self.assertRaisesRegex(RuntimeError,
                                    "The model label mapping definitions have "
                                    "been altered without additional training. "
                                    "Please train the model or reset the "
                                    "label mapping to predict."):
            model_predictions = data_labeler.predict(data=self.df[0])

        model_predictions = data_labeler.fit(x=self.df[0], y=self.df[1])

        self.assertEqual(1, len(model_predictions))
        self.assertEqual(3, len(model_predictions[0]))  # history, f1, f1_report
        self.assertIsInstance(model_predictions[0][0], dict)  # history
        self.assertIsInstance(model_predictions[0][1], float)  # f1
        self.assertIsInstance(model_predictions[0][2], dict)  # f1_report
        self.assertIn(new_label, data_labeler.label_mapping) # Ensure new label added
        self.assertEqual(original_max_label+1, new_max_label) # Ensure new label iterated
        self.assertDictEqual(expected_label_mapping, data_labeler.label_mapping)

        # no bg, pad, but includes micro, macro, weighted
        self.assertEqual(new_label_count+1, len(model_predictions[0][2].keys()))

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
        expected_output = {'pred': [None, 'UNKNOWN', 'UNKNOWN', None]}
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

    def test_structured_data_labeler_fit_predict_take_data_obj(self):
        data = pd.DataFrame(["123 Fake st", "1/1/2021", "blah", "333-44-2341",
                             "foobar@gmail.com", "John Doe", "123-4567"])
        labels = pd.DataFrame(["ADDRESS", "DATETIME", "UNKNOWN", "SSN",
                               "EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"])
        for dt in ["csv", "json", "parquet"]:
            data_obj = dp.Data(data=data, data_type=dt)
            label_obj = dp.Data(data=labels, data_type=dt)
            labeler = dp.DataLabeler(labeler_type="structured", trainable=True)
            self.assertIsNotNone(labeler.fit(x=data_obj, y=label_obj))
            self.assertIsNotNone(labeler.predict(data=data_obj))

    def test_warning_tf(self):

        test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_dir = os.path.join(test_root_path, 'data')
        path = os.path.join(test_dir, 'csv/diamonds.csv')
        data = dp.Data(path)

        profile_options = dp.ProfilerOptions()
        profile_options.set({"text.is_enabled": False,
                             "int.is_enabled": False,
                             "float.is_enabled": False,
                             "order.is_enabled": False,
                             "category.is_enabled": False,
                             "datetime.is_enabled": False, })

        profile = dp.Profiler(data, profiler_options=profile_options)
        results = profile.report()

        columns = []
        predictions = []
        for col in results['data_stats']:
            columns.append(col)
            predictions.append(results['data_stats'][col]['data_label'])


    def test_warning_tf_run_dp_multiple_times(self):
        test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_dir = os.path.join(test_root_path, 'data')
        path = os.path.join(test_dir, 'csv/diamonds.csv')

        for i in range(3):
            print('running dp =============================', i)
            data = dp.Data(path)
            profile_options = dp.ProfilerOptions()
            profile_options.set({"text.is_enabled": False,
                                 "int.is_enabled": False,
                                 "float.is_enabled": False,
                                 "order.is_enabled": False,
                                 "category.is_enabled": False,
                                 "datetime.is_enabled": False, })

            profile = dp.Profiler(data, profiler_options=profile_options)

            results = profile.report()

            columns = []
            predictions = []
            for col in results['data_stats']:
                columns.append(col)
                predictions.append(results['data_stats'][col]['data_label'])

    def test_warning_tf_run_dp_merge(self):
        test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_dir = os.path.join(test_root_path, 'data')
        path = os.path.join(test_dir, 'csv/diamonds.csv')

        data = dp.Data(path)
        profile_options = dp.ProfilerOptions()
        profile_options.set({"text.is_enabled": False,
                             "int.is_enabled": False,
                             "float.is_enabled": False,
                             "order.is_enabled": False,
                             "category.is_enabled": False,
                             "datetime.is_enabled": False, })
        print('running dp1')
        profile1 = dp.Profiler(data, profiler_options=profile_options)

        data = dp.Data(path)
        profile_options = dp.ProfilerOptions()
        profile_options.set({"text.is_enabled": False,
                             "int.is_enabled": False,
                             "float.is_enabled": False,
                             "order.is_enabled": False,
                             "category.is_enabled": False,
                             "datetime.is_enabled": False, })
        print('running dp2')
        profile2 = dp.Profiler(data, profiler_options=profile_options)

        profile = profile1 + profile2

    def test_warning_tf_multiple_dp_with_update(self):
        test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_dir = os.path.join(test_root_path, 'data')
        path = os.path.join(test_dir, 'csv/diamonds.csv')

        data = dp.Data(path)
        profile_options = dp.ProfilerOptions()
        profile_options.set({"text.is_enabled": False,
                             "int.is_enabled": False,
                             "float.is_enabled": False,
                             "order.is_enabled": False,
                             "category.is_enabled": False,
                             "datetime.is_enabled": False, })
        print('running dp1')
        profile1 = dp.Profiler(data, profiler_options=profile_options)

        data = dp.Data(path)
        profile_options = dp.ProfilerOptions()
        profile_options.set({"text.is_enabled": False,
                             "int.is_enabled": False,
                             "float.is_enabled": False,
                             "order.is_enabled": False,
                             "category.is_enabled": False,
                             "datetime.is_enabled": False, })
        print('running dp2')
        profile2 = dp.Profiler(data, profiler_options=profile_options)

        profile1.update_profile(data)

if __name__ == '__main__':
    unittest.main()
