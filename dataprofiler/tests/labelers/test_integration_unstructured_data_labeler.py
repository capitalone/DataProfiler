import unittest

import numpy as np
import pandas as pd

import dataprofiler as dp


class TestUnstructuredDataLabeler(unittest.TestCase):

    # simple test for new default TF model + predict()
    def test_fit_with_default_model(self):
        data = [
            [
                "this is my test sentence.",
                [
                    (5, 7, "ADDRESS"),
                    (11, 20, "INTEGER"),
                    (20, 22, "ADDRESS"),
                    (22, 24, "INTEGER"),
                ],
            ],
            ["How nice.", [(0, 2, "ADDRESS"), (4, 5, "INTEGER"), (6, 8, "INTEGER")]],
        ]
        new_labels = ["UNKNOWN", "ADDRESS", "INTEGER"]
        data = pd.DataFrame(data * 50)

        # constructing default UnstructuredDataLabeler()
        default = dp.DataLabeler(labeler_type="unstructured", trainable=True)

        # get char-level predictions on default model
        model_predictions = default.fit(x=data[0], y=data[1], labels=new_labels)
        self.assertEqual(1, len(model_predictions))
        self.assertEqual(3, len(model_predictions[0]))
        self.assertIsInstance(model_predictions[0][0], dict)
        self.assertIsInstance(model_predictions[0][1], float)
        self.assertIsInstance(model_predictions[0][2], dict)

        # no bg, pad, but includes micro, macro, weighted
        self.assertEqual(len(default.labels) + 1, len(model_predictions[0][2].keys()))

        # test default no validation
        model_predictions = default.fit(x=data[0], y=data[1], validation_split=0)
        self.assertEqual(1, len(model_predictions))
        self.assertEqual(3, len(model_predictions[0]))
        self.assertIsInstance(model_predictions[0][0], dict)
        self.assertIsNone(model_predictions[0][1])  # no f1 since no validation
        self.assertDictEqual(model_predictions[0][2], {})  # empty f1_report

        # validate epoch id
        self.assertEqual(2, default.model._epoch_id)

    def test_data_labeler_change_labels(self):
        data = [
            [
                "this is my test sentence.",
                [
                    (5, 7, "ADDRESS"),
                    (11, 20, "INTEGER"),
                    (20, 22, "ADDRESS"),
                    (22, 24, "INTEGER"),
                ],
            ],
            ["How nice.", [(0, 2, "ADDRESS"), (4, 5, "INTEGER"), (6, 8, "INTEGER")]],
        ]
        data = pd.DataFrame(data * 50)

        # constructing default UnstructuredDataLabeler()
        default = dp.DataLabeler(labeler_type="unstructured", trainable=True)

        # get char-level predictions on default model
        model_predictions = default.fit(
            x=data[0], y=data[1], labels=["UNKNOWN", "INTEGER", "ADDRESS"]
        )
        self.assertEqual(1, len(model_predictions))
        self.assertEqual(3, len(model_predictions[0]))
        self.assertIsInstance(model_predictions[0][0], dict)
        self.assertIsInstance(model_predictions[0][1], float)
        self.assertIsInstance(model_predictions[0][2], dict)

        # no bg, pad, but includes micro,macro, weighted
        self.assertEqual(5, len(model_predictions[0][2].keys()))

    def test_default_tf_model(self):
        """simple test for new default TF model + predict()"""
        sample = [
            "Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234.\t"
            "His\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912.\n",
            "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n",
        ]

        # constructing default UnstructuredDataLabeler()
        default = dp.UnstructuredDataLabeler()

        # get char-level predictions on default model
        model_predictions = default.predict(sample)
        final_results = model_predictions["pred"]
        print(final_results)

        # for now just checking that it's not empty, previous line prints out
        # results
        self.assertIsNotNone(final_results)

    def test_default_confidences(self):
        """tests confidence scores output"""
        sample = [
            "Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234.\t"
            "His\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912.\n",
            "Hi my name is joe, \t SSN: 123456789 r@nd0m numb3rz!\n",
        ]

        # constructing default UnstructuredDataLabeler()
        default = dp.UnstructuredDataLabeler()

        # get char-level predictions/confidence scores on default model
        results = default.predict(sample, predict_options=dict(show_confidences=True))
        model_predictions_char_level, model_confidences_char_level = (
            results["pred"],
            results["conf"],
        )

        # for now just checking that it's not empty and appropriate size
        num_labels = max(default.label_mapping.values()) + 1
        len_text = len(sample[0])
        self.assertIsNotNone(model_confidences_char_level)
        self.assertEqual((len_text, num_labels), model_confidences_char_level[0].shape)

        len_text = len(sample[1])
        self.assertEqual((len_text, num_labels), model_confidences_char_level[1].shape)

    def test_default_edge_cases(self):
        """more complicated test for edge cases for the default model"""
        sample = ["1234567890", "!@#$%&^*$)*#%)#*%-=+~.,/?{}[]|`", "\n \n \n \t \t"]

        # constructing default UnstructuredDataLabeler()
        default = dp.UnstructuredDataLabeler()

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

        # constructing default UnstructuredDataLabeler()
        default = dp.UnstructuredDataLabeler()

        # get char-level predictions on default model
        output_results = default.predict(
            sample1, predict_options=dict(show_confidences=True)
        )

        # test that we get empty list for predictions/confidences:
        expected_result = {
            "pred": [np.array([]), np.array([]), np.array([])],
            "conf": [np.array([]), np.array([]), np.array([])],
        }
        for expected, output in zip(expected_result["pred"], output_results["pred"]):
            self.assertTrue((expected == output).all())
        for expected, output in zip(expected_result["conf"], output_results["conf"]):
            self.assertTrue((expected == output).all())

        # Now we test mixed samples case:
        sample2 = ["", "abc", "\t", ""]

        expected_output = {
            "pred": [
                np.array([]),
                np.array([1.0, 1.0, 1.0]),
                np.array([1.0]),
                np.array([]),
            ]
        }
        output_result = default.predict(sample2)
        for expected, output in zip(expected_output["pred"], output_result["pred"]):
            self.assertTrue((expected == output).all())

    def test_set_pipeline_params(self):
        def does_dict_contains_subset(subset_dict, full_dict):
            return dict(full_dict, **subset_dict) == full_dict

        data_labeler = dp.UnstructuredDataLabeler()

        # validate preset values are not to be set values
        self.assertNotEqual("a", data_labeler.preprocessor._parameters["default_label"])
        self.assertNotEqual("b", data_labeler.model._parameters["default_label"])
        self.assertNotEqual("c", data_labeler.postprocessor._parameters["pad_label"])

        # set parameters of pipeline components
        data_labeler.set_params(
            {
                "preprocessor": {"default_label": "a"},
                "model": {"default_label": "b"},
                "postprocessor": {"pad_label": "c"},
            }
        )

        # preprocessor
        self.assertTrue(
            does_dict_contains_subset(
                {"default_label": "a"}, data_labeler.preprocessor._parameters
            )
        )

        # model
        self.assertTrue(
            does_dict_contains_subset(
                {"default_label": "b"}, data_labeler.model._parameters
            )
        )

        # postprocessor
        self.assertTrue(
            does_dict_contains_subset(
                {"pad_label": "c"}, data_labeler.postprocessor._parameters
            )
        )

    def test_check_pipeline_overlap_mismatch(self):

        data_labeler = dp.UnstructuredDataLabeler()

        # check preprocessor model mismatch
        data_labeler.set_params(
            {"preprocessor": {"default_label": "a"}, "model": {"default_label": "b"}}
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Model and preprocessor value for " "`default_label` do not match. b != a",
        ):
            data_labeler.check_pipeline(skip_postprocessor=True, error_on_mismatch=True)

        # make preprocess and model the same, but different from postprocessor
        data_labeler.set_params(
            {
                "model": {"default_label": "a"},
                "postprocessor": {"default_label": "b"},
            }
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Model and postprocessor value for "
            "`default_label` do not match. "
            "a != b",
        ):
            data_labeler.check_pipeline(
                skip_postprocessor=False, error_on_mismatch=True
            )

    def test_unstructured_data_labeler_fit_predict_take_data_obj(self):
        # Determine string index in joined data at cell i
        def data_ind(i, data):
            # Take off 1 in base case so we don't include trailing comma
            if i == -1:
                return -1
            # Add 1 with every pass to account for commas
            return len(data[i]) + 1 + data_ind(i - 1, data)

        # Generate entities list for a set of structured data and labels
        def entities(data, labels):
            return [(0, len(data[0]), labels[0])] + [
                (data_ind(i - 1, data) + 1, data_ind(i, data), labels[i])
                for i in range(1, len(data))
            ]

        data_cells = [
            "123 Fake st",
            "1/1/2021",
            "blah",
            "555-55-5555",
            "foobar@gmail.com",
            "John Doe",
            "123-4567",
        ]
        label_cells = [
            "ADDRESS",
            "DATETIME",
            "UNKNOWN",
            "SSN",
            "EMAIL_ADDRESS",
            "PERSON",
            "PHONE_NUMBER",
        ]

        # Test with one large string of data
        data_str = ",".join(data_cells)
        label_str = entities(data_cells, label_cells)
        for dt in ["csv", "json", "parquet"]:
            data_obj = dp.Data(data=pd.DataFrame([data_str]), data_type=dt)
            labeler = dp.DataLabeler(labeler_type="unstructured", trainable=True)
            self.assertIsNotNone(labeler.fit(x=data_obj, y=[label_str]))
            self.assertIsNotNone(labeler.predict(data=data_obj))

        # Test with the string broken up into different df entries
        data_1 = data_cells[:3]
        data_2 = data_cells[3:5]
        data_3 = data_cells[5:]
        data_df = pd.DataFrame([",".join(data_1), ",".join(data_2), ",".join(data_3)])
        zipped = [
            (data_1, label_cells[:3]),
            (data_2, label_cells[3:5]),
            (data_3, label_cells[5:]),
        ]
        three_labels = [entities(d, l) for (d, l) in zipped]
        for dt in ["csv", "json", "parquet"]:
            data_obj = dp.Data(data=data_df, data_type=dt)
            labeler = dp.DataLabeler(labeler_type="unstructured", trainable=True)
            self.assertIsNotNone(labeler.fit(x=data_obj, y=three_labels))
            self.assertIsNotNone(labeler.predict(data=data_obj))

        # Test with text data object
        text_obj = dp.Data(data=data_str, data_type="text")
        labeler = dp.DataLabeler(labeler_type="unstructured", trainable=True)
        self.assertIsNotNone(labeler.fit(x=text_obj, y=[label_str]))
        self.assertIsNotNone(labeler.predict(data=text_obj))


if __name__ == "__main__":
    unittest.main()
