import os
import unittest

from data_profiler.data_readers import data_utils


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestDataReadingWriting(unittest.TestCase):

    def test_file_UTF_encoding_detection(self):
        """
        Tests the ability for `data_utils.detect_file_encoding` to detect the
        encoding of text files. This test is specifically for UTF-8, UTF-16,
        and UTF-32 of csv or JSON.
        :return:
        """
        test_dir = os.path.join(test_root_path, 'data')
        input_files = [
            dict(path=os.path.join(test_dir, 'csv/iris-utf-8.csv'),
                 encoding="utf-8"),
            dict(path=os.path.join(test_dir, 'csv/iris-utf-16.csv'),
                 encoding="utf-16"),
            dict(path=os.path.join(test_dir, 'csv/iris-utf-32.csv'),
                 encoding="utf-32"),
            dict(path=os.path.join(test_dir, 'json/iris-utf-8.json'),
                 encoding="utf-8"),
            dict(path=os.path.join(test_dir, 'json/iris-utf-16.json'),
                 encoding="utf-16"),
            dict(path=os.path.join(test_dir, 'json/iris-utf-32.json'),
                 encoding="utf-32"),
        ]

        for input_file in input_files:
            detected_encoding = \
                data_utils.detect_file_encoding(file_path=input_file["path"])
            self.assertEqual(detected_encoding.lower(), input_file["encoding"])
