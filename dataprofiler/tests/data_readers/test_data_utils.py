import os
import unittest

from dataprofiler.data_readers import data_utils


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

    def test_nth_loc_detection(self):
        """
        Tests the ability for the `data_utils.find_nth_location` to detect the
        nth index of a search_query in a string. 
        """
        test_queries = [
            dict(string="This is a test.", query=".", n=0, index=15, occurances=1)
        ]

        for q in test_queries:
            self.assertEqual(data_utils.find_nth_loc(
                q['string'], q['query'], q['n']), (q['index'], q['occurances']))
