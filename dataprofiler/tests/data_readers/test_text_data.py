import os
import unittest

from dataprofiler.data_readers.data import Data


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestTextDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'txt/discussion_hn.txt'), count=10000),
            dict(path=os.path.join(test_dir, 'txt/discussion_reddit.txt'), count=10000),
            dict(path=os.path.join(test_dir, 'txt/code.txt'), count=10000),
            dict(path=os.path.join(test_dir, 'txt/empty.txt'), count=0),
            dict(path=os.path.join(test_dir, 'txt/sentence.txt'), count=1),
            dict(path=os.path.join(test_dir, 'txt/sentence-3x.txt'), count=3),
            dict(path=os.path.join(test_dir, 'txt/sentence-10x.txt'), count=10),
            dict(path=os.path.join(test_dir, 'txt/quote-test-incorrect-csv.txt'), count=8),
        ]
        cls.output_file_path = None

    def test_auto_file_identification(self):
        """
        Determine if the text file can be automatically identified
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'text', input_file['path'])

    def test_specifying_data_type(self):
        """
        Determine if the text file can be loaded with manual data_type setting
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"], data_type='text')
            self.assertEqual(input_data_obj.data_type, 'text')

    def test_reload_data(self):
        """
        Determine if the text file can be reloaded
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            input_data_obj.reload(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'text', input_file['path'])


if __name__ == '__main__':
    unittest.main()
