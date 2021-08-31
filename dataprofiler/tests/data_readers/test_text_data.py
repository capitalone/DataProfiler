import os
import unittest
from io import StringIO, BytesIO

from dataprofiler.data_readers.data import Data, TextData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestTextDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'txt/discussion_hn.txt'),
                 count=58028, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/discussion_reddit.txt'),
                 count=80913, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/code.txt'),
                 count=57489, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/empty.txt'),
                 count=0, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/sentence.txt'),
                 count=19, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/sentence-3x.txt'),
                 count=77, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/sentence-10x.txt'),
                 count=430, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/quote-test-incorrect-csv.txt'),
                 count=60, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/utf8.txt'),
                 count=109, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'txt/utf16.txt'),
                 count=109, encoding='utf-16'),
            dict(path=os.path.join(test_dir, 'txt/html-csv-confusion.html'),
                 count=696, encoding='utf-8'),
        ]
        cls.output_file_path = None

        cls.buffer_list = []
        for input_file in cls.input_file_names:
            # add StringIO
            buffer_info = input_file.copy()
            with open(input_file['path'], 'r',
                      encoding=input_file['encoding']) as fp:
                buffer_info['path'] = StringIO(fp.read())
            cls.buffer_list.append(buffer_info)
            
            # add BytesIO
            buffer_info = input_file.copy()
            with open(input_file['path'], 'rb') as fp:
                buffer_info['path'] = BytesIO(fp.read())
            cls.buffer_list.append(buffer_info)

        cls.file_or_buf_list = cls.input_file_names + cls.buffer_list

    @classmethod
    def setUp(cls):
        for buffer in cls.buffer_list:
            buffer['path'].seek(0)

    def test_is_match(self):
        """
        Determine if the text file can be automatically identified from
        byte stream or stringio stream or filepath
        """
        for input_file in self.file_or_buf_list:
            self.assertTrue(TextData.is_match(input_file['path']))

    def test_samples_per_line(self):
        for input_file in self.file_or_buf_list:
            input_data_obj = TextData(input_file['path'],
                                      options={'samples_per_line': 1})
            self.assertEqual(input_file['count'], len(input_data_obj),
                             input_file['path'])

    def test_auto_file_identification(self):
        """
        Determine if the text file can be automatically identified
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            self.assertEqual('text', input_data_obj.data_type,
                             input_file['path'])

    def test_specifying_data_type(self):
        """
        Determine if the text file can be loaded with manual data_type setting
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"], data_type='text')
            self.assertEqual(input_data_obj.data_type, 'text')

    def test_reload_data(self):
        """
        Determine if the text file can be reloaded
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            input_data_obj.reload(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'text',
                             input_file['path'])
            self.assertEqual(input_file['path'], input_data_obj.input_file_path)

    def test_is_structured(self):
        """
        Ensures TextData.is_structured is False
        """
        self.assertFalse(TextData().is_structured)


if __name__ == '__main__':
    unittest.main()
