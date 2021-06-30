import unittest
import os
from io import open, StringIO, BytesIO

import pandas as pd

from dataprofiler.data_readers.filepath_or_buffer import FileOrBufferHandler


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class TestFilepathOrBuffer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/diamonds.csv'),
                 count=53940, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris.csv'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),  
            dict(path=os.path.join(test_dir, 'json/iris-utf-8.json'), encoding='utf-8', count=150),
            dict(path=os.path.join(test_dir, 'txt/discussion_hn.txt'), count=10000),
            dict(path=os.path.join(test_dir, 'txt/discussion_reddit.txt'), count=10000),   
        ]
        cls.output_file_path = None

    def test_make_buffer_from_filepath(self):
        """
        Make sure FileOrBufferHandler can input a file and read it similarly to open()
        """
        for input_file in self.input_file_names:
            with FileOrBufferHandler(input_file['path'], 'r') as filepath_or_buffer, open(input_file['path'], 'r') as input_file_check:
                # check every 10th line for the first 100 lines
                for i in range(0, 10):    
                    for i in range(0, 9):
                        filepath_or_buffer.readline()
                        input_file_check.readline()
                    self.assertEqual(filepath_or_buffer.readline(), input_file_check.readline())

            # check that file was properly closed
            self.assertEqual(filepath_or_buffer.closed, input_file_check.closed)

    def test_pass_in_StringIO_buffer(self):
        """
        Make sure FileOrBufferHandler can take StringIO and read it similarly to open() 
        """
        for input_file in self.input_file_names:
            with FileOrBufferHandler(StringIO(open(input_file['path'], 'r').read())) as filepath_or_buffer, open(input_file['path'], 'r') as input_file_check:
                # check every 10th line for the first 100 lines
                for i in range(0, 10):    
                    for i in range(0, 9):
                        filepath_or_buffer.readline()
                        input_file_check.readline()

                    self.assertEqual(filepath_or_buffer.readline(), input_file_check.readline())
                    
            # check that file was properly closed
            self.assertEqual(filepath_or_buffer.closed, input_file_check.closed)

    def test_pass_in_StringIO_seek_buffer(self):
        """
        Make sure FileOrBufferHandler can take StringIO with seek and read it similarly to open() with seek
        """
        for input_file in self.input_file_names:
            seek_offset_test = 100
            with FileOrBufferHandler(StringIO(open(input_file['path'], 'r').read()), seek_offset=seek_offset_test) as filepath_or_buffer, open(input_file['path'], 'r') as input_file_check:
                input_file_check.seek(seek_offset_test)

                # check every 10th line for the first 100 lines
                for i in range(0, 10):    
                    for i in range(0, 9):
                        filepath_or_buffer.readline()
                        input_file_check.readline()

                    self.assertEqual(filepath_or_buffer.readline(), input_file_check.readline())
                    
            # check that file was properly closed
            self.assertEqual(filepath_or_buffer.closed, input_file_check.closed)

    def test_pass_in_BytesIO_buffer(self):
        """
        Make sure FileOrBufferHandler can take BytesIO and read it similarly to open() 
        """
        for input_file in self.input_file_names:
            with FileOrBufferHandler(BytesIO(open(input_file['path'], 'rb').read())) as filepath_or_buffer, open(input_file['path'], 'rb') as input_file_check:
                # check every 10th line for the first 100 lines
                for i in range(0, 10):    
                    for i in range(0, 9):
                        filepath_or_buffer.readline()
                        input_file_check.readline()

                    self.assertEqual(filepath_or_buffer.readline(), input_file_check.readline())
                    
            # check that file was properly closed
            self.assertEqual(filepath_or_buffer.closed, input_file_check.closed)

    def test_pass_in_BytesIO_seek_buffer(self):
        """
        Make sure FileOrBufferHandler can take BytesIO with seek and read it similarly to open() with seek
        """
        for input_file in self.input_file_names:
            seek_offset_test = 100
            with FileOrBufferHandler(BytesIO(open(input_file['path'], 'rb').read()), seek_offset=seek_offset_test) as filepath_or_buffer, open(input_file['path'], 'rb') as input_file_check:
                input_file_check.seek(seek_offset_test)
                
                # check every 10th line for the first 100 lines
                for i in range(0, 10):    
                    for i in range(0, 9):
                        filepath_or_buffer.readline()
                        input_file_check.readline()

                    self.assertEqual(filepath_or_buffer.readline(), input_file_check.readline())
                    
            # check that file was properly closed
            self.assertEqual(filepath_or_buffer.closed, input_file_check.closed)

if __name__ == '__main__':
    unittest.main()
