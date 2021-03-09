import os
import unittest

import pandas as pd

from dataprofiler.data_readers.data import Data, CSVData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCSVDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            # dict(path=os.path.join(test_dir, 'csv/diamonds.csv'),
            #      count=1000, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/iris.csv'),
            #      count=150, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/iris-utf-8.csv'),
            #      count=150, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/iris-utf-16.csv'),
            #      count=150, delimiter=',', encoding='utf-16'),
            # dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.parquet'),
            #      count=150, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.txt'),
            #      count=150, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.json'),
            #      count=150, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/guns.csv'),
            #      count=1316, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/wisconsin_cancer_train.csv'),
            #      count=25, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/aws_honeypot_marx_geo.csv'),
            #      count=25, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num.csv'),
                 count=5, delimiter=None, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col.txt'),
                 count=5, delimiter=None, encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/names-col-empty.txt'),
            #      count=5, delimiter=None, encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/log_data_long.txt'),
            #      count=753, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/sparse-last-column.txt'),
            #      count=5, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/sparse-first-column.txt'),
            #      count=5, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column.txt'),
            #      count=5, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/log_data_sparse.txt'),
            #      count=20, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/log_data_super_sparse.txt'),
            #      count=20, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/sparse-columns-test.csv'),
            #      count=20, delimiter=',', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/sentence-4x.txt'),
            #      count=4, delimiter='.', encoding='utf-8'),
            # dict(path=os.path.join(test_dir, 'csv/quote-test.txt'),
            #      count=10, delimiter=' ', encoding='utf-8'),
        ]
        cls.output_file_path = None

    def test_auto_file_identification(self):
        """
        Determine if the csv file can be automatically identified
        """
        for input_file in self.input_file_names:
            input_data_obj = CSVData(input_file['path'])
            #input_data_obj = CSVData(input_file['path'], options={'delimiter': ',', 'header': 0})
            print(input_file)
            print(input_data_obj.data)