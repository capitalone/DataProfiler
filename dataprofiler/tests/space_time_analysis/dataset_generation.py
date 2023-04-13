import string
from typing import List

import numpy as np
import pandas as pd
from numpy.random import Generator
try:
    import sys

    sys.path.insert(0, "../../..")
    import dataprofiler as dp
except ImportError:
    import dataprofiler as dp


def convert_data_to_df(np_data: np.array, path: str=None) -> pd.DataFrame:
    """
    Converts np array to a pandas dataframe

    :param np_data: np array to be converted
    :type np_data: numpy array
    :param path: path to output a csv of the dataframe generated
    :type path: str
    :return: a pandas dataframe
    """
    # convert array into dataframe
    dataframe = pd.DataFrame(np_data)

    # save the dataframe as a csv file
    if path:
        dataframe.to_csv(path)
        print(f"Created {path}!")
    return dataframe


def random_integers(rng: Generator, min_value: int=-1e6, max_value: int=1e6,
                    num_rows: int=1) -> np.array:
    """
    Randomly generates an array of integers between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min_value: the minimum integer that can be returned
    :type min_value: int
    :param max_value: the maximum integer that can be returned
    :type max_value: int
    :param num_rows: the number of rows in np array generated
    :type num_rows: int

    :return: np array of integers
    """
    return rng.integers(min_value, max_value, (num_rows,))


def random_floats(rng: Generator, min_value: int=-1e6, max_value: int=1e6,
                  sig_figs: int=3, num_rows: int=1) -> np.array:
    """
    Randomly generates an array of floats between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min_value: the minimum float that can be returned
    :type min_value: int
    :param max_value: the maximum float that can be returned
    :type max_value: int
    :param sig_figs: restricts float to a number of sig_figs after decimal
    :type sig_figs: int
    :param num_rows: the number of rows in np array generated
    :type num_rows: int

    :return: np array of floats
    """
    return round(rng.random(min_value, max_value, (num_rows,)), sig_figs)


def random_string(rng: Generator, categories: List[str]=None, num_rows: int=1,
                  str_len_min: int=1, str_len_max: int=256) -> np.array:
    """
    Randomly generates an array of strings with length between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param categories: a list of values that are allowed in a string or None
    :type categories: List[str], None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int
    :param str_len_min: the minimum length a string can be
    :type str_len_min: int
    :param str_len_max: the maximum length a string can be
    :type str_len_max: int


    :return: numpy array of strings
    """
    if categories is None:
        chars = list(string.ascii_uppercase + string.ascii_lowercase + \
                string.digits + " " + string.punctuation)
    string_list = []

    for _ in range(num_rows):
        length = rng.integers(str_len_min, str_len_max)
        string_entry = "".join(rng.choice(chars, (length,)))
        string_list.append(string_entry)

    return np.array(string_list)


def generate_datetime(rng: Generator, date_format: str, start_date: str=None,
                      end_date: str=None) -> str:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, optional
    :param start_date: the latest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, optional

    :return: generated datetime
    :rtype: str
    """
    if not start_date:
        # 100 years in past
        start_date = pd.Timestamp(1920, 1, 1)
    if not end_date:
        # protection of 30 years in future
        end_date = pd.Timestamp(2049, 12, 31)
    t = rng.random()
    ptime = start_date + t * (end_date - start_date)

    return ptime.strftime(date_format)


def random_datetimes(rng: Generator, date_format_list: str=None,
                     start_date: str=None, end_date: str=None,
                     num_rows: int=1) -> np.array:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, optional
    :param start_date: the latest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, optional

    :return: array of generated datetimes
    :rtype: numpy array
    """
    date_list = []
    if not date_format_list:
        date_format_list = ["%B %d %Y %H:%M:%S"]

    for _ in range(num_rows):
        date_format = rng.choice(date_format_list)
        datetime = generate_datetime(rng, date_format=date_format,
                                     start_date=start_date, end_date=end_date)
        date_list.append(datetime)

    return np.array(date_list)


def random_categorical(rng: Generator, categories: List[str]=None,
                       num_rows: int=1) -> np.array:
    """
    Randomly generates an array of categorical chosen out of categories

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param categories: a list of values that are allowed in a categorical or None
    :type categories: string, None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int

    :return: np array of categories
    """
    if categories is None:
        categories = ["A", "B", "C", "D", "E"]

    return rng.choice(categories, (num_rows,))


def get_ordered_column(start: int=0, num_rows: int=1, **kwarg):
    """
    Generates an array of ordered integers

    :param start: integer that the ordered list should start at
    :type str_len_min: int
    :param num_rows: the number of rows in np array generated
    :type num_rows: int

    :return: np array of ordered integers
    """
    return np.array(list(range(start, num_rows)))


def generate_dataset_by_class(rng: Generator, classes_to_generate: List[str]=None,
                              dataset_length: int=100000,
                              path: str=None) -> pd.DataFrame:
    """
    Randomly a dataset with a mixture of different data classes

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param classes_to_generate: Classes of data to be included in the dataset
    :type classes_to_generate: List[str] or None
    :param dataset_length: length of the dataset generated
    :type dataset_length: int
    :param path: path to output a csv of the dataframe generated
    :type path: str

    :return: pandas DataFrame
    """
    possible_classes = ["text", "string", "categorical", "integer", "float",
                        "ordered", "datetime"]
    if classes_to_generate is None:
        classes_to_generate = possible_classes

    dataset = []
    if "integer" in classes_to_generate:
        dataset.append(random_integers(rng, num_rows=dataset_length))
    if "float" in classes_to_generate:
        dataset.append(random_integers(rng, num_rows=dataset_length))
    if "datetime" in classes_to_generate:
        dataset.append(random_datetimes(
            rng,
            date_format_list=dp.profilers.DateTimeColumn._date_formats,
            num_rows=dataset_length
        ))
    if "string" in classes_to_generate:
        dataset.append(random_string(rng, num_rows=dataset_length))
    if "categorical" in classes_to_generate:
        dataset.append(random_categorical(rng, num_rows=dataset_length))
    if "text" in classes_to_generate:
        dataset.append(random_string(rng, num_rows=dataset_length,
                                     str_len_min=256, str_len_max=1000))
    if "ordered" in classes_to_generate:
        dataset.append(get_ordered_column(num_rows=dataset_length))

    for cl in classes_to_generate:
        if cl not in possible_classes:
            print(f"Class: {cl} is not in possible class list and is not "
                  f"included in the dataset")

    return convert_data_to_df(dataset, path)
