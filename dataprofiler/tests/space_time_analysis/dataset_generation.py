import copy
import json
import string
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator

try:
    import sys

    sys.path.insert(0, "../../..")
    import dataprofiler as dp
except ImportError:
    import dataprofiler as dp


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def nan_injection(
    rng: Generator, df: pd.DataFrame, percent_to_nan: float = 0.0
) -> pd.DataFrame:
    """
    Inject NAN values into a dataset based on percentage global (PERCENT_TO_NAN)

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param df: DataFrame that is to be injected with NAN values
    :type df: pandas.DataFrame
    :param df: DataFrame that is to be injected with NAN values
    :type df: pandas.DataFrame
    :param percent_to_nan: Percentage of dataset that needs to be nan values
    :type percent_to_nan: float, optional

    :return: New DataFrame with injected NAN values
    """
    samples_to_nan = int(len(df) * percent_to_nan / 100)
    for col_name in df:
        ind_to_nan = rng.choice(list(df.index), samples_to_nan)
        df[col_name][ind_to_nan] = "None"
    return df


def convert_data_to_df(
    np_data: np.array,
    path: Optional[str] = None,
    index: bool = False,
    column_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts np array to a pandas dataframe

    :param np_data: np array to be converted
    :type np_data: numpy array
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional
    :param index: whether to include index in output to csv
    :type path: bool, optional
    :param column_names: The names of the columns of a dataset
    :type path: List, None, optional
    :return: a pandas dataframe
    """
    # convert array into dataframe
    if not column_names:
        column_names = [x for x in range(len(np_data))]
    dataframe = pd.DataFrame.from_dict(dict(zip(column_names, np_data)))
    # save the dataframe as a csv file
    if path:
        dataframe.to_csv(path, index=index, encoding="utf-8")
        print(f"Created {path}!")
    return dataframe


def random_integers(
    rng: Generator, min_value: int = -1e6, max_value: int = 1e6, num_rows: int = 1
) -> np.array:
    """
    Randomly generates an array of integers between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min_value: the minimum integer that can be returned
    :type min_value: int, optional
    :param max_value: the maximum integer that can be returned
    :type max_value: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of integers
    """
    return rng.integers(min_value, max_value, (num_rows,))


def random_floats(
    rng: Generator,
    min_value: int = -1e6,
    max_value: int = 1e6,
    sig_figs: int = 3,
    num_rows: int = 1,
) -> np.array:
    """
    Randomly generates an array of floats between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min_value: the minimum float that can be returned
    :type min_value: int, optional
    :param max_value: the maximum float that can be returned
    :type max_value: int, optional
    :param sig_figs: restricts float to a number of sig_figs after decimal
    :type sig_figs: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of floats
    """
    return np.around(rng.uniform(min_value, max_value, num_rows), sig_figs)


def random_string(
    rng: Generator,
    chars: Optional[List[str]] = None,
    num_rows: int = 1,
    str_len_min: int = 1,
    str_len_max: int = 256,
) -> np.array:
    """
    Randomly generates an array of strings with length between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param chars: a list of values that are allowed in a string or None
    :type chars: List[str], None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional
    :param str_len_min: the minimum length a string can be
    :type str_len_min: int, optional
    :param str_len_max: the maximum length a string can be
    :type str_len_max: int, optional

    :return: numpy array of strings
    """
    if chars is None:
        chars = list(
            string.ascii_uppercase
            + string.ascii_lowercase
            + string.digits
            + " "
            + string.punctuation
        )
    string_list = []

    for _ in range(num_rows):
        length = rng.integers(str_len_min, str_len_max)
        string_entry = "".join(rng.choice(chars, (length,)))
        string_list.append(string_entry)

    return np.array(string_list)


def generate_datetime(
    rng: Generator,
    date_format: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, None, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional
    :param start_date: the latest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional

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


def random_datetimes(
    rng: Generator,
    date_format_list: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_rows: int = 1,
) -> np.array:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, None, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional
    :param start_date: the latest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional

    :return: array of generated datetimes
    :rtype: numpy array
    """
    date_list = []
    if not date_format_list:
        date_format_list = ["%B %d %Y %H:%M:%S"]

    for _ in range(num_rows):
        date_format = rng.choice(date_format_list)
        datetime = generate_datetime(
            rng, date_format=date_format, start_date=start_date, end_date=end_date
        )
        date_list.append(datetime)

    return np.array(date_list)


def random_categorical(
    rng: Generator, categories: Optional[List[str]] = None, num_rows: int = 1
) -> np.array:
    """
    Randomly generates an array of categorical chosen out of categories

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param categories: a list of values that are allowed in a categorical or None
    :type categories: string, None, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of categories
    """
    if categories is None:
        categories = ["A", "B", "C", "D", "E"]

    return rng.choice(categories, (num_rows,))


def get_ordered_column(start: int = 0, num_rows: int = 1, **kwarg) -> np.array:
    """
    Generates an array of ordered integers

    :param start: integer that the ordered list should start at
    :type str_len_min: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of ordered integers
    """
    return np.arange(start, start + num_rows)


def random_text(
    rng: Generator,
    chars: Optional[str] = None,
    num_rows: int = 1,
    str_len_min: int = 256,
    str_len_max: int = 1000,
) -> np.array:
    """
    Randomly generates an array of text with length between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param chars: a list of values that are allowed in a string or None
    :type chars: List[str], None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional
    :param str_len_min: the minimum length a string can be (must be larger than 256)
    :type str_len_min: int, optional
    :param str_len_max: the maximum length a string can be
    :type str_len_max: int, optional

    :return: numpy array of text
    """
    if str_len_min < 256:
        raise ValueError(
            f"str_len_min must be > 256. " f"Value provided: {str_len_min}."
        )

    return random_string(rng, chars, num_rows, str_len_min, str_len_max)


def generate_dataset_by_class(
    rng: Generator,
    columns_to_generate: Optional[List[dict]] = None,
    dataset_length: int = 100000,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Randomly a dataset with a mixture of different data classes

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param columns_to_generate: Classes of data to be included in the dataset
    :type columns_to_generate: List[dict], None, optional
    :param dataset_length: length of the dataset generated
    :type dataset_length: int, optional
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional

    :return: pandas DataFrame
    """
    gen_funcs = {
        "integer": random_integers,
        "float": random_floats,
        "categorical": random_categorical,
        "ordered": get_ordered_column,
        "text": random_text,
        "datetime": random_datetimes,
        "string": random_string,
    }

    if columns_to_generate is None:
        columns_to_generate = [
            dict(generator="datetime"),
            dict(generator="integer"),
            dict(generator="float"),
            dict(generator="categorical"),
            dict(generator="ordered"),
            dict(generator="text"),
            dict(generator="string"),
        ]

    dataset = []
    for col in columns_to_generate:
        col_ = copy.deepcopy(col)
        col_generator = col_.pop("generator")
        if col_generator not in gen_funcs:
            raise ValueError(f"generator: {col_generator} is not a valid generator.")

        col_generator_function = gen_funcs.get(col_generator)
        dataset.append(col_generator_function(**col_, num_rows=dataset_length, rng=rng))
    return convert_data_to_df(dataset, path)


if __name__ == "__main__":
    # Params
    random_seed = 0
    GENERATED_DATASET_SIZE = 100000
    rng = np.random.default_rng(seed=random_seed)
    CLASSES_TO_GENERATE = [
        dict(
            generator="datetime", date_format_list=None, start_date=None, end_date=None
        ),
        dict(generator="integer", min_value=-1e6, max_value=1e6),
        dict(generator="float", min_value=-1e6, max_value=1e6, sig_figs=3),
        dict(generator="categorical", categories=None),
        dict(generator="ordered", start=0),
        dict(generator="text", chars=None, str_len_min=256, str_len_max=1000),
        dict(generator="string", chars=None, str_len_min=1, str_len_max=256),
    ]
    output_path = (
        f"data/seed_{random_seed}_"
        f"{'all' if CLASSES_TO_GENERATE is None else 'subset'}_"
        f"size_{GENERATED_DATASET_SIZE}.csv"
    )

    # Generate dataset
    data = generate_dataset_by_class(
        rng,
        columns_to_generate=CLASSES_TO_GENERATE,
        dataset_length=GENERATED_DATASET_SIZE,
        path=output_path,
    )
