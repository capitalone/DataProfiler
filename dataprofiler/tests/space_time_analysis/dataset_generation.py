import string

import numpy as np
import pandas as pd

try:
    import sys

    sys.path.insert(0, "../../..")
    import dataprofiler as dp
except ImportError:
    import dataprofiler as dp


def convert_data_to_df(np_data, path=None):
    # convert array into dataframe
    dataframe = pd.DataFrame(np_data)

    # save the dataframe as a csv file
    if path:
        dataframe.to_csv(path)
        print(f"Created {path}!")
    return dataframe


def random_integers(rng, min_value=-1e6, max_value=1e6, num_rows=1):
    """
    Randomly generates an integer between a min and max value
    :param min_value: the minimum integer that can be returned
    :param max_value: the maximum integer that can be returned

    :return: integer
    """
    return rng.integers(min_value, max_value, (num_rows,))


def random_floats(rng, min_value=-1e6, max_value=1e6, sig_figs=3, num_rows=1):
    """
    Randomly generates an float between a min and max value
    :param min_value: the minimum float that can be returned
    :param max_value: the maximum float that can be returned
    :param sig_figs: restricts float to a number of sig_figs after decimal

    :return: float
    """
    return round(rng.random(min_value, max_value, (num_rows,)), sig_figs)


def random_string(rng, categories=None, num_rows=1,
                  str_len_min=1, str_len_max=256):

    if categories is None:
        chars = list(string.ascii_uppercase + string.ascii_lowercase + \
                string.digits + " " + string.punctuation)
    string_list = []

    for _ in range(num_rows):
        length = rng.integers(str_len_min, str_len_max)
        string_entry = "".join(rng.choice(chars, (length,)))
        string_list.append(string_entry)

    return np.array(string_list)


def generate_datetime(rng, date_format, start_date=None, end_date=None):
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param random_state: a random state that is used to randomly generate the entity
    :type random_state: Union[random.Random, numbers.Integral]
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


def random_datetimes(rng, date_format_list=None, start_date=None, end_date=None, num_rows=1):
    date_list = []
    if not date_format_list:
        date_format_list = ["%B %d %Y %H:%M:%S"]

    for _ in range(num_rows):
        date_format = rng.choice(date_format_list)
        datetime = generate_datetime(rng, date_format=date_format,
                                     start_date=start_date, end_date=end_date)
        date_list.append(datetime)

    return np.array(date_list)


def random_categorical(rng, categories=None, num_rows=1):

    if categories is None:
        categories = ["A", "B", "C", "D", "E"]

    return rng.choice(categories, (num_rows,))


def get_ordered_column(start=0, num_rows=1, **kwarg):
    return np.array(list(range(start, num_rows)))


def generate_dataset_by_class(rng, classes_to_generate=None,
                              dataset_length=100000, path=None):
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
