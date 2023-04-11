import string

import numpy as np
import pandas as pd

import dataprofiler as dp


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


def random_text(rng, categories=None, num_rows=1):

    if categories is None:
        chars = string.ascii_uppercase + string.ascii_lowercase + \
                string.digits + " " + string.punctuation

    text_list = []

    for _ in range(num_rows):
        length = rng.integers(256, 1000)
        text_entry = "".join(rng.choice(chars, (length,)))
        text_list.append(text_entry)

    return np.array(text_list)


def random_string(rng, categories=None, num_rows=1):

    if categories is None:
        chars = list(string.ascii_uppercase + string.ascii_lowercase + \
                string.digits + " " + string.punctuation)
    string_list = []

    for _ in range(num_rows):
        length = rng.integers(1, 256)
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


def convert_data_to_csv(np_data, path="test.csv"):
    # convert array into dataframe
    dataframe = pd.DataFrame(np_data)

    # save the dataframe as a csv file
    dataframe.to_csv(path)
    print(f"Created {path}!")


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    for dataset_length in [100, 1000, 100000]:
        # Type 1
        data_list = []
        int_data = random_integers(rng, num_rows=dataset_length)
        data_list.append(int_data)
        convert_data_to_csv(int_data, f"output_data/int_{dataset_length}.csv")
        float_data = random_integers(rng, num_rows=dataset_length)
        data_list.append(float_data)
        convert_data_to_csv(float_data, f"output_data/float_{dataset_length}.csv")
        datetime_data = random_datetimes(
            rng,
            date_format_list=dp.profilers.DateTimeColumn._date_formats,
            num_rows=dataset_length
        )
        data_list.append(datetime_data)
        convert_data_to_csv(datetime_data, f"output_data/datetime_{dataset_length}.csv")
        string_data = random_string(rng, num_rows=dataset_length)
        data_list.append(string_data)
        convert_data_to_csv(string_data, f"output_data/string_{dataset_length}.csv")
        cat_data = random_categorical(rng, num_rows=dataset_length)
        data_list.append(cat_data)
        convert_data_to_csv(cat_data, f"output_data/cat_{dataset_length}.csv")
        ordered_data = get_ordered_column(num_rows=dataset_length)
        data_list.append(ordered_data)
        convert_data_to_csv(ordered_data, f"output_data/ordered_{dataset_length}.csv")

        #Type 2
        index = 0
        while index < len(data_list):
            pointer = index + 1
            while pointer < len(data_list):
                mixed_class_data = np.column_stack((data_list[index], data_list[pointer]))
                convert_data_to_csv(
                    mixed_class_data,
                    f"output_data/mixed_class_data_{index}_{pointer}_{dataset_length}.csv"
                )
                pointer += 1
            index += 1





