#!/usr/bin/env python
"""
coding=utf-8

Build model for a dataset by identifying type of column along with its
respective parameters.
"""
from __future__ import print_function
from __future__ import division


def is_in_range(x, config):
    """
    Checks to see x is in the range of the config.

    :param x: number
    :type x: int/float
    :param config: configuration
    :type config: dict
    :returns: bool
    """
    try:
        return config['start'] <= float(x) <= config['end']
    except:
        raise TypeError("Value is not a float")


def is_in_list(x, config):
    """
    Checks to see x is in the config list.

    :param x: item
    :type x: string
    :param config: configuration
    :type config: dict
    :returns: bool
    """
    return float(x) in config


class Validator():

    def __init__(self):
        self.config = None
        self.report = None
        self.validation_run = False
        self.validation_report = dict()

    def validate(self, data, config):
        """
        Validate a data set. No option for validating a partial data set.

        Set configuration on run not on instantiation of the class such that
        you have the option to run multiple times with different configurations
        without having to also reinstantiate the class.

        :param data: The data to be processed by the validator. Processing
                occurs in a column-wise fashion.
        :type data: DataFrame Dask/Pandas
        :param config: configuration for how the validator should
                run across the given data. Validator will only run over columns
                specified in the configuration.
        :type config: dict

        :Example:
        
            This is an example of the config::
            
                config = {
                        <column_name>: {
                                range: {
                                    'start': 1,
                                    'end':2
                                },
                                list: [1,2,3]
                            }
                        }
        """
        if not config:
            raise ValueError("Config is required")

        known_anomaly_validation = config.get('known_anomaly_validation', {})
        for iter_key, value in known_anomaly_validation.items():
            if len(value) < 1:
                raise Warning(
                    f"Pass at a minimum one value for a specified column "
                    f"(i.e. iter_key variable) -- not both for {iter_key}")

        self.config = config

        df_type = config.get('df_type', '').lower()

        for iter_key, value in known_anomaly_validation.items():
            self.validation_report[iter_key] = dict()
            df_series = data[iter_key]
            for sub_key, sub_value in value.items():
                self.validation_report[iter_key][sub_key] = dict()

                if sub_key not in ["range", "list"]:
                    raise TypeError(
                        'Range and list only acceptable key values.')
                apply_type = is_in_range if sub_key == "range" else is_in_list

                if df_type == 'dask':
                    temp_results = df_series.apply(
                        apply_type,
                        meta=(iter_key, 'bool'),
                        args=(sub_value,))
                    temp_results = temp_results.compute()
                    # Dask evaluates this to be an nd array so we have to
                    # convert it to a normal list
                    self.validation_report[iter_key][sub_key] = [
                        idx for idx, val
                        in enumerate(temp_results.values.tolist()) if val
                    ]
                elif df_type == 'pandas':
                    temp_results = df_series.apply(
                        apply_type,
                        args=(sub_value,)
                    )
                    self.validation_report[iter_key][sub_key] = [
                        idx for idx, val in temp_results.items() if val
                    ]
                else:
                    raise ValueError(
                        'Dask and Pandas are the only supported dataframe '
                        'types.')
                del temp_results
        self.validation_run = True

    def get(self):
        """
        Get the results of the validation run.
        """
        if self.validation_run:
            return self.validation_report

        else:
            raise Warning(
                "Precondition for get method not met. Must validate data prior "
                "to getting results.")
