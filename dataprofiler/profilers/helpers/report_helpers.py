import math

import numpy as np


def calculate_quantiles(num_quantile_groups, quantiles):
    len_quant = len(quantiles)
    if not (num_quantile_groups and 0 < num_quantile_groups <= len_quant):
        num_quantile_groups = 4
    quant_multiplier = len_quant/num_quantile_groups
    # quantile is one less than group
    # Goes from zero (inclusive) to number of groups (exclusive)
    # +1 because 0 + 1 * multiplier = correct first quantile
    # -1 because 0 index
    # i.e. quantile:
    # quant_multiplier = 1000 / 4 = 250
    # [0 + 1] * (quant_multiplier) - 1 = 1 * 250 - 1 = 249 (first quantile)
    quantiles = {
        ind: quantiles[math.ceil((ind + 1) * quant_multiplier) - 1]
        for ind in range(num_quantile_groups - 1)
    }
    return quantiles


def flat_dict(od, separator='_', key=''):
    """
    Function to flatten nested dictionary. Each level is collapsed and joined with the specified seperator.

    :param od: dictionary or dictionary-like object
    :type od: dict
    :param seperator: character(s) joining successive levels
    :type seperator: str
    :param key: concatenated keys
    :type key: str
    :returns: unnested dictionary
    :rtype: dict
    """
    return {str(key).replace(' ','_') + separator + str(k) if key else k : v
                for kk, vv in od.items()
                    for k, v in flat_dict(vv, separator, kk).items()
           } if isinstance(od, dict) else {key:od}


def _prepare_report(report, output_format=None, omit_keys=[]):
    """
    Prepares report dictionary for users upon request

    :param report: contains the values identified from the profile
    :type report: dict()
    :param output_format: designation for how to format the returned report
    :type output_format: dict()
    :param omit_keys: Keys to omit from the output report
    :type omit_keys: list(str)
    :return report: handle to the updated report
    :type report: dict()
    """
    
    format_options = ['pretty', 'serializable', 'flat']
    if output_format:
        output_format = output_format.lower()
    if not output_format or output_format not in format_options:
        return report
    
    fmt_report = report.copy()
    max_str_len = 50
    max_array_len = 5
    for key in fmt_report:
        value = fmt_report[key]
        if isinstance(value, dict):
            fmt_report[key] = _prepare_report(
                fmt_report[key], output_format=output_format)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            if output_format == "pretty":
                if isinstance(value, list):
                    fmt_report[key] = np.array(value)
                str_value = np.array2string(value, separator=', ')
                if len(str_value) > max_str_len and len(value) > max_array_len:
                    ind = 1
                    str_value = ''
                    while len(str_value) <= max_str_len:
                        str_value = \
                            np.array2string(value[:ind], separator=', ')[:-1] + \
                            ', ... , ' + \
                            np.array2string(
                                value[-ind:], separator=', ')[1:]
                        ind += 1
                fmt_report[key] = str_value
            elif output_format == "serializable" and isinstance(value, np.ndarray):
                fmt_report[key] = value.tolist()
        elif output_format == "pretty" and isinstance(value, float):
            fmt_report[key] = round(fmt_report[key], 4)

        # Remove any keys omitted
        if key in omit_keys:
            fmt_report.pop(key, None)
            
    if output_format == 'flat':
        fmt_report = flat_dict(fmt_report)
        
    return fmt_report
