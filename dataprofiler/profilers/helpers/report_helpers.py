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


def _prepare_report(report, output_format=None):
    if output_format:
        output_format = output_format.lower()
    if not output_format or output_format not in ['pretty', 'serializable', 'flat']:
        return report
    report = report.copy()
    max_str_len = 50
    max_array_len = 5
    for key in report:
        if isinstance(report[key], dict):
            report[key] = _prepare_report(report[key], output_format=output_format)
        elif isinstance(report[key], list) or isinstance(report[key], np.ndarray):
            if output_format == "pretty":
                if isinstance(report[key], list):
                    report[key] = np.array(report[key])
                str_value = np.array2string(report[key], separator=', ')
                if len(str_value) > max_str_len and len(report[key]) > max_array_len:
                    ind = 1
                    str_value = ''
                    while len(str_value) <= max_str_len:
                        str_value = \
                            np.array2string(report[key][:ind], separator=', ')[:-1] + \
                            ', ... , ' + \
                            np.array2string(
                                report[key][-ind:], separator=', ')[1:]
                        ind += 1
                report[key] = str_value
            elif output_format == "serializable" and isinstance(report[key], np.ndarray):
                report[key] = report[key].tolist()
        elif output_format == "pretty" and isinstance(report[key], float):
            report[key] = round(report[key], 4)
    if output_format == 'flat':
        report = flat_dict(report)
    return report
