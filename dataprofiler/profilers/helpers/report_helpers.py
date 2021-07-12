import math

import numpy as np


def calculate_quantiles(num_quantile_groups, quantiles):
    len_quant = len(quantiles)
    if not (num_quantile_groups and 0 < num_quantile_groups <= (len_quant + 1)):
        num_quantile_groups = 4
    quant_multiplier = (len_quant + 1) / num_quantile_groups
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
    Function to flatten nested dictionary. Each level is collapsed and 
    joined with the specified seperator.

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


def _prepare_report(report, output_format=None, omit_keys=None):
    """
    Prepares report dictionary for users upon request.

    output_format options: 

    - Pretty: floats are rounded to four decimal places & lists are shortened.
    - Compact: Similar to pretty, but removes detailed statistics such as 
               runtimes, label probabilities, index locations of null types
    - Serializable: Output is json serializable and not prettified
    - Flat: Nested output is returned as a flattened dictionary

    :param report: contains the values identified from the profile
    :type report: dict()
    :param output_format: designation for how to format the returned report;
                          possible options: pretty, serializable, flat, compact
    :type output_format: dict()
    :param omit_keys: Keys to omit from the output report, to omit keys in the 
                      report a '.' represents a level of recursion example: 
                      report: { 'test1': { 'test2': val, 'test3': val }, 
                      to omit key 'test3' from report: omit_keys=['test1.test3']
                      wildcards are also possible, so: omit_keys=['*.test3']
    :type omit_keys: list(str)
    :return report: handle to the updated report
    :type report: dict()
    """

    if output_format is not None:
        output_format = output_format.lower()
    if omit_keys is None:
        omit_keys = []

    fmt_report = {}
    max_str_len = 50
    max_array_len = 5

    if output_format == 'compact':
        omit_keys.extend([
            "data_stats.*.statistics.times",
            "data_stats.*.statistics.avg_predictions",
            "data_stats.*.statistics.data_label_representation",
            "data_stats.*.statistics.null_types_index",
            "data_stats.*.statistics.histogram"
        ])
        output_format = "pretty"
    
    for key in report:
        
        # Remove any keys omitted
        if key in omit_keys:
            continue
        
        value = report[key]

        # Convert set to list, for report generation
        if isinstance(value, set):
            value = sorted(list(value))

        if "*" in omit_keys:
            continue

        # For data_stats (in structured case), need to recurse through a list
        elif key == "data_stats" and isinstance(value, list) \
                and "data_stats" not in omit_keys:

            fmt_report["data_stats"] = []

            for col_ind in range(len(value)):
                col_name = str(value[col_ind].get('column_name'))

                # update omit keys
                next_layer_omit_keys = []
                is_omitted_col = False
                for omit_key in omit_keys:

                    # Omit this column
                    if omit_key in {f"*.{col_name}", "data_stats.*",
                                    f"data_stats.{col_name}"}:
                        fmt_report["data_stats"].append(None)
                        is_omitted_col = True
                        break

                    # Skip this omit_key if it doesn't involve data_stat cols
                    omit_key_split = omit_key.split('.', 1)
                    if len(omit_key_split) == 1 \
                            or omit_key_split[0] not in {"data_stats", "*"}:
                        continue

                    next_key_split = omit_key_split[1].split('.', 1)
                    if next_key_split[0] in {"*", col_name}:
                        next_layer_omit_keys.append(next_key_split[1])

                # update report and list for column we are keeping
                if not is_omitted_col:
                    fmt_report["data_stats"].append(
                        _prepare_report(value[col_ind], output_format,
                                        next_layer_omit_keys))

        # Do not recurse or modify profile_schema
        elif key == "profile_schema" and "profile_schema" not in omit_keys:
            fmt_report[key] = value

        elif isinstance(value, dict):

            # split off any remaining keys for the recursion
            # i.e. [test0, test1.test2] -> omit_keys => [test1.test2]
            next_layer_omit_keys = []
            for omit_key in omit_keys:
                omit_key_split = omit_key.split('.', 1)
                
                # Must have more keys left for recursion 
                if len(omit_key_split) > 1: 
                    next_key_layer = omit_key_split[-1]
                    prior_key_layer = omit_key_split[0]
                    if len(next_key_layer) > 0:
                        if prior_key_layer == '*' or prior_key_layer == key:
                            next_layer_omit_keys.append(next_key_layer)

            # Recursively add keys to the final report
            fmt_report[key] = _prepare_report(value, output_format,
                                              next_layer_omit_keys)
            
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            
            if output_format == "pretty":
                
                if isinstance(value, list):
                    value = np.array(value)
                    
                str_value = np.array2string(value, separator=', ')
                
                if len(str_value) > max_str_len and len(value) > max_array_len:
                    ind = 1
                    str_value = ''
                    while len(str_value) <= max_str_len:
                        str_value = \
                            np.array2string(value[:ind], separator=', ')[:-1] + \
                            ', ... , ' + \
                            np.array2string(value[-ind:], separator=', ')[1:]
                        ind += 1
                        
                fmt_report[key] = str_value
                
            elif output_format == "serializable" and isinstance(value, np.ndarray):
                fmt_report[key] = value.tolist()
            else:
                fmt_report[key] = value
                
        elif isinstance(value, float) and output_format == "pretty":
            fmt_report[key] = round(value, 4)
        else:
            fmt_report[key] = value
            
    if output_format == 'flat':
        fmt_report = flat_dict(fmt_report)

    return fmt_report
