import fastavro

from . import data_utils
from .base_data import BaseData
from .json_data import JSONData


class AVROData(JSONData, BaseData):
    """
    AVROData class to save and load spreadsheet data
    """

    data_type = 'avro'

    def __init__(self, input_file_path=None, data=None, options=None):
        """
        Data class for loading datasets of type AVRO. Can be specified by
        passing in memory data or via a file path. Options pertaining the AVRO
        may also be specified using the options dict parameter.
        Possible Options::
        
            options = dict(
                data_format= type: str, choices: "dataframe", "records", "avro"
                selected_keys= type: list(str)
            )
            
        data_format: user selected format in which to return data can only be of specified types
        selected_keys: keys being selected from the entire dataset

        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        JSONData.__init__(self, input_file_path, data, options)

    def _load_data_from_file(self, input_file_path):
        with open(input_file_path, "rb") as input_file:
            # Currently, string reading with 'r' option has the unicode issue,
            # even when the option encoding='utf-8' is added. It may come from
            # some special compression codec, e.g., snappy. Then, binary mode
            # reading is currently used to get the dict-formatted lines.
            df_reader = fastavro.reader(input_file)
            lines = list()
            for line in df_reader:
                lines.append(line)
            return lines

    @classmethod
    def is_match(cls, file_path, options=None):
        """
        Test the given file to check if the file has valid
        AVRO format or not.
        
        :param file_path: path to the file to be examined
        :type file_path: str
        :param options: avro read options
        :type options: dict
        :return: is file a avro file or not
        :rtype: bool
        """
        if options is None:
            options = dict()

        is_valid_avro = fastavro.is_avro(file_path)
        return is_valid_avro

    @classmethod
    def _get_nested_key(cls, dict_line, nested_key):
        """
        Update nested keys from a dictionary and the current nested key.
        Example of output:
            {'name': 1, 'favorite_number': 1, 'favorite_color': 1,
            'address': {'streetaddress': 1, 'city': 1}}
            
        :param dict_line: dictionary that may contain nested keys
        :type dict_line: dict
        :param nested_key: the current nested keys that needs to be updated
        :type nested_key: dict
        :return: a dictionary containing nested keys
        """
        for key, value in dict_line.items():
            if key in nested_key:
                if type(value) is dict:
                    nested_key[key] = cls._get_nested_key(
                                                value, nested_key[key])
            else:
                if type(value) is dict:
                    nested_key[key] = {}
                    nested_key[key] = cls._get_nested_key(
                                                value, nested_key[key])
                else:
                    nested_key[key] = 1

        return nested_key

    @classmethod
    def _get_nested_keys_from_dicts(cls, dicts):
        """
        Extract nested keys from a list of dictionaries. Example of output:
        {'name': 1, 'favorite_number': 1, 'favorite_color': 1,
        'address': {'streetaddress': 1, 'city': 1}}
        
        :param dicts: list of dictionaries
        :type dicts: list(dict)
        :return: a dictionary containing nested keys
        """
        nested_keys = {}
        for dict_line in dicts:
            nested_keys = cls._get_nested_key(dict_line, nested_keys)
        return nested_keys

    @classmethod
    def _get_schema_avro(cls, nested_keys, schema_avro):
        """
        Update avro schema from the nested keys and the current avro schema
        
        :param nested_keys: a dictionary containing nested keys, from that
            avro schema is extracted. E.g.
            {'name': 1, 'favorite_number': 1, 'favorite_color': 1,
            'address': {'streetaddress': 1, 'city': 1}}
        :type nested_keys: dict
        :param schema_avro: the current avro schema needed to be updated. E.g.:
            {
              'namespace': 'avro_namespace',
              'name': 'avro_filename',
              'type': 'record',
              'fields': [
                {'name': 'name', 'type': ['string', 'null']},
                {'name': 'favorite_number', 'type': ['string', 'null']},
                {'name': 'favorite_color', 'type': ['string', 'null']},
                {
                  'name': 'address',
                  'type': [{
                      'namespace': 'avro_namespace',
                      'name': 'address',
                      'type': 'record',
                      'fields': [
                          {'name': 'streetaddress', 'type': ['string', 'null']},
                          {'name': 'city', 'type': ['string', 'null']}
                      ]
                    },
                    'null'
                  ]
                }
              ]
            }
        :type schema_avro: avro schema
        :return : updated avro schema
        """
        for key, value in nested_keys.items():
            if type(value) is dict:
                # here, the null option to specify keys not required
                # for every lines
                schema_avro_temp = {
                    "name": key,
                    "type": [{
                        "name": key,
                        "type": "record",
                        "fields": []
                    }, "null"]
                }
                schema_avro_temp["type"][0] = cls._get_schema_avro(
                                    value, schema_avro_temp["type"][0])
                schema_avro["fields"].append(schema_avro_temp)
            else:
                schema_avro["fields"].append(
                                {"name": key, "type": ["string", "null"]})

        return schema_avro

