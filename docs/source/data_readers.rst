.. _data_readers:

Data Readers
************

The `Data` class itself will identify then output one of the following `Data` class types. It's also possible to specifically call one of these data classes such as the following command:

.. code-block:: python

    from dataprofiler.data_readers.csv_data import CSVData
    data = CSVData("your_file.csv", options={"delimiter": ","})


Below are descriptions of the various `Data` classes and the available options.

CSVData
=======

Data class for loading datasets of type CSV. Can be specified by passing
in memory data or via a file path. Options pertaining the CSV may also
be specified using the options dict parameter.

`CSVData(input_file_path=None, data=None, options=None)`

Possible `options`:

* delimiter - Must be a string, for example `"delimiter": ","`
* data_format - must be a string, possible choices: "dataframe", "records"
* selected_columns - columns being selected from the entire dataset, must be a 
  list `["column 1", "ssn"]`
* header - Define the header, for example

  * `"header": 'auto'` for auto detection
  * `"header": None` for no header
  * `"header": <INT>` to specify the header row (0 based index)

JSONData
========

Data class for loading datasets of type JSON. Can be specified by
passing in memory data or via a file path. Options pertaining the JSON
may also be specified using the options dict parameter.

`JSONData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "json", "flattened_dataframe"
  
  * "flattened_dataframe" is best used for JSON structure typically found in data streams that contain
    nested lists of dictionaries and a payload. For example: `{"data": [ columns ], "response": 200}`
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`
* payload_keys - The dictionary keys for the payload of the JSON, typically called "data"
  or "payload". Defaults to ["data", "payload", "response"].


AVROData
========

Data class for loading datasets of type AVRO. Can be specified by
passing in memory data or via a file path. Options pertaining the AVRO
may also be specified using the options dict parameter.

`AVROData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "avro"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

ParquetData
===========

Data class for loading datasets of type PARQUET. Can be specified by
passing in memory data or via a file path. Options pertaining the
PARQUET may also be specified using the options dict parameter.

`ParquetData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "json"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

TextData
========

Data class for loading datasets of type TEXT. Can be specified by
passing in memory data or via a file path. Options pertaining the TEXT
may also be specified using the options dict parameter.

`TextData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format: user selected format in which to return data can only be of specified types
* samples_per_line - chunks by which to read in the specified dataset
