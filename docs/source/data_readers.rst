.. _data_readers:

Data Readers
************

The `Data` class itself will identify then output one of the following `Data` class types. 
Using the data reader is easy, just pass it through the Data object. 

.. code-block:: python

    import dataprofiler as dp
    data = dp.Data("your_file.csv")

The supported file types are: 

* CSV file (or any delimited file)
* JSON object
* Avro file
* Parquet file
* Graph data file
* Text file
* Pandas DataFrame
* A URL that points to one of the supported file types above

It's also possible to specifically call one of the data classes such as the following command:

.. code-block:: python

    from dataprofiler.data_readers.csv_data import CSVData
    data = CSVData("your_file.csv", options={"delimiter": ","})

Additionally any of the data classes can be loaded using a URL:

.. code-block:: python

    import dataprofiler as dp
    data = dp.Data("https://you_website.com/your_file.file", options={"verify_ssl": "True"})

Below are descriptions of the various `Data` classes and the available options.

CSVData
=======

Data class for loading datasets of type CSV. Can be specified by passing
in memory data or via a file path. Options pertaining the CSV may also
be specified using the options dict parameter.

`CSVData(input_file_path=None, data=None, options=None)`

Possible `options`:

* delimiter - Must be a string, for example `"delimiter": ","`
* data_format - Must be a string, possible choices: "dataframe", "records"
* selected_columns - Columns being selected from the entire dataset, must be a 
  list `["column 1", "ssn"]`
* sample_nrows - Reservoir sampling to sample `"n"` rows out of a total of `"M"` rows.
  Specified for how many rows to sample, default None.
* header - Define the header, for example

  * `"header": 'auto'` for auto detection
  * `"header": None` for no header
  * `"header": <INT>` to specify the header row (0 based index)

JSONData
========

Data class for loading datasets of type JSON. Can be specified by
passing in memory data or via a file path. Options pertaining the JSON
may also be specified using the options dict parameter. JSON data can be 
accessed via the "data" property, the "metadata" property, and the 
"data_and_metadata" property.

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

* data_format - must be a string, choices: "dataframe", "records", "avro", "json", "flattened_dataframe"

  * "flattened_dataframe" is best used for AVROs with a JSON structure typically found in data streams that contain
    nested lists of dictionaries and a payload. For example: `{"data": [ columns ], "response": 200}`
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

GraphData
=========

Data Class for loading datasets of graph data. Currently takes CSV format,
further type formats will be supported. Can be specified by passing
in memory data (NetworkX Graph) or via a file path. Options pertaining the CSV file may also
be specified using the options dict parameter. Loads data from CSV into memory
as a NetworkX Graph.

`GraphData(input_file_path=None, data=None, options=None)`

Possible `options`:

* delimiter - must be a string, for example `"delimiter": ","`
* data_format - must be a string, possible choices: "graph", "dataframe", "records"
* header - Define the header, for example

  * `"header": 'auto'` for auto detection
  * `"header": None` for no header
  * `"header": <INT>` to specify the header row (0 based index)

TextData
========

Data class for loading datasets of type TEXT. Can be specified by
passing in memory data or via a file path. Options pertaining the TEXT
may also be specified using the options dict parameter.

`TextData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format: user selected format in which to return data. Currently only supports "text".
* samples_per_line - chunks by which to read in the specified dataset


Data Using a URL
================

Data class for loading datasets of any type using a URL. Specified by passing in 
any valid URL that points to one of the valid data types. Options pertaining the 
URL may also be specified using the options dict parameter.

`Data(input_file_path=None, data=None, options=None)`

Possible `options`:

* verify_ssl: must be a boolean string, choices: "True", "False". Set to "True" by default.