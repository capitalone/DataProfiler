.. _Data Profiler:

====================================
Data Profiler | What's in your data?
====================================

Purpose
=======

The DataProfiler is a Python library designed to make data analysis, monitoring and **sensitive data detection** easy.

Loading **Data** with a single command, the library automatically formats & loads files into a DataFrame. **Profiling** the Data, the library identifies the schema, statistics, entities and more. Data Profiles can then be used in downstream applications or reports.

The Data Profiler comes with a cutting edge pre-trained deep learning model, used to efficiently identify **sensitive data** (or **PII**). If customization is needed, it's easy to add new entities to the existing pre-trained model or insert a new pipeline for entity recognition.

The best part? Getting started only takes a few lines of code (`Example CSV`_):

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler
    
    data = Data("your_file.csv") # Auto-Detect & Load: CSV, AVRO, Parquet, JSON, Text
    print(data.data.head(5)) # Access data directly via a compatible Pandas DataFrame
    
    profile = Profiler(data) # Calculate Statistics, Entity Recognition, etc
    readable_report = profile.report(report_options={"output_format":"pretty"})
    print(json.dumps(readable_report, indent=4))


To install the full package from pypi: 

.. code-block:: console

    pip install DataProfiler[ml]

If the ML requirements are too strict (say, you don't want to install tensorflow), you can install a slimmer package. The slimmer package disables the default sensitive data detection / entity recognition (labler)

Install from pypi: 

.. code-block:: console

    pip install DataProfiler

If you have suggestions or find a bug, please open an `issue`_.

Visit the :ref:`API<API>` to explore Data Profiler's terminology.


What is a Data Profile?
=======================

In the case of this library, a data profile is a dictionary containing statistics and predictions about the underlying dataset. There are "global statistics" or `global_stats`, which contain dataset level data and there are "column/row level statistics" or `data_stats` (each column is a new key-value entry). 

The format for a profile is below:

.. code-block:: python

    "global_stats": {
        "samples_used": int,
        "column_count": int,
        "row_count": int,
        "row_has_null_ratio": float,
        "row_is_null_ratio": float,    
        "unique_row_ratio": float,
        "duplicate_row_count": int,
        "file_type": string,
        "encoding": string,
    },
    "data_stats": {
        <column name>: {
            "column_name": string,
            "data_type": string,
            "data_label": string,
            "categorical": bool,
            "order": string,
            "samples": list(str),
            "statistics": {
                "sample_size": int,
                "null_count": int,
                "null_types": list(string),
                "null_types_index": {
                    string: list(int)
                },
                "data_type_representation": string,
                "min": [null, float],
                "max": [null, float],
                "mean": float,
                "variance": float,
                "stddev": float,
                "histogram": { 
                    "bin_counts": list(int),
                    "bin_edges": list(float),
                },
                "quantiles": {
                    int: float
                }
                "vocab": list(char),
                "avg_predictions": dict(float), 
                "data_label_representation": dict(float),
                "categories": list(str),
                "unique_count": int,
                "unique_ratio": float,
                "precision": {
                'min': int,
                'max': int,
                'mean': float,
                'var': float,
                'std': float,
                'sample_size': int,
                'margin_of_error': float,
                'confidence_level': float		
            },
            "times": dict(float),
            "format": string
            }
        }
    }

Support
~~~~~~~

Supported Data Formats
----------------------

* Any delimited file (CSV, TSV, etc.)
* JSON object
* Avro file
* Parquet file
* Pandas DataFrame


Data Labels
-----------

*Data Labels* are determined per cell for structured data (column/row when the *profiler* is used) or at the character level for unstructured data.

* UNKNOWN
* ADDRESS
* BAN (bank account number, 10-18 digits)
* CREDIT_CARD
* EMAIL_ADDRESS
* UUID 
* HASH_OR_KEY (md5, sha1, sha256, random hash, etc.)
* IPV4
* IPV6
* MAC_ADDRESS
* PERSON
* PHONE_NUMBER
* SSN
* URL
* US_STATE
* DRIVERS_LICENSE
* DATE
* TIME
* DATETIME
* INTEGER
* FLOAT
* QUANTITY
* ORDINAL


Get Started
===========

Load a File
~~~~~~~~~~~

The Data Profiler can profile the following data/file types:

* CSV file (or any delimited file)
* JSON object
* Avro file
* Parquet file
* Pandas DataFrame

The profiler should automatically identify the file type and load the data into a `Data Class`.

Along with other attributtes the `Data class` enables data to be accessed via a valid Pandas DataFrame.

.. code-block:: python

    # Load a csv file, return a CSVData object
    csv_data = Data('your_file.csv') 

    # Print the first 10 rows of the csv file
    print(csv_data.data.head(10))

    # Load a parquet file, return a ParquetData object
    parquet_data = Data('your_file.parquet')

    # Sort the data by the name column
    parquet_data.data.sort_values(by='name', inplace=True)

    # Print the sorted first 10 rows of the parquet data
    print(parquet_data.data.head(10))


If the file type is not automatically identified (rare), you can specify them 
specifically, see section [Specifying a Filetype or Delimiter](#specifying-a-filetype-or-delimiter).

Profile a File 
~~~~~~~~~~~~~~

Example uses a CSV file for example, but CSV, JSON, Avro or Parquet should also work.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load file (CSV should be automatically identified)
    data = Data("your_file.csv") 

    # Profile the dataset
    profile = Profiler(data)

    # Generate a report and use json to prettify.
    report  = profile.report(report_options={"output_format":"pretty"})

    # Print the report
    print(json.dumps(report, indent=4))

Updating Profiles
~~~~~~~~~~~~~~~~~

Currently, the data profiler is equipped to update its profile in batches.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load and profile a CSV file
    data = Data("your_file.csv")
    profile = Profiler(data)

    # Update the profile with new data:
    new_data = Data("new_data.csv")
    profile.update_profile(new_data)

    # Print the report using json to prettify.
    report  = profile.report(report_options={"output_format":"pretty"})
    print(json.dumps(report, indent=4))


Merging Profiles
~~~~~~~~~~~~~~~~

If you have two files with the same schema (but different data), it is possible to merge the two profiles together via an addition operator. 

This also enables profiles to be determined in a distributed manner.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load a CSV file with a schema
    data1 = Data("file_a.csv")
    profile1 = Profiler(data)

    # Load another CSV file with the same schema
    data2 = Data("file_b.csv")
    profile2 = Profiler(data)

    profile3 = profile1 + profile2

    # Print the report using json to prettify.
    report  = profile3.report(report_options={"output_format":"pretty"})
    print(json.dumps(report, indent=4))

Profile a Pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import dataprofiler as dp
    import json

    my_dataframe = pd.DataFrame([[1, 2.0],[1, 2.2],[-1, 3]])
    profile = dp.Profiler(my_dataframe)

    # print the report using json to prettify.
    report = profile.report(report_options={"output_format":"pretty"})
    print(json.dumps(report, indent=4))

    # read a specified column, in this case it is labeled 0:
    print(json.dumps(report["data stats"][0], indent=4))


Specifying a Filetype or Delimiter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example of specifying a CSV data type, with a `,` delimiter.
In addition, it utilizes only the first 10,000 rows.

.. code-block:: python

    import json
    import os
    from dataprofiler import Data, Profiler
    from dataprofiler.data_readers.csv_data import CSVData

    # Load a CSV file, with "," as the delimiter
    data = CSVData("your_file.csv", options={"delimiter": ","})

    # Split the data, such that only the first 10,000 rows are used
    data = data.data[0:10000]

    # Read in profile and print results
    profile = Profiler(data)
    print(json.dumps(profile.report(report_options={"output_format":"pretty"}), indent=4))


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started:

   Intro<self>
   install.rst
   profiler.rst
   data_readers.rst
   data_labeling.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide:

   examples.rst
   API.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Community:

   Changelog<https://github.com/capitalone/DataProfiler/releases>
   Feedback<https://github.com/capitalone/DataProfiler/issues/new/choose>
   GitHub<https://github.com/capitalone/DataProfiler>

.. _Example CSV: https://raw.githubusercontent.com/capitalone/DataProfiler/main/dataprofiler/tests/data/csv/aws_honeypot_marx_geo.csv
.. _issue: https://github.com/capitalone/DataProfiler/issues/new/choose

Versions
========
* `0.4.3`_
* `0.3.0`_

.. _0.3.0: ../../v0.3/html/index.html
.. _0.4.3: ../../0.4.3/html/index.html

