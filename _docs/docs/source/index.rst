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

The format for a structured profile is below:

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
        "correlation_matrix": list[list[int]], (*)
        "chi2_matrix": list[list[float]],
        "profile_schema": dict[string, list[int]]
    },
    "data_stats": [
        {
            "column_name": string,
            "data_type": string,
            "data_label": string,
            "categorical": bool,
            "order": string,
            "samples": list[str],
            "statistics": {
                "sample_size": int,
                "null_count": int,
                "null_types": list[string],
                "null_types_index": dict[string, list[int]],
                "data_type_representation": dict[string, list[string]],
                "min": [null, float],
                "max": [null, float],
                "sum": float,
                "mode": list[float],
                "median": float,
                "median_absolute_deviation": float,
                "mean": float,
                "variance": float,
                "stddev": float,
                "skewness": float,
                "kurtosis": float,
                "num_zeros": int,
                "num_negatives": int,
                "histogram": { 
                    "bin_counts": list[int],
                    "bin_edges": list[float],
                },
                "quantiles": {
                    int: float
                },
                "vocab": list[char],
                "avg_predictions": dict[string, float], 
                "data_label_representation": dict[string, float],
                "categories": list[str],
                "unique_count": int,
                "unique_ratio": float,
                "categorical_count": dict[string, int],
                "gini_impurity": float,
                "unalikeability": float,
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
                "times": dict[string, float],
                "format": string
            },
            "null_replication_metrics": {
                "class_prior": list[int],
                "class_sum": list[list[int]],
                "class_mean": list[list[int]]
            }
        }
    ]

(*) Currently the correlation matrix update is toggled off. It will be reset in a later update. Users can still use it as desired with the is_enable option set to True.

The format for an unstructured profile is below:

.. code-block:: python

    "global_stats": {
        "samples_used": int,
        "empty_line_count": int,
        "file_type": string,
        "encoding": string,
        "memory_size": float, # in MB
    },
    "data_stats": {
        "data_label": {
            "entity_counts": {
                "word_level": dict[string, int],
                "true_char_level": dict[string, int],
                "postprocess_char_level": dict[string, int]
            },
            "entity_percentages": {
                "word_level": dict[string, float],
                "true_char_level": dict[string, float],
                "postprocess_char_level": dict[string, float]
            },
            "times": dict[string, float]
        },
        "statistics": {
            "vocab": list[char],
            "vocab_count": dict[string, int],
            "words": list[string],
            "word_count": dict[string, int],
            "times": dict[string, float]
        }
    }

The format for a graph profile is below:

.. code-block:: python
    
    "num_nodes": int,
    "num_edges": int,
    "categorical_attributes": list[string],
    "continuous_attributes": list[string],
    "avg_node_degree": float,
    "global_max_component_size": int,
    "continuous_distribution": {
        "<attribute_1>": {
            "name": string,
            "scale": float,
            "properties": list[float, np.array]
        },
        "<attribute_2>": None,
    },
    "categorical_distribution": {
        "<attribute_1>": None,
        "<attribute_2>": {
            "bin_counts": list[int],
            "bin_edges": list[float]
        },
    }, 
    "times": dict[string, float]

Supported Data Formats
~~~~~~~~~~~~~~~~~~~~~~

* Any delimited file (CSV, TSV, etc.)
* JSON object
* Avro file
* Parquet file
* Text file
* Pandas DataFrame
* A URL that points to one of the supported file types above


Data Labels
~~~~~~~~~~~

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

The profiler should automatically identify the file type and load the data into a `Data Class`.

Along with other attributtes the `Data class` enables structured data to be accessed via a valid Pandas DataFrame.

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
specifically, see section Data Readers.

Profile a File 
~~~~~~~~~~~~~~

Example uses a CSV file for example, but CSV, JSON, Avro, Parquet or Text should also work.

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


Unstructured Profiler
~~~~~~~~~~~~~~~~~~~~~

In addition to the structured profiler, the Data Profiler provides unstructured 
profiling for the TextData object or string. Unstructured profiling also works 
with list(string), pd.Series(string) or pd.DataFrame(string) given profiler_type
option specified as `unstructured`. Below is an example of unstructured profile 
with a text file. 

.. code-block:: python

    import dataprofiler as dp
    import json
    my_text = dp.Data('text_file.txt')
    profile = dp.Profiler(my_text)
    
    # print the report using json to prettify.
    report = profile.report(report_options={"output_format":"pretty"})
    print(json.dumps(report, indent=4))
    
Another example of unstructured profile with pd.Series of string is given as below

.. code-block:: python

    import dataprofiler as dp
    import pandas as pd
    import json
    
    text_data = pd.Series(['first string', 'second string'])
    profile = dp.Profiler(text_data, profiler_type="unstructured")
    
    # print the report using json to prettify.
    report = profile.report(report_options={"output_format":"pretty"})
    print(json.dumps(report, indent=4))


Graph Profiler
~~~~~~~~~~~~~~

DataProfiler also provides the ability to profile graph data from a csv file. Below is an example of the graph profiler with a graph data csv file:

.. code-block:: python

    import dataprofiler as dp
    import pprint

    my_graph = dp.Data('graph_file.csv')
    profile = dp.Profiler(my_graph)

    # print the report using pretty print (json dump does not work on numpy array values inside dict)
    report = profile.report()
    printer = pprint.PrettyPrinter(sort_dicts=False, compact=True)
    printer.pprint(report)


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
   data_readers.rst
   profiler.rst
   data_labeling.rst
   graphs.rst
   architecture.rst

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

   roadmap.rst
   Changelog<https://github.com/capitalone/DataProfiler/releases>
   Feedback<https://github.com/capitalone/DataProfiler/issues/new/choose>
   GitHub<https://github.com/capitalone/DataProfiler>
   Contributing<https://github.com/capitalone/DataProfiler/blob/main/.github/CONTRIBUTING.md>

.. _Example CSV: https://raw.githubusercontent.com/capitalone/DataProfiler/main/dataprofiler/tests/data/csv/aws_honeypot_marx_geo.csv
.. _issue: https://github.com/capitalone/DataProfiler/issues/new/choose



