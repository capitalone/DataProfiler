# Data Profiler | What's in your data?

The DataProfiler is a Python library designed to make data analysis, monitoring and **sensitive data detection** easy.

Loading **Data** with a single command, the library automatically formats & loads files into a DataFrame. **Profiling** the Data, the library identifies the schema, statistics, entities and more. Data Profiles can then be used in downstream applications or reports.

The Data Profiler comes with a cutting edge pre-trained deep learning model, used to efficiently identify **sensitive data** (or **PII**). If customization is needed, it's easy to add new entities to the existing pre-trained model or insert a new pipeline for entity recognition.

The best part? Getting started only takes a few lines of code ([example csv](https://raw.githubusercontent.com/capitalone/DataProfiler/main/dataprofiler/tests/data/csv/aws_honeypot_marx_geo.csv)):

```python
import json
from dataprofiler import Data, Profiler

data = Data("your_file.csv") # Auto-Detect & Load: CSV, AVRO, Parquet, JSON, Text

print(data.data.head(5)) # Access data directly via a compatible Pandas DataFrame

profile = Profiler(data) # Calculate Statistics, Entity Recognition, etc

readable_report = profile.report(report_options={"output_format":"pretty"})

print(json.dumps(readable_report, indent=4))
```

To install the full package from pypi: `pip install DataProfiler[ml]`

If the ML requirements are too strict (say, you don't want to install tensorflow), you can install a slimmer package. The slimmer package disables the default sensitive data detection / entity recognition (labler)

Install from pypi: `pip install DataProfiler`


For API documentation, visit the [documentation page](https://capitalone.github.io/DataProfiler/).

If you have suggestions or find a bug, [please open an issue](https://github.com/capitalone/dataprofiler/issues/new/choose).

# Table of Contents

* [What is a Data Profile?](#what-is-a-data-profile)
* [Support](#support)
    * [Supported Data Formats](#supported-data-formats)
    * [Data Types](#data-types)
    * [Data Labels](#data-labels)
* [Installation](#installation)
    * [Snappy Installation](#snappy-installation)
    * [Data Profiler Installation](#data-profiler-installation)
    * [Testing](#testing)
* [Get Started](#get-started)
    * [Load a File](#load-a-file)
    * [Profile a File](#profile-a-file)
    * [Updating Profiles](#updating-profiles)
    * [Merging Profiles](#merging-profiles)
    * [Profile a Pandas DataFrame](#profile-a-pandas-dataframe)
    * [Specifying a Filetype or Delimiter](#specifying-a-filetype-or-delimiter)
* [Profile Options](#profile-options)
* [Data Classes and Options](#data-classes-and-options)
* [Data Labeling](#data-labeling)
    * [Identify Entities in Structured Data](#identify-entities-in-structured-data)
    * [Identify Entities in Unstructured Data](#identify-entities-in-unstructured-data)
    * [Train a New Data Labeler](#train-a-new-data-labeler)
    * [Load an Existing Data Labeler](#load-an-existing-data-labeler)
    * [Extending a Data Labeler with Transfer Learning](#extending-a-data-labeler-with-transfer-learning)
* [Build Your Own Data Labeler](#build-your-own-data-labeler)
* [Updating Documentation](#updating-documentation)
* [References](#references)
* [Contributors](#contributors)

------------------

# What is a Data Profile?

In the case of this library, a data profile is a dictionary containing statistics and predictions about the underlying dataset. There are "global statistics" or `global_stats`, which contain dataset level data and there are "column/row level statistics" or `data_stats` (each column is a new key-value entry). 

The format for a profile is below:

```
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
```

# Support

### Supported Data Formats

* Any delimited file (CSV, TSV, etc.)
* JSON object
* Avro file
* Parquet file
* Pandas DataFrame

### Data Types

*Data Types* are determined at the column level for structured data

* Int
* Float
* String
* DateTime

### Data Labels

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

# Installation

### Snappy Installation

This is required to profile parquet/avro datasets

MacOS with homebrew:
```
brew install snappy
```

Linux install:
```
sudo apt-get -y install libsnappy-dev
```

### Data Profiler Installation

NOTE: Installation for python3

virtualenv install:
```
python3 -m pip install virtualenv
```

Setup virtual env:
```
python3 -m virtualenv --python=python3 venv3
source venv3/bin/activate
```

Install requirements:
```
pip3 install -r requirements.txt
```

Install labeler dependencies:
```
pip3 install -r requirements-ml.txt
```

Install via the repo -- Build setup.py and install locally:
```
python3 setup.py sdist bdist bdist_wheel
pip3 install dist/DataProfiler*-py3-none-any.whl
```

If you see:
```
 ERROR: Double requirement given:dataprofiler==X.Y.Z from dataprofiler/dist/DataProfiler-X.Y.Z-py3-none-any.whl (already in dataprofiler==X2.Y2.Z2 from dataprofiler/dist/DataProfiler-X2.Y2.Z2-py3-none-any.whl, name='dataprofiler')
 ```
This means that you have multiple versions of the DataProfiler distribution 
in the dist folder.
To resolve, either remove the older one or delete the folder and rerun the steps
 above.

Install via github:
```
pip3 install git+https://github.com/capitalone/dataprofiler.git#egg=dataprofiler
```


### Testing

For testing, install test requirements:
```
pip3 install -r requirements-test.txt
```

To run all unit tests, use:
```
DATAPROFILER_SEED=0 python3 -m unittest discover -p "test*.py"
```

To run file of unit tests, use form:

```
DATAPROFILER_SEED=0 python3 -m unittest discover -p test_profile_builder.py
```

To run a file with Pytest use:
```
DATAPROFILER_SEED=0 pytest dataprofiler/tests/data_readers/test_csv_data.py -v
```

To run individual of unit test, use form:
```
DATAPROFILER_SEED=0 python3 -m unittest dataprofiler.tests.profilers.test_profile_builder.TestProfiler
```

# Get Started

### Load a File

The Data Profiler can profile the following data/file types:

* CSV file (or any delimited file)
* JSON object
* Avro file
* Parquet file
* Pandas DataFrame

The profiler should automatically identify the file type and load the data into a `Data Class`.

Along with other attributtes the `Data class` enables data to be accessed via a valid Pandas DataFrame.

```python
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
```

If the file type is not automatically identified (rare), you can specify them 
specifically, see section [Specifying a Filetype or Delimiter](#specifying-a-filetype-or-delimiter).

### Profile a File 

Example uses a CSV file for example, but CSV, JSON, Avro or Parquet should also work.

```python
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
```

### Updating Profiles

Currently, the data profiler is equipped to update its profile in batches.

```python
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
```

### Merging Profiles

If you have two files with the same schema (but different data), it is possible to merge the two profiles together via an addition operator. 

This also enables profiles to be determined in a distributed manner.

```python
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
```

### Profile a Pandas DataFrame
```python
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
```

### Specifying a Filetype or Delimiter

Example of specifying a CSV data type, with a `,` delimiter.
In addition, it utilizes only the first 10,000 rows.

```python
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
```

# Profile Options

The data profiler accepts several options to toggle on and off 
features. The 8 columns (int options, float options, datetime options,
text options, order options, category options, data labeler options) can be 
enabled or disabled. By default, all options are toggled on. Below is an example
of how to alter these options. 

```python
import json
from dataprofiler import Data, Profiler, ProfilerOptions

# Load and profile a CSV file
data = Data("your_file.csv")
profile_options = ProfilerOptions()

#All of these are different examples of adjusting the profile options

# Options can be toggled directly like this:
profile_options.structured_options.text.is_enabled = False
profile_options.structured_options.text.vocab.is_enabled = True
profile_options.structured_options.int.variance.is_enabled = True
profile_options.structured_options.data_labeler.data_labeler_dirpath = \
    "Wheres/My/Datalabeler"
profile_options.structured_options.data_labeler.is_enabled = False

# A dictionary can be sent in to set the properties for all the options
profile_options.set({"data_labeler.is_enabled": False, "min.is_enabled": False})

# Specific columns can be set/disabled/enabled in the same way
profile_options.structured_options.text.set({"max.is_enabled":True, 
                                             "variance.is_enabled": True})

# numeric stats can be turned off/on entirely
profile_options.set({"is_numeric_stats_enabled": False})
profile_options.set({"int.is_numeric_stats_enabled": False})

profile = Profiler(data, profiler_options=profile_options)

# Print the report using json to prettify.
report  = profile.report(report_options={"output_format":"pretty"})
print(json.dumps(report, indent=4))
```

Below is an breakdown of all the options.

* **ProfilerOptions** - The top-level options class that contains options for the Profiler class
    * **structured_options** - Options responsible for all structured data
      	* **multiprocess** - Option to enable multiprocessing. Automatically selects the optimal number of processes to utilize based on system constraints.
            * is_enabled - (Boolean) Enables or disables multiprocessing
        * **int** - Options for the integer columns
            * is_enabled - (Boolean) Enables or disables the integer operations
            * min - Finds minimum value in a column
                * is_enabled - (Boolean) Enables or disables min
            * max - Finds maximum value in a column
                * is_enabled - (Boolean) Enables or disables max
            * sum - Finds sum of all values in a column
                * is_enabled - (Boolean) Enables or disables sum
            * variance - Finds variance of all values in a column
                * is_enabled - (Boolean) Enables or disables variance
            * histogram_and_quantiles - Generates a histogram and quantiles
            from the column values
                * bin_count_or_method - (String/List[String]) Designates preferred method for calculating histogram bins or the number of bins to use.  
                If left unspecified (None) the optimal method will be chosen by attempting all methods.  
                If multiple specified (list) the optimal method will be chosen by attempting the provided ones.  
                methods: 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'  
                Note: 'auto' is used to choose optimally between 'fd' and 'sturges'
                * is_enabled - (Boolean) Enables or disables histogram and quantiles
        * **float** - Options for the float columns
            * is_enabled - (Boolean) Enables or disables the float operations
            * precision - Finds the precision (significant figures) within the column
                * is_enabled - (Boolean) Enables or disables precision
		* sample_ratio - (Float) The ratio of 0 to 1 how much data (identified as floats) to utilize as samples in determining precision 
            * min - Finds minimum value in a column
                * is_enabled - (Boolean) Enables or disables min
            * max - Finds maximum value in a column
                * is_enabled - (Boolean) Enables or disables max
            * sum - Finds sum of all values in a column
                * is_enabled - (Boolean) Enables or disables sum
            * variance - Finds variance of all values in a column
                * is_enabled - (Boolean) Enables or disables variance
            * histogram_and_quantiles - Generates a histogram and quantiles
            from the column values
                * bin_count_or_method - (String/List[String]) Designates preferred method for calculating histogram bins or the number of bins to use.  
                If left unspecified (None) the optimal method will be chosen by attempting all methods.  
                If multiple specified (list) the optimal method will be chosen by attempting the provided ones.  
                methods: 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'  
                Note: 'auto' is used to choose optimally between 'fd' and 'sturges'
                * is_enabled - (Boolean) Enables or disables histogram and quantiles        
        * **text** - Options for the text columns
            * is_enabled - (Boolean) Enables or disables the text operations
            * vocab - Finds all the unique characters used in a column
                * is_enabled - (Boolean) Enables or disables vocab
            * min - Finds minimum value in a column
                * is_enabled - (Boolean) Enables or disables min
            * max - Finds maximum value in a column
                * is_enabled - (Boolean) Enables or disables max
            * sum - Finds sum of all values in a column
                * is_enabled - (Boolean) Enables or disables sum
            * variance - Finds variance of all values in a column
                * is_enabled - (Boolean) Enables or disables variance
            * histogram_and_quantiles - Generates a histogram and quantiles
            from the column values
                * bin_count_or_method - (String/List[String]) Designates preferred method for calculating histogram bins or the number of bins to use.  
                If left unspecified (None) the optimal method will be chosen by attempting all methods.  
                If multiple specified (list) the optimal method will be chosen by attempting the provided ones.  
                methods: 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'  
                Note: 'auto' is used to choose optimally between 'fd' and 'sturges'
                * is_enabled - (Boolean) Enables or disables histogram and quantiles  
        * **datetime** - Options for the datetime columns
            * is_enabled - (Boolean) Enables or disables the datetime operations
        * **order** - Options for the order columns
            * is_enabled - (Boolean) Enables or disables the order operations
        * **category** - Options for the category columns
            * is_enabled  - (Boolean) Enables or disables the category operations
        * **data_labeler** - Options for the data labeler columns
            * is_enabled - (Boolean) Enables or disables the data labeler operations
            * data_labeler_dirpath - (String) Directory path to data labeler
            * data_labeler_object - (BaseDataLabeler) Datalabeler to replace 
            the default labeler 
            * max_sample_size - (Int) The max number of samples for the data 
            labeler


#### Statistical Dependency on Order of Updates

Some profile features/statistics are dependent on the order in which the profiler
is updated with new data.

##### Order Profile

The order profiler utilizes the last value in the previous data batch to ensure
the subsequent dataset is above/below/equal to that value when predicting
non-random order.

For instance, a dataset to be predicted as ascending would require the following
batch data update to be ascending and its first value `>=` than that of the
previous batch of data.

###### Ex. of ascending
```
batch_1 = [0, 1, 2]
batch_2 = [3, 4, 5]
```

###### Ex. of random
```
batch_1 = [0, 1, 2]
batch_2 = [1, 2, 3] # notice how the first value is less than the last value in the previous batch
```

####  Reporting structure

For every profile, we can provide a report and customize it with a couple optional parameters:
* output_format (string)
  * This will allow the user to decide the output format for report.
    * Options are one of [pretty, compact, serializable, flat]:
      * Pretty: floats are rounded to four decimal places, and lists are shortened.
      * Compact: Similar to pretty, but removes detailed statistics such as runtimes, label probabilities, index locations of null types, etc.
      * Serializable: Output is json serializable and not prettified
      * Flat: Nested output is returned as a flattened dictionary
* num_quantile_groups (int)
  * You can sample your data as you like! With a minimum of one and a maximum of 1000, you can decide the number of quantile groups!
```python
report  = profile.report(report_options={"output_format": "pretty"})
report  = profile.report(report_options={"output_format": "compact"})
report  = profile.report(report_options={"output_format": "serializable"})
report  = profile.report(report_options={"output_format": "flat"})
```

# Data Classes and Options

The `Data` class itself will identify then output one of the following `Data` class types. It's also possible to specifically call one of these data classes such as the following command:

```python
from dataprofiler.data_readers.csv_data import CSVData
data = CSVData("your_file.csv", options={"delimiter": ","})
```

Below are descriptions of the various `Data` classes and the available options.

##### CSVData

Data class for loading datasets of type CSV. Can be specified by passing
in memory data or via a file path. Options pertaining the CSV may also
be specified using the options dict parameter.

`CSVData(input_file_path=None, data=None, options=None)`

Possible `options`:

* delimiter - Must be a string, for example `"delimiter": ","`
* data_format - must be a string, possible choices: "dataframe", "records"
* selected_columns - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`
* header - Define the header, for example
    - `"header": 'auto'` for auto detection
    - `"header": None` for no header
    - `"header": <INT>` to specify the header row (0 based index)

##### JSONData

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


##### AVROData

Data class for loading datasets of type AVRO. Can be specified by
passing in memory data or via a file path. Options pertaining the AVRO
may also be specified using the options dict parameter.

`AVROData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "avro"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

##### ParquetData

Data class for loading datasets of type PARQUET. Can be specified by
passing in memory data or via a file path. Options pertaining the
PARQUET may also be specified using the options dict parameter.

`ParquetData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "json"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

##### TextData

Data class for loading datasets of type TEXT. Can be specified by
passing in memory data or via a file path. Options pertaining the TEXT
may also be specified using the options dict parameter.

`TextData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format: user selected format in which to return data can only be of specified types
* samples_per_line - chunks by which to read in the specified dataset

# Data Labeling

In this library, the term *data labeling* refers to entity recognition.

Builtin to the data profiler is a classifier which evaluates the complex data types of the dataset.
For structured data, it determines the complex data type of each column. When
running the data profile, it uses the default data labeling model builtin to the
library. However, the data labeler allows users to train their own data labeler
as well.

## Identify Entities in Structured Data

Makes predictions and identifying labels:

```python
import dataprofiler as dp

# load data and data labeler
data = dp.Data("your_data.csv")
data_labeler = dp.DataLabeler(labeler_type='structured')

# make predictions and get labels per cell
predictions = data_labeler.predict(data)
```

## Identify Entities in Unstructured Data

Predict which class characters belong to in unstructured text:

```python
import dataprofiler as dp

data_labeler = dp.DataLabeler(labeler_type='unstructured')

# Example sample string, must be in an array (multiple arrays can be passed)
sample = ["Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
          "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912.\n"]

# Prediction what class each character belongs to
model_predictions = data_labeler.predict(
    sample, predict_options=dict(show_confidences=True))

# Predictions / confidences are at the character level
final_results = model_predictions["pred"]
final_confidences = model_predictions["conf"]
```

It's also possible to change output formats, output similar to a **SpaCy** format:

```python
import dataprofiler as dp

data_labeler = dp.DataLabeler(labeler_type='unstructured', trainable=True)

# Example sample string, must be in an array (multiple arrays can be passed)
sample = ["Help\tJohn Macklemore\tneeds\tfood.\tPlease\tCall\t555-301-1234."
          "\tHis\tssn\tis\tnot\t334-97-1234. I'm a BAN: 000043219499392912.\n"]

# Set the output to the NER format (start position, end position, label)
data_labeler.set_params(
    { 'postprocessor': { 'output_format':'ner', 'use_word_level_argmax':True } } 
)

results = data_labeler.predict(sample)

print(results)
```

## Train a New Data Labeler

Mechanism for training your own data labeler on their own set of structured data
 (tabular):

```python
import dataprofiler as dp

# Will need one column with a default label of UNKNOWN
data = dp.Data("your_file.csv")

data_labeler = dp.train_structured_labeler(
    data=data,
    save_dirpath="/path/to/save/labeler",
    epochs=2
)

data_labeler.save_to_disk("my/save/path") # Saves the data labeler for reuse
```

## Load an Existing Data Labeler

Mechanism for loading an existing data_labeler:

```python
import dataprofiler as dp

data_labeler = dp.DataLabeler(
    labeler_type='structured', dirpath="/path/to/my/labeler")

# get information about the parameters/inputs/output formats for the DataLabeler
data_labeler.help()
```

## Extending a Data Labeler with Transfer Learning

Extending or changing labels of a data labeler w/ transfer learning:
Note: By default, **a labeler loaded will not be trainable**. In order to load a 
trainable DataLabeler, the user must set `trainable=True` or load a labeler 
using the `TrainableDataLabeler` class.

The following illustrates how to change the labels:
```python
import dataprofiler as dp

labels = ['label1', 'label2', ...]  # new label set can also be an encoding dict
data = dp.Data("your_file.csv")  # contains data with new labels

# load default structured Data Labeler w/ trainable set to True
data_labeler = dp.DataLabeler(labeler_type='structured', trainable=True)

# this will use transfer learning to retrain the data labeler on your new 
# dataset and labels.
# NOTE: data must be in an acceptable format for the preprocessor to interpret.
#       please refer to the preprocessor/model for the expected data format.
#       Currently, the DataLabeler cannot take in Tabular data, but requires 
#       data to be ingested with two columns [X, y] where X is the samples and 
#       y is the labels.
model_results = data_labeler.fit(x=data['samples'], y=data['labels'], 
                                 validation_split=0.2, epochs=2, labels=labels)

# final_results, final_confidences are a list of results for each epoch
epoch_id = 0
final_results = model_results[epoch_id]["pred"]
final_confidences = model_results[epoch_id]["conf"]
```

The following illustrates how to extend the labels:
```python
import dataprofiler as dp

new_labels = ['label1', 'label2', ...]
data = dp.Data("your_file.csv")  # contains data with new labels

# load default structured Data Labeler w/ trainable set to True
data_labeler = dp.DataLabeler(labeler_type='structured', trainable=True)

# this will maintain current labels and model weights, but extend the model's 
# labels
for label in new_labels:
    data_labeler.add_label(label)
    
# NOTE: a user can also add a label which maps to the same index as an existing 
# label
# data_labeler.add_label(label, same_as='<label_name>')

# For a trainable model, the user must then train the model to be able to 
# continue using the labeler since the model's graph has likely changed
# NOTE: data must be in an acceptable format for the preprocessor to interpret.
#       please refer to the preprocessor/model for the expected data format.
#       Currently, the DataLabeler cannot take in Tabular data, but requires 
#       data to be ingested with two columns [X, y] where X is the samples and 
#       y is the labels.
model_results = data_labeler.fit(x=data['samples'], y=data['labels'], 
                                 validation_split=0.2, epochs=2)

# final_results, final_confidences are a list of results for each epoch
epoch_id = 0
final_results = model_results[epoch_id]["pred"]
final_confidences = model_results[epoch_id]["conf"]
```

Changing pipeline parameters:
```python
import dataprofiler as dp

# load default Data Labeler
data_labeler = dp.DataLabeler(labeler_type='structured')

# change parameters of specific component
data_labeler.preprocessor.set_params({'param1': 'value1'})

# change multiple simultaneously.
data_labeler.set_params({
    'preprocessor':  {'param1': 'value1'},
    'model':         {'param2': 'value2'},
    'postprocessor': {'param3': 'value3'}
})
```


# Build Your Own Data Labeler

The DataLabeler has 3 main components: preprocessor, model, and postprocessor. 
To create your own DataLabeler, each one would have to be created or an 
existing component can be reused.

Given a set of the 3 components, you can construct your own DataLabeler:

```python
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler, \
                                                     TrainableDataLabeler
from dataprofiler.labelers.character_level_cnn_model import CharacterLevelCnnModel
from dataprofiler.labelers.data_processing import \
    StructCharPreprocessor, StructCharPostprocessor

# load a non-trainable data labeler
model = CharacterLevelCnnModel(...)
preprocessor = StructCharPreprocessor(...)
postprocessor = StructCharPostprocessor(...)

data_labeler = BaseDataLabeler.load_with_components(
    preprocessor=preprocessor, model=model, postprocessor=postprocessor)

# check for basic compatibility between the processors and the model
data_labeler.check_pipeline()


# load trainable data labeler
data_labeler = TrainableDataLabeler.load_with_components(
    preprocessor=preprocessor, model=model, postprocessor=postprocessor)

# check for basic compatibility between the processors and the model
data_labeler.check_pipeline()
```

Option for swapping out specific components of an existing labeler.
```python
import dataprofiler as dp
from dataprofiler.labelers.character_level_cnn_model import \
    CharacterLevelCnnModel
from dataprofiler.labelers.data_processing import \
    StructCharPreprocessor, StructCharPostprocessor

model = CharacterLevelCnnModel(...)
preprocessor = StructCharPreprocessor(...)
postprocessor = StructCharPostprocessor(...)

data_labeler = dp.DataLabeler(labeler_type='structured')
data_labeler.set_preprocessor(preprocessor)
data_labeler.set_model(model)
data_labeler.set_postprocessor(postprocessor)

# check for basic compatibility between the processors and the model
data_labeler.check_pipeline()
```


### Model Component
In order to create your own model component for data labeling, you can utilize 
the `BaseModel` class from `dataprofiler.labelers.base_model` and
overriding the abstract class methods.

Reviewing `CharacterLevelCnnModel` from 
`dataprofiler.labelers.character_level_cnn_model` illustrates the functions 
which need an override. 
  1. `__init__`: specifying default parameters and calling base `__init__`
  1. `_validate_parameters`: validating parameters given by user during setting
  1. `_need_to_reconstruct_model`: flag for when to reconstruct a model (i.e. 
  parameters change or labels change require a model reconstruction)
  1. `_construct_model`: initial construction of the model given the parameters
  1. `_reconstruct_model`: updates model architecture for new label set while 
  maintaining current model weights
  1. `fit`: mechanism for the model to learn given training data
  1. `predict`: mechanism for model to make predictions on data
  1. `details`: prints a summary of the model construction
  1. `save_to_disk`: saves model and model parameters to disk
  1. `load_from_disk`: loads model given a path on disk
  
  
### Preprocessor Component
In order to create your own preprocessor component for data labeling, you can 
utilize the `BaseDataPreprocessor` class 
from `dataprofiler.labelers.data_processing` and override the abstract class 
methods.

Reviewing `StructCharPreprocessor` from 
`dataprofiler.labelers.data_processing` illustrates the functions which 
need an override.
  1. `__init__`: passing parameters to the base class and executing any 
  extraneous calculations to be saved as parameters
  1. `_validate_parameters`: validating parameters given by user during
  setting
  1. `process`: takes in the user data and converts it into an digestible, 
  iterable format for the model
  1. `set_params` (optional): if a parameter requires processing before setting,
   a user can override this function to assist with setting the parameter
  1. `_save_processor` (optional): if a parameter is not JSON serializable, a 
  user can override this function to assist in saving the processor and its 
  parameters
  1. `load_from_disk` (optional): if a parameter(s) is not JSON serializable, a 
  user can override this function to assist in loading the processor 

### Postprocessor Component
The postprocessor is nearly identical to the preprocessor except it handles 
the output of the model for processing. In order to create your own 
postprocessor component for data  labeling, you can utilize the 
`BaseDataPostprocessor` class from  `dataprofiler.labelers.data_processing` 
and override the abstract class methods.

Reviewing `StructCharPostprocessor` from 
`dataprofiler.labelers.data_processing` illustrates the functions which 
need an override.
  1. `__init__`: passing parameters to the base class and executing any 
  extraneous calculations to be saved as parameters
  1. `_validate_parameters`: validating parameters given by user during
  setting
  1. `process`: takes in the output of the model and processes for output to 
  the user
  1. `set_params` (optional): if a parameter requires processing before setting,
   a user can override this function to assist with setting the parameter
  1. `_save_processor` (optional): if a parameter is not JSON serializable, a 
  user can override this function to assist in saving the processor and its 
  parameters
  1. `load_from_disk` (optional): if a parameter(s) is not JSON serializable, a 
  user can override this function to assist in loading the processor 


# Updating Documentation  
To update the docs branch, checkout the gh-pages branch. Make sure it is up to
date, then copy the `dataprofiler` folder from the feature branch you want to 
update the documentation with (probably master).

In /docs run:

    python update_documentation.py [version]

where [version] is the name of the version you want like "v0.1". If you make
adjustments to the code comments, you may rerun the command again to overwrite
the specified version. 

Once the documentation is updated, commit and push the whole 
/docs folder. API documentation will only update when pushed to the master 
branch. 

If you make a mistake naming the version, you will have to delete it from
the /docs/source/index.rst file.

To update the documentation of a feature branch, go to the /docs folder
and run:
```bash
python update_documentation.py [version]
```

# References
```
Sensitive Data Detection with High-Throughput Neural Network Models for Financial Institutions
Authors: Anh Truong, Austin Walters, Jeremy Goodsitt
2020 https://arxiv.org/abs/2012.09597
The AAAI-21 Workshop on Knowledge Discovery from Unstructured Data in Financial Services
```
# Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/JGSweets"><img src="https://avatars.githubusercontent.com/u/7725753?s=460&u=03cd86a04ddc00a29df188a8bc956c1179ba54ae&v=4" width="100px;" alt=""/><br /><sub><b>Jeremy Goodsitt</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/lettergram"><img src="https://avatars.githubusercontent.com/u/1498748?s=460&u=77393a45f1669d68d988d7206e468281a545818d&v=4" width="100px;" alt=""/><br /><sub><b>Austin Walters</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/AnhTruong"><img src="https://avatars.githubusercontent.com/u/11826571?s=460&v=4" width="100px;" alt=""/><br /><sub><b>Anh Truong</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/gme5078"><img src="https://avatars.githubusercontent.com/u/56846128?s=460&v=4" width="100px;" alt=""/><br /><sub><b>Grant Eden</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/derekli-NJ"><img src="https://avatars.githubusercontent.com/u/25947344?s=460&u=b7cef3e7a65c62b3c4fcde299f6ef5bb9df27779&v=4" width="100px;" alt=""/><br /><sub><b>Derek Li</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/vsinghai"><img src="https://avatars.githubusercontent.com/u/22135924?s=460&u=43af9cd8a5656fd55c8d8ebff2c8aae82c90ac30&v=4" width="100px;" alt=""/><br /><sub><b>Varun Singhai</b></sub></a><br /></td>
  </tr>
</table>
