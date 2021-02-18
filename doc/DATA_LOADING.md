Data Profiler | Data Loading
############################

Our object with the data profiler library is a "one function" load of any file. The file should be correctly loaded into a standardized format capable of being profiled by the `Profiler class`. In practice, this means identifying the correct file format, encoding, and then loading the data into a pandas DataFrame (internally). 

Today, it can identify & load any of the following file types:

* A delimited file (CSV, TSV, etc.)
* JSON object
* Avro file
* Parquet file
* Text file

How to use:
```python
import json
import dataprofiler as dp

# Auto-Detect & Load a file
data = dp.Data("your_file.csv") 

# Pandas DataFrame
print(data.data.head(5))
```

The neat part about the `Data class` is that it is a [factory class](https://www.geeksforgeeks.org/factory-method-python-design-patterns/). What's a factory class? It's a class which lets subclasses decide when to instantiate. Put simply, the `Data class` will load the specific class required to interpret & maniuplate the given file. In way of an example:

```python
import dataprofiler as dp

# Auto-Detect & Load a file
data = dp.Data("your_file.csv")

# returns CSVData
print(type(data))

# <class 'dataprofiler.data_readers.csv_data.CSVData'>
```

We provide an interface which all data classes must match, with a template provided in the `base_data class` ([data_readers/base_data.py](/data_profiler/data_readers/base_data.py)). What that means is that the end user doesn't have to worry about functions for the specific class, as the template provides a set of required functions. For general purposes, the various the various data classes can be treated the exact same. However, a given class can be checked and special functions added (if desired), making it more extensible. 

The current classes are:

* **CSVData** - A delimited file (CSV, TSV, etc.)
* **JSONData** - JSON object
* **AVROData** - Avro file
* **ParquetData** - Parquet file
* **TextData** - Text file

To override this automated functionality of determining a file, you must specify a `data_type`:

```python
import pandas as pd
import data_profiler as dp

data = dp.Data(data=pd.DataFrame([1, 2]), data_type='csv')
```

*Note: This override can still if fail, if the `data_reader` cannot load the file.*

It is important to note that by default the `Data class` does not utilize file extensions to identify *file type*. Instead, it inspects the file itself to determine which `data_reader` it should use. See [data_readers/csv_data.py](/data_profiler/data_readers/csv_data.py) as an example. The `is_match` function is utilized by the `CSVData` class to determine if it can be used to load the given file.

Finally, if a `data_reader` is missing please feel free to contribute one back to the project.

# Data Classes and Options

If further specificity is needed due to a failure of automated detection it's possible to use the specific `Data Class`
The `Data` class itself will identify then output one of the following `Data` class types. It's also possible to specifically call one of these data classes such as the following command:

```python
from dataprofiler.data_readers.csv_data import CSVData
data = CSVData("your_file.csv", options={"delimiter": ","})
```

Below are descriptions of the various `Data` classes and the available options.

### CSVData

Data class for loading datasets of type CSV. Can be specified by passing
in memory data or via a file path. Options pertaining the CSV may also
be specified using the options dict parameter.

`CSVData(input_file_path=None, data=None, options=None)`

Possible `options`:

* delimiter - Must be a string, for example `"delimiter": ","`
* data_format - must be a string, possible choices: "dataframe", "records"
* selected_columns - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`
* header - Define the header, similar to pandas

**Example**:
```python
from dataprofiler.data_readers.csv_data import CSVData
data = CSVData("your_file.csv",
		options={
		    "delimiter": ",",
		    "selected_columns": ["column 1", "ssn"]
		})
```

### JSONData

Data class for loading datasets of type JSON. Can be specified by
passing in memory data or via a file path. Options pertaining the JSON
may also be specified using the options dict parameter.

`JSONData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "json"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

**Example**:
```python
from dataprofiler.data_readers.json_data import JSONData
data = JSONData("your_file.json",
		options={
		    "data_format": "dataframe",
		    "selected_keys": ["name", "ssn"]
		})
```

### AVROData

Data class for loading datasets of type AVRO. Can be specified by
passing in memory data or via a file path. Options pertaining the AVRO
may also be specified using the options dict parameter.

`AVROData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "avro"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

**Example**:
```python
from dataprofiler.data_readers.avro_data import AVROData
data = AVROData("your_file.avro",
		options={
		    "data_format": "dataframe",
		    "selected_keys": ["name", "ssn"]
		})
```

### ParquetData

Data class for loading datasets of type PARQUET. Can be specified by
passing in memory data or via a file path. Options pertaining the
PARQUET may also be specified using the options dict parameter.

`ParquetData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format - must be a string, choices: "dataframe", "records", "json"
* selected_keys - columns being selected from the entire dataset, must be a list `["column 1", "ssn"]`

**Example**:
```python
from dataprofiler.data_readers.parquet_data import ParquetData
data = ParquetData("your_file.parquet",
		options={
		    "data_format": "dataframe",
		    "selected_keys": ["name", "ssn"]
		})
```

### TextData

Data class for loading datasets of type TEXT. Can be specified by
passing in memory data or via a file path. Options pertaining the TEXT
may also be specified using the options dict parameter.

`TextData(input_file_path=None, data=None, options=None)`

Possible `options`:

* data_format: user selected format in which to return data can only be of specified types
* samples_per_line - chunks by which to read in the specified dataset

**Example**:
```python
from dataprofiler.data_readers.text_data import TextData
data = TextData("your_file.txt")
```