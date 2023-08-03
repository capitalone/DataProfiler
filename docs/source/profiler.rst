.. _profiler:

Profiler
********

Profile Your Data
=================

Profiling your data is easy. Just use the data reader, send the data to the 
profiler, and print out the report.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler
    
    data = Data("your_file.csv") # Auto-Detect & Load: CSV, AVRO, Parquet, JSON, Text
    
    profile = Profiler(data) # Calculate Statistics, Entity Recognition, etc
    
    readable_report = profile.report(report_options={"output_format": "pretty"})
    print(json.dumps(readable_report, indent=4))

If the data is structured, the profile will return global statistics as well as
column by column statistics. The vast amount of statistics are listed on the 
intro page.

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

Example uses a CSV file for example, but CSV, JSON, Avro or Parquet should also work.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load file (CSV should be automatically identified)
    data = Data("your_file.csv") 

    # Profile the dataset
    profile = Profiler(data)

    # Generate a report and use json to prettify.
    report  = profile.report(report_options={"output_format": "pretty"})

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
    report  = profile.report(report_options={"output_format": "pretty"})
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
    report  = profile3.report(report_options={"output_format": "pretty"})
    print(json.dumps(report, indent=4))


Profile Differences
~~~~~~~~~~~~~~~~~~~

Profile differences take two profiles and find the differences
between them. Create the difference report like this:

.. code-block:: python

    from dataprofiler import Data, Profiler
    
    # Load a CSV file
    data1 = Data("file_a.csv")
    profile1 = Profiler(data)
    
    # Load another CSV file
    data2 = Data("file_b.csv")
    profile2 = Profiler(data)
    
    diff_report = profile1.diff(profile2)
    print(diff_report)
    
The `.diff()` operation is available between two profiles, although there are different
outputs depending on the type of profile being differenced. For example, for numerical
column profiles (e.g. integers and floats), two valuable calculations that 
`.diff()` returns are `t-test`, `chi2-test`, and `psi` (Popoulation Stability Index)
for understanding distributional changes.

The difference report contains a dictionary that mirrors the profile report. 
Each data type has its own difference:

* **Int/Float** - One profile subtracts the value from the other.

* **String** - The strings will be shown in a list:

  - [profile1 str, profile2 str]
* **List** - A list of 3 will be returned showing the unique values of
  each profile and the shared values:

  - [profile 1 unique values, shared values, profile 2 unique values]
* **Dict** - Some dictionaries with varied keys will also return a list
  of three in the format:

  - [profile 1 unique key-values, shared key differences, profile 2 unique key-values]

Otherwise, when no differences occur:

* **Any Type No Differences** - A string will report: "unchanged".

Below is the structured difference report:

.. code-block:: python

    {
        'global_stats': {
            'file_type': [str, str], 
            'encoding': [str, str],
            'samples_used': int, 
            'column_count': int,
            'row_count': int, 
            'row_has_null_ratio': float,
            'row_is_null_ratio': float,
            'unique_row_ratio': float,
            'duplicate_row_count': int,
            'correlation_matrix': list[list[float]],
            'chi2_matrix': list[list[float]],
            'profile_schema': list[dict[str, int]]
        },
        'data_stats': [{
            'column_name': str, 
            'data_type': [str, str],
            'data_label': [list[str], list[str], list[str]],
            'categorical': [str, str],
            'order': [str, str],
            'statistics': {
                'min': float,
                'max': float,
                'sum': float,
                'mean': float,
                'median': float,
                'mode': [list[float], list[float], list[float]],
                'median_absolute_deviation': float,
                'variance': float,
                'stddev': float,
                't-test': {
                    't-statistic': float,
                    'conservative': {'df': int,
                                     'p-value': float},
                    'welch': {'df': float,
                              'p-value': float}},
                'psi': float,
                "chi2-test": {
                    "chi2-statistic": float,
                    "df": int,
                    "p-value": float
                },
                'unique_count': int,
                'unique_ratio': float,
                'categories': [list[str], list[str], list[str]],
                'gini_impurity': float,
                'unalikeability': float,
                'categorical_count': [dict[str, int], dict[str, int], dict[str, int]],
                'avg_predictions': [dict[str, float]],
                'label_representation': [dict[str, float]],
                'sample_size': int,
                'null_count': int,
                'null_types': [list[str], list[str], list[str]],
                'null_types_index': [dict[str, int], dict[str, int], dict[str, int]],
                'data_type_representation': [dict[str, float]]
            },
            "null_replication_metrics": {
                "class_prior": list[int],
                "class_sum": list[list[int]],
                "class_mean": list[list[int]]
            }
        }
        
Below is the unstructured difference report:

.. code-block:: python
    
    {
        'global_stats': {
            'file_type': [str, str], 
            'encoding': [str, str], 
            'samples_used': int, 
            'empty_line_count': int, 
            'memory_size': float
        }, 
        'data_stats': {
            'data_label': {
                'entity_counts': {
                    'word_level': dict[str, int], 
                    'true_char_level': dict[str, int], 
                    'postprocess_char_level': dict[str, int]
                }, 
                'entity_percentages': {
                    'word_level': dict[str, float], 
                    'true_char_level': dict[str, float], 
                    'postprocess_char_level': dict[str, float]
                }
            }, 
            'statistics': {
                'vocab': [list[str], list[str], list[str]], 
                'vocab_count': [dict[str, int], dict[str, int], dict[str, int]], 
                'words': [list[str], list[str], list[str]], 
                'word_count': [dict[str, int], dict[str, int], dict[str, int]]
            }
        }
    }
    

Saving and Loading a Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The profiles can easily be saved and loaded as shown below:

**NOTE: Json saving and loading only supports Structured Profiles currently.**

There are two save/load methods:

* **Pickle save/load**

  * Save a profile as a `.pkl` file.
  * Load a `.pkl` file as a profile object.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load a CSV file, with "," as the delimiter
    data = Data("your_file.csv")

    # Read data into profile
    profile = Profiler(data)

    # save structured profile to pkl file
    profile.save(filepath="my_profile.pkl")

    # load pkl file to structured profile
    loaded_pkl_profile = dp.Profiler.load(filepath="my_profile.pkl")

    print(json.dumps(loaded_pkl_profile.report(report_options={"output_format": "compact"}),
                                           indent=4))

* **Json save/load**

  * Save a profile as a human-readable `.json` file.
  * Load a `.json` file as a profile object.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load a CSV file, with "," as the delimiter
    data = Data("your_file.csv")

    # Read data into profile
    profile = Profiler(data)

    # save structured profile to json file
    profile.save(filepath="my_profile.json", save_method="json")

    # load json file to structured profile
    loaded_json_profile = dp.Profiler.load(filepath="my_profile.json", load_method="json")

    print(json.dumps(loaded_json_profile.report(report_options={"output_format": "compact"}),
                                           indent=4))


Structured vs Unstructured Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the profiler, the data profiler will automatically infer whether to
create the structured profile or the unstructured profile. However, you can be 
explicit as shown below: 

.. code-block:: python
    
    import json
    from dataprofiler import Data, Profiler
    
    # Creating a structured profile
    data1 = Data("normal_csv_file.csv")
    structured_profile = Profiler(data1, profiler_type="structured")
    
    structured_report = structured_profile.report(report_options={"output_format": "pretty"})
    print(json.dumps(structured_report, indent=4))
    
    # Creating an unstructured profile
    data2 = Data("normal_text_file.txt")
    unstructured_profile = Profiler(data2, profiler_type="unstructured")
    
    unstructured_report = unstructured_profile.report(report_options={"output_format": "pretty"})
    print(json.dumps(unstructured_report, indent=4))
    

Setting the Sample Size
~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to set sample size in a profile: samples_per_update and 
min_true_samples. Samples_per_update takes an integer as the exact amount that
will be sampled. Min_true_samples will set the minimum amount of samples that
are not null. For example:

.. code-block:: python

    from dataprofiler import Profiler
    
    sample_array = [1.0, NULL, 2.0]
    profile = dp.Profiler(sample_array, samples_per_update=2) 
    
The first two samples (1.0 and NULL) are used for the statistical analysis.
 
In contrast, if we also set min_true_samples to 2 then the Data Reader will 
continue to read until the minimum true samples were found for the given column.
For example: 

.. code-block:: python

    from dataprofiler import Profiler
    
    sample_array = [1.0, NULL, 2.0]
    profile = dp.Profiler(sample_array, samples_per_update=2, min_true_samples=2)
   
This will use all samples in the statistical analysis until the number of "true" 
(non-NULL) values are reached. Both min_true_samples and 
samples_per_update conditions must be met. In this case, the profile will grab
the first two samples (1.0 and NULL) to satisfy the samples_per_update, and then
it will grab the first two VALID samples (1.0 and 2.0) to satisfy the 
min_true_samples.

Profile a Pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import dataprofiler as dp
    import json

    my_dataframe = pd.DataFrame([[1, 2.0],[1, 2.2],[-1, 3]])
    profile = dp.Profiler(my_dataframe)

    # print the report using json to prettify.
    report = profile.report(report_options={"output_format": "pretty"})
    print(json.dumps(report, indent=4))

    # read a specified column, in this case it is labeled 0:
    print(json.dumps(report["data stats"][0], indent=4))


Specifying a Filetype or Delimiter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example of specifying a CSV data type, with a `,` delimiter.
In addition, it utilizes only the first 10,000 rows.

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler
    from dataprofiler.data_readers.csv_data import CSVData

    # Load a CSV file, with "," as the delimiter
    data = CSVData("your_file.csv", options={"delimiter": ","})

    # Split the data, such that only the first 10,000 rows are used
    data = data.data[0:10000]

    # Read in profile and print results
    profile = Profiler(data)
    print(json.dumps(profile.report(report_options={"output_format": "pretty"}), indent=4))

Setting Profiler Seed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example of specifying a seed for reproducibility.

.. code-block:: python

    import dataprofiler as dp

    # Set seed to non-negative integer value or None
    dp.set_seed(0)


Profile Statistic Descriptions
==============================

Structured Profile
~~~~~~~~~~~~~~~~~~

**global_stats**:

* samples_used - number of input data samples used to generate this profile
* column_count - the number of columns contained in the input dataset
* row_count - the number of rows contained in the input dataset
* row_has_null_ratio - the proportion of rows that contain at least one null value to the total number of rows
* row_is_null_ratio - the proportion of rows that are fully comprised of null values (null rows) to the total number of rows
* unique_row_ratio - the proportion of distinct rows in the input dataset to the total number of rows
* duplicate_row_count - the number of rows that occur more than once in the input dataset
* file_type - the format of the file containing the input dataset (ex: .csv)
* encoding - the encoding of the file containing the input dataset (ex: UTF-8)
* correlation_matrix - matrix of shape `column_count` x `column_count` containing the correlation coefficients between each column in the dataset 
* chi2_matrix - matrix of shape `column_count` x `column_count` containing the chi-square statistics between each column in the dataset
* profile_schema - a description of the format of the input dataset labeling each column and its index in the dataset
    * string - the label of the column in question and its index in the profile schema
* times - the duration of time it took to generate the global statistics for this dataset in milliseconds

**data_stats**:

* column_name - the label/title of this column in the input dataset
* data_type - the primitive python data type that is contained within this column
* data_label - the label/entity of the data in this column as determined by the Labeler component
* categorical - 'true' if this column contains categorical data
* order - the way in which the data in this column is ordered, if any, otherwise “random”
* samples - a small subset of data entries from this column
* statistics - statistical information on the column
    * sample_size - number of input data samples used to generate this profile
    * null_count - the number of null entries in the sample
    * null_types - a list of the different null types present within this sample
    * null_types_index - a dict containing each null type and a respective list of the indicies that it is present within this sample
    * data_type_representation - the percentage of samples used identifying as each data_type
    * min - minimum value in the sample
    * max - maximum value in the sample
    * mode - mode of the entries in the sample
    * median - median of the entries in the sample
    * median_absolute_deviation - the median absolute deviation of the entries in the sample
    * sum - the total of all sampled values from the column
    * mean - the average of all entries in the sample
    * variance - the variance of all entries in the sample
    * stddev - the standard deviation of all entries in the sample
    * skewness - the statistical skewness of all entries in the sample
    * kurtosis - the statistical kurtosis of all entries in the sample
    * num_zeros - the number of entries in this sample that have the value 0
    * num_negatives - the number of entries in this sample that have a value less than 0
    * histogram - contains histogram relevant information
        * bin_counts - the number of entries within each bin
        * bin_edges - the thresholds of each bin
    * quantiles - the value at each percentile in the order they are listed based on the entries in the sample
    * vocab - a list of the characters used within the entries in this sample
    * avg_predictions - average of the data label prediction confidences across all data points sampled
    * categories - a list of each distinct category within the sample if `categorial` = 'true'
    * unique_count - the number of distinct entries in the sample
    * unique_ratio - the proportion of the number of distinct entries in the sample to the total number of entries in the sample
    * categorical_count - number of entries sampled for each category if `categorical` = 'true'
    * gini_impurity - measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset
    * unalikeability - a value denoting how frequently entries differ from one another within the sample
    * precision - a dict of statistics with respect to the number of digits in a number for each sample
    * times - the duration of time it took to generate this sample's statistics in milliseconds
    * format - list of possible datetime formats
* null_replication_metrics - statistics of data partitioned based on whether column value is null (index 1 of lists referenced by dict keys) or not (index 0)
    * class_prior - a list containing probability of a column value being null and not null
    * class_sum - a list containing sum of all other rows based on whether column value is null or not
    * class_mean - a list containing mean of all other rows based on whether column value is null or not

Unstructured Profile
~~~~~~~~~~~~~~~~~~~~

**global_stats**:

* samples_used - number of input data samples used to generate this profile
* empty_line_count - the number of empty lines in the input data
* file_type - the file type of the input data (ex: .txt)
* encoding - file encoding of the input data file (ex: UTF-8)
* memory_size - size of the input data in MB
* times - duration of time it took to generate this profile in milliseconds

**data_stats**:

* data_label - labels and statistics on the labels of the input data
    * entity_counts - the number of times a specific label or entity appears inside the input data
        * word_level - the number of words counted within each label or entity
        * true_char_level - the number of characters counted within each label or entity as determined by the model
        * postprocess_char_level - the number of characters counted within each label or entity as determined by the postprocessor
    * entity_percentages - the percentages of each label or entity within the input data
        * word_level - the percentage of words in the input data that are contained within each label or entity
        * true_char_level - the percentage of characters in the input data that are contained within each label or entity as determined by the model
        * postprocess_char_level - the percentage of characters in the input data that are contained within each label or entity as determined by the postprocessor
    * times - the duration of time it took for the data labeler to predict on the data
* statistics - statistics of the input data
    * vocab - a list of each character in the input data
    * vocab_count - the number of occurrences of each distinct character in the input data
    * words - a list of each word in the input data
    * word_count - the number of occurrences of each distinct word in the input data
    * times - the duration of time it took to generate the vocab and words statistics in milliseconds

Graph Profile
~~~~~~~~~~~~~~~~~~

* num_nodes - number of nodes in the graph
* num_edges - number of edges in the graph
* categorical_attributes - list of categorical edge attributes
* continuous_attributes - list of continuous edge attributes
* avg_node_degree - average degree of nodes in the graph
* global_max_component_size: size of the global max component

**continuous_distribution**:

* <attribute_N>: name of N-th edge attribute in list of attributes
    * name - name of distribution for attribute
    * scale - negative log likelihood used to scale and compare distributions
    * properties - list of statistical properties describing the distribution
        * [shape (optional), loc, scale, mean, variance, skew, kurtosis]

**categorical_distribution**:

* <attribute_N>: name of N-th edge attribute in list of attributes
    * bin_counts: counts in each bin of the distribution histogram
    * bin_edges: edges of each bin of the distribution histogram

* times - duration of time it took to generate this profile in milliseconds

Profile Options
===============

The data profiler accepts several options to toggle on and off 
features. The 8 columns (int options, float options, datetime options,
text options, order options, category options, data labeler options) can be 
enabled or disabled. By default, all options are toggled on. Below is an example
of how to alter these options. Options shared by structured and unstructured options
must be specified as structured or unstructured when setting (ie. datalabeler options).


.. code-block:: python

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
    profile_options.set({"structured_options.data_labeler.is_enabled": False, "min.is_enabled": False})

    # Specific columns can be set/disabled/enabled in the same way
    profile_options.structured_options.text.set({"max.is_enabled":True, 
                                             "variance.is_enabled": True})

    # numeric stats can be turned off/on entirely
    profile_options.set({"is_numeric_stats_enabled": False})
    profile_options.set({"int.is_numeric_stats_enabled": False})

    profile = Profiler(data, options=profile_options)

    # Print the report using json to prettify.
    report  = profile.report(report_options={"output_format": "pretty"})
    print(json.dumps(report, indent=4))


Below is an breakdown of all the options.

* **ProfilerOptions** - The top-level options class that contains options for the Profiler class

  * **presets** - A pre-configured mapping of a string name to group of options:

    * **default is None**

    * **"complete"**

    .. code-block:: python

        options = ProfilerOptions(presets="complete")

    * **"data_types"**

    .. code-block:: python

        options = ProfilerOptions(presets="data_types")

    * **"numeric_stats_disabled"**

    .. code-block:: python

        options = ProfilerOptions(presets="numeric_stats_disabled")

    * **"lower_memory_sketching"**

    .. code-block:: python

        options = ProfilerOptions(presets="lower_memory_sketching")

  * **structured_options** - Options responsible for all structured data

    * **multiprocess** - Option to enable multiprocessing. If on, multiprocessing is toggled on if the dataset contains more than 750,000 rows or more than 20 columns.
      Automatically selects the optimal number of pooling processes to utilize based on system constraints when toggled on.

      * is_enabled - (Boolean) Enables or disables multiprocessing

    * **sampling_ratio** - A percentage, as a decimal, ranging from greater than 0 to less than or equal to 1 indicating how much input data to sample. Default value set to 0.2.

    * **int** - Options for the integer columns

      * is_enabled - (Boolean) Enables or disables the integer operations
      * min - Finds minimum value in a column

      * is_enabled - (Boolean) Enables or disables min
      * max - Finds maximum value in a column

        * is_enabled - (Boolean) Enables or disables max
      * mode - Finds mode(s) in a column

        * is_enabled - (Boolean) Enables or disables mode
        * top_k_modes - (Int) Sets the number of modes to return if multiple exist. Default returns max 5 modes.
      * median - Finds median value in a column

        * is_enabled - (Boolean) Enables or disables median
      * sum - Finds sum of all values in a column

        * is_enabled - (Boolean) Enables or disables sum

      * variance - Finds variance of all values in a column

        * is_enabled - (Boolean) Enables or disables variance
      * skewness - Finds skewness of all values in a column

        * is_enabled - (Boolean) Enables or disables skewness
      * kurtosis - Finds kurtosis of all values in a column

        * is_enabled - (Boolean) Enables or disables kurtosis
      * median_abs_deviation - Finds median absolute deviation of all values in a column

        * is_enabled - (Boolean) Enables or disables median absolute deviation
      * num_zeros - Finds the count of zeros in a column

        * is_enabled - (Boolean) Enables or disables num_zeros
      * num_negatives - Finds the count of negative numbers in a column

        * is_enabled - (Boolean) Enables or disables num_negatives
      * bias_correction - Applies bias correction to variance, skewness, and kurtosis calculations

        * is_enabled - (Boolean) Enables or disables bias correction
      * histogram_and_quantiles - Generates a histogram and quantiles
        from the column values

        * bin_count_or_method - (String/List[String]) Designates preferred method for calculating histogram bins or the number of bins to use.  
          If left unspecified (None) the optimal method will be chosen by attempting all methods.  
          If multiple specified (list) the optimal method will be chosen by attempting the provided ones.  
          methods: 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'  
          Note: 'auto' is used to choose optimally between 'fd' and 'sturges'
        * num_quantiles - (Int) Number of quantiles to bin the data. 
          Default value is set to 1,000 quantiles.
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
      * mode - Finds mode(s) in a column

        * is_enabled - (Boolean) Enables or disables mode
        * top_k_modes - (Int) Sets the number of modes to return if multiple exist. Default returns max 5 modes.
      * median - Finds median value in a column

        * is_enabled - (Boolean) Enables or disables median
      * sum - Finds sum of all values in a column

        * is_enabled - (Boolean) Enables or disables sum
      * variance - Finds variance of all values in a column

        * is_enabled - (Boolean) Enables or disables variance
      * skewness - Finds skewness of all values in a column

        * is_enabled - (Boolean) Enables or disables skewness
      * kurtosis - Finds kurtosis of all values in a column

        * is_enabled - (Boolean) Enables or disables kurtosis
      * median_abs_deviation - Finds median absolute deviation of all values in a column

        * is_enabled - (Boolean) Enables or disables median absolute deviation
      * is_numeric_stats_enabled - (Boolean) enable or disable all numeric stats
      * num_zeros - Finds the count of zeros in a column

        * is_enabled - (Boolean) Enables or disables num_zeros
      * num_negatives - Finds the count of negative numbers in a column

        * is_enabled - (Boolean) Enables or disables num_negatives
      * bias_correction - Applies bias correction to variance, skewness, and kurtosis calculations

        * is_enabled - (Boolean) Enables or disables bias correction
      * histogram_and_quantiles - Generates a histogram and quantiles
        from the column values

        * bin_count_or_method - (String/List[String]) Designates preferred method for calculating histogram bins or the number of bins to use.  
          If left unspecified (None) the optimal method will be chosen by attempting all methods.  
          If multiple specified (list) the optimal method will be chosen by attempting the provided ones.  
          methods: 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'  
          Note: 'auto' is used to choose optimally between 'fd' and 'sturges'
        * num_quantiles - (Int) Number of quantiles to bin the data. 
          Default value is set to 1,000 quantiles.
        * is_enabled - (Boolean) Enables or disables histogram and quantiles        
    * **text** - Options for the text columns

      * is_enabled - (Boolean) Enables or disables the text operations
      * vocab - Finds all the unique characters used in a column

        * is_enabled - (Boolean) Enables or disables vocab
      * min - Finds minimum value in a column

        * is_enabled - (Boolean) Enables or disables min
      * max - Finds maximum value in a column

        * is_enabled - (Boolean) Enables or disables max
      * mode - Finds mode(s) in a column

        * is_enabled - (Boolean) Enables or disables mode
        * top_k_modes - (Int) Sets the number of modes to return if multiple exist. Default returns max 5 modes.
      * median - Finds median value in a column

        * is_enabled - (Boolean) Enables or disables median
      * sum - Finds sum of all values in a column

        * is_enabled - (Boolean) Enables or disables sum
      * variance - Finds variance of all values in a column

        * is_enabled - (Boolean) Enables or disables variance
      * skewness - Finds skewness of all values in a column

        * is_enabled - (Boolean) Enables or disables skewness
      * kurtosis - Finds kurtosis of all values in a column

        * is_enabled - (Boolean) Enables or disables kurtosis
      * median_abs_deviation - Finds median absolute deviation of all values in a column

        * is_enabled - (Boolean) Enables or disables median absolute deviation
      * bias_correction - Applies bias correction to variance, skewness, and kurtosis calculations

        * is_enabled - (Boolean) Enables or disables bias correction
      * is_numeric_stats_enabled - (Boolean) enable or disable all numeric stats
      * num_zeros - Finds the count of zeros in a column

        * is_enabled - (Boolean) Enables or disables num_zeros
      * num_negatives - Finds the count of negative numbers in a column

        * is_enabled - (Boolean) Enables or disables num_negatives
      * histogram_and_quantiles - Generates a histogram and quantiles
        from the column values

        * bin_count_or_method - (String/List[String]) Designates preferred method for calculating histogram bins or the number of bins to use.  
          If left unspecified (None) the optimal method will be chosen by attempting all methods.  
          If multiple specified (list) the optimal method will be chosen by attempting the provided ones.  
          methods: 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'  
          Note: 'auto' is used to choose optimally between 'fd' and 'sturges'
        * num_quantiles - (Int) Number of quantiles to bin the data. 
          Default value is set to 1,000 quantiles.
        * is_enabled - (Boolean) Enables or disables histogram and quantiles  
    * **datetime** - Options for the datetime columns

      * is_enabled - (Boolean) Enables or disables the datetime operations
    * **order** - Options for the order columns

      * is_enabled - (Boolean) Enables or disables the order operations
    * **category** - Options for the category columns

      * is_enabled  - (Boolean) Enables or disables the category operations
      * top_k_categories - (int) Number of categories to be displayed when reporting
      * max_sample_size_to_check_stop_condition - (int) The maximum sample size before categorical stop conditions are checked
      * stop_condition_unique_value_ratio - (float) The highest ratio of unique values to dataset size that is to be considered a categorical type
      * cms - (Boolean) Enables or Disables the use of count min sketch / heavy hitters for approximate frequency counts
      * cms_confidence - (float) Defines the number of hashes used in CMS, default 0.95
      * cms_relative_error - (float) Defines the number of buckets used in CMS, default 0.01
      * cms_max_num_heavy_hitters - (int) The value used to define the threshold for minimum frequency required by a category to be counted
    * **data_labeler** - Options for the data labeler columns

      * is_enabled - (Boolean) Enables or disables the data labeler operations
      * data_labeler_dirpath - (String) Directory path to data labeler
      * data_labeler_object - (BaseDataLabeler) Datalabeler to replace 
        the default labeler 
      * max_sample_size - (Int) The max number of samples for the data
        labeler
    * **correlation** - Option set for correlation profiling
      * is_enabled - (Boolean) Enables or disables performing correlation profiling
      * columns - Columns considered to calculate correlation
    * **row_statistics** - (Boolean) Option to enable/disable row statistics calculations

      * unique_count - (UniqueCountOptions) Option to enable/disable unique row count calculations

        * is_enabled - (Bool) Enables or disables options for unique row count
        * hashing_method - (String) Property to specify row hashing method ("full" | "hll")
        * hll - (HyperLogLogOptions) Options for alternative method of estimating unique row count (activated when `hll` is the selected hashing_method)

          * seed - (Int) Used to set HLL hashing function seed
          * register_count - (Int) Number of registers is equal to 2^register_count

      * null_count - (Boolean) Option to enable/disable functionalities for row_has_null_ratio and row_is_null_ratio
    * **chi2_homogeneity** - Options for the chi-squared test matrix

      * is_enabled - (Boolean) Enables or disables performing chi-squared tests for homogeneity between the categorical columns of the dataset.
    * **null_replication_metrics** - Options for calculating null replication metrics

      * is_enabled - (Boolean) Enables or disables calculation of null replication metrics
  * **unstructured_options** - Options responsible for all unstructured data

    * **text** - Options for the text profile
      
      * is_case_sensitive - (Boolean) Specify whether the profile is case sensitive
      * stop_words - (List of Strings) List of stop words to be removed when profiling
      * top_k_chars - (Int) Number of top characters to be retrieved when profiling
      * top_k_words - (Int) Number of top words to be retrieved when profiling
      * vocab - Options for vocab count

        * is_enabled - (Boolean) Enables or disables the vocab stats
      * words - Options for word count

        * is_enabled - (Boolean) Enables or disables the word stats
    * **data_labeler** - Options for the data labeler

      * is_enabled - (Boolean) Enables or disables the data labeler operations
      * data_labeler_dirpath - (String) Directory path to data labeler
      * data_labeler_object - (BaseDataLabeler) Datalabeler to replace 
        the default labeler 
      * max_sample_size - (Int) The max number of samples for the data 
        labeler
    


Statistical Dependency on Order of Updates
==========================================

Some profile features/statistics are dependent on the order in which the profiler
is updated with new data.

Order Profile
~~~~~~~~~~~~~

The order profiler utilizes the last value in the previous data batch to ensure
the subsequent dataset is above/below/equal to that value when predicting
non-random order.

For instance, a dataset to be predicted as ascending would require the following
batch data update to be ascending and its first value `>=` than that of the
previous batch of data.

Ex. of ascending:

.. code-block:: python

    batch_1 = [0, 1, 2]
    batch_2 = [3, 4, 5]

Ex. of random:

.. code-block:: python

    batch_1 = [0, 1, 2]
    batch_2 = [1, 2, 3] # notice how the first value is less than the last value in the previous batch


Reporting Structure
===================

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

.. code-block:: python

    report  = profile.report(report_options={"output_format": "pretty"})
    report  = profile.report(report_options={"output_format": "compact"})
    report  = profile.report(report_options={"output_format": "serializable"})
    report  = profile.report(report_options={"output_format": "flat"})
