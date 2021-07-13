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


Saving and Loading a Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The profiles can easily be saved and loaded as shown below:

.. code-block:: python

    import json
    from dataprofiler import Data, Profiler

    # Load a CSV file, with "," as the delimiter
    data = Data("your_file.csv")

    # Read in profile and print results
    profile = Profiler(data)
    profile.save(filepath="my_profile.pkl")
    
    loaded_profile = dp.Profiler.load("my_profile.pkl")
    print(json.dumps(loaded_profile.report(report_options={"output_format": "compact"}), 
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
      * skewness - Finds skewness of all values in a column

        * is_enabled - (Boolean) Enables or disables skewness
      * kurtosis - Finds kurtosis of all values in a column

        * is_enabled - (Boolean) Enables or disables kurtosis
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
      * skewness - Finds skewness of all values in a column

        * is_enabled - (Boolean) Enables or disables skewness
      * kurtosis - Finds kurtosis of all values in a column

        * is_enabled - (Boolean) Enables or disables kurtosis
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
      * skewness - Finds skewness of all values in a column

        * is_enabled - (Boolean) Enables or disables skewness
      * kurtosis - Finds kurtosis of all values in a column

        * is_enabled - (Boolean) Enables or disables kurtosis
      * bias_correction - Applies bias correction to variance, skewness, and kurtosis calculations

        * is_enabled - (Boolean) Enables or disables bias correction
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

