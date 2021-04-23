.. _profiler:

Profiler
********

Profile Options
===============

The data profiler accepts several options to toggle on and off 
features. The 8 columns (int options, float options, datetime options,
text options, order options, category options, data labeler options) can be 
enabled or disabled. By default, all options are toggled on. Below is an example
of how to alter these options. 

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

