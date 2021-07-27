Visit our [documentation page.](https://github.cloud.capitalone.com/pages/data-innovation/data-profiler/)

### How to properly write documentation:  

#### Packages  
In any package directory, overall package comments can be made in the 
\_\_init\_\_.py of the directory. At the top of the \_\_init\_\_.py,
include your comments in between triple quotations.

#### Classes  
In any class file, include overall class comments at the top of the file
in between triple quotes and/or in the init function.

#### Functions  
reStructuredText Docstring Format is the standard. Here is an example:

    def format_data(self, predictions, verbose=False):
        """
        Formats word level labeling of the Unstructured Data Labeler as you want
        
        :param predictions: A 2D list of word level predictions/labeling
        :type predictions: Dict
        :param verbose: A flag to determine verbosity
        :type verbose: Bool
        :return: JSON structure containing specified formatted output
        :rtype: JSON
        
        :Example:
            Look at this test. Don't forget the double colons to make a code block::
                This is a codeblock
                Type example code here
        """

**To install the full package from pypi**: `pip install DataProfiler[ml]`

If the ML requirements are too strict (say, you don't want to install tensorflow), you can install a slimmer package. The slimmer package disables the default sensitive data detection / entity recognition (labler)

Install from pypi: `pip install DataProfiler`


------------------

# What is a Data Profile?

To update the documentation of a feature branch, go to the /docs folder
and run:
```bash
python update_documentation.py
```

Make sure you run sphinx version Sphinx==3.5.4 since the Furo theme library
doesn't work with the latest version of Sphinx.
