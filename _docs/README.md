Visit our [documentation page.](https://capitalone.github.io/DataProfiler)

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

### How to update the documentation:


1. Set up your local environment
```bash
# install sphinx requirements
# install the requirements from the feature branch
pip install pandoc &&
pip install -r requirements.txt &&
pip install -r requirements-ml.txt && 
pip install -r requirements-reports.txt && 
pip install -r requirements-docs.txt  &&
pip install -e . 

```
2. And finally, from the root of `DataProfiler`, run the following commands to generate the sphinx documentation:
```bash
cd _docs/docs
python update_documentation.py

```

3. View new docs
```bash
open index.html
```
