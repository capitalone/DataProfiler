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

1. Either with an existing clone of `capitalone/DataProfiler` or clone the `capitalone/DataProfiler` reposotory to your local computer with the following command:
```bash
git clone https://github.com/capitalone/DataProfiler
```

2. Next ensure that `gh-pages` branch is checked out in `DataProfiler` repository folder:
```bash
cd DataProfiler
git checkout gh-pages
```

3. Next inside `DataProfiler` repo that we just cloned down to your local machine, clone the repository under the alias `feature_branch` inside the root of `DataProfiler` from step one:
```bash
git clone https://github.com/capitalone/DataProfiler feature_branch
```

4. Still in the root of `DataProfiler`, install the requirements needed for generating the documentation:
```bash
# install sphinx requirements
brew install pandoc
pip install -r requirements.txt
```

and

```bash
# install the requirements from the feature branch
pip install -r feature_branch/requirements.txt
pip install -r feature_branch/requirements-ml.txt
pip install -r feature_branch/requirements-reports.txt
```

5. Then install the pre-commit hooks by running the following:
```bash
pre-commit install
```

6. And finally, from the root of `DataProfiler`, run the following commands to generate the sphinx documentation:
```bash
cd docs/
python update_documentation.py
```

If you make adjustments to the code comments, you may rerun the command again to overwrite the specified version.

Once the documentation is updated, commit and push the whole
/docs folder as well as the index.html file. API documentation
will only update when pushed to the main branch.

If you make a mistake naming the version, you will have to delete it from
the /docs/source/index.rst file.

To update the documentation of a feature branch, go to the /docs folder
and run:
```bash
cd docs
python update_documentation.py
```
