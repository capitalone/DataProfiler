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

### How to update the documentation:

To update the docs branch, checkout the gh-pages branch. Make sure it is up to
date, then clone the dataprofiler from the feature branch you want in order to
update the documentation (probably main).

    # clone the feature branch into the base of this repository
    git clone https://github.com/capitalone/DataProfiler feature_branch

Install requirements:

    # install sphinx requirements
    brew install pandoc
    pip install requirements.txt
    
    # install the requirements from the feature branch
    pip install feature_branch/requirements.txt
    pip install feature_branch/requirements-ml.txt
    pip install feature_branch/requirements-reports.txt

In /docs run:

    cd docs
    python update_documentation.py

If you make adjustments to the code comments, you may rerun the command again to
 overwrite the specified version. 

Once the documentation is updated, commit and push the whole 
/docs folder. API documentation will only update when pushed to the main 
branch. 

If you make a mistake naming the version, you will have to delete it from
the /docs/source/index.rst file.

To update the documentation of a feature branch, go to the /docs folder
and run:
```bash
cd docs
python update_documentation.py
```







