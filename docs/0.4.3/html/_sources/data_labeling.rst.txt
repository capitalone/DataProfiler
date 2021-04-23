.. _data_labeling:

Labeling (Sensitive Data)
*************************

In this library, the term *data labeling* refers to entity recognition.

Builtin to the data profiler is a classifier which evaluates the complex data types of the dataset.
For structured data, it determines the complex data type of each column. When
running the data profile, it uses the default data labeling model builtin to the
library. However, the data labeler allows users to train their own data labeler
as well.

Identify Entities in Structured Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Makes predictions and identifying labels:

.. code-block:: python

    import dataprofiler as dp

    # load data and data labeler
    data = dp.Data("your_data.csv")
    data_labeler = dp.DataLabeler(labeler_type='structured')

    # make predictions and get labels per cell
    predictions = data_labeler.predict(data)

Identify Entities in Unstructured Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict which class characters belong to in unstructured text:

.. code-block:: python

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

It's also possible to change output formats, output similar to a **SpaCy** format:

.. code-block:: python

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

Train a New Data Labeler
~~~~~~~~~~~~~~~~~~~~~~~~

Mechanism for training your own data labeler on their own set of structured data
(tabular):

.. code-block:: python
    
    import dataprofiler as dp

    # Will need one column with a default label of UNKNOWN
    data = dp.Data("your_file.csv")

    data_labeler = dp.train_structured_labeler(
        data=data,
        save_dirpath="/path/to/save/labeler",
        epochs=2
    )

    data_labeler.save_to_disk("my/save/path") # Saves the data labeler for reuse

Load an Existing Data Labeler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mechanism for loading an existing data_labeler:

.. code-block:: python

    import dataprofiler as dp

    data_labeler = dp.DataLabeler(
        labeler_type='structured', dirpath="/path/to/my/labeler")

    # get information about the parameters/inputs/output formats for the DataLabeler
    data_labeler.help()

Extending a Data Labeler with Transfer Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extending or changing labels of a data labeler w/ transfer learning:
Note: By default, **a labeler loaded will not be trainable**. In order to load a 
trainable DataLabeler, the user must set `trainable=True` or load a labeler 
using the `TrainableDataLabeler` class.

The following illustrates how to change the labels:

.. code-block:: python

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

The following illustrates how to extend the labels:

.. code-block:: python

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


Changing pipeline parameters:

.. code-block:: python

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


Build Your Own Data Labeler
===========================

The DataLabeler has 3 main components: preprocessor, model, and postprocessor. 
To create your own DataLabeler, each one would have to be created or an 
existing component can be reused.

Given a set of the 3 components, you can construct your own DataLabeler:

.. code-block:: python
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

Option for swapping out specific components of an existing labeler.

.. code-block:: python

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


Model Component
~~~~~~~~~~~~~~~

In order to create your own model component for data labeling, you can utilize 
the `BaseModel` class from `dataprofiler.labelers.base_model` and
overriding the abstract class methods.

Reviewing `CharacterLevelCnnModel` from 
`dataprofiler.labelers.character_level_cnn_model` illustrates the functions 
which need an override. 

#. `__init__`: specifying default parameters and calling base `__init__`
#. `_validate_parameters`: validating parameters given by user during setting
#. `_need_to_reconstruct_model`: flag for when to reconstruct a model (i.e. 
   parameters change or labels change require a model reconstruction)
#. `_construct_model`: initial construction of the model given the parameters
#. `_reconstruct_model`: updates model architecture for new label set while 
   maintaining current model weights
#. `fit`: mechanism for the model to learn given training data
#. `predict`: mechanism for model to make predictions on data
#. `details`: prints a summary of the model construction
#. `save_to_disk`: saves model and model parameters to disk
#. `load_from_disk`: loads model given a path on disk
  
  
Preprocessor Component
~~~~~~~~~~~~~~~~~~~~~~

In order to create your own preprocessor component for data labeling, you can 
utilize the `BaseDataPreprocessor` class 
from `dataprofiler.labelers.data_processing` and override the abstract class 
methods.

Reviewing `StructCharPreprocessor` from 
`dataprofiler.labelers.data_processing` illustrates the functions which 
need an override.

#. `__init__`: passing parameters to the base class and executing any 
   extraneous calculations to be saved as parameters
#. `_validate_parameters`: validating parameters given by user during
   setting
#. `process`: takes in the user data and converts it into an digestible, 
   iterable format for the model
#. `set_params` (optional): if a parameter requires processing before setting,
   a user can override this function to assist with setting the parameter
#. `_save_processor` (optional): if a parameter is not JSON serializable, a 
   user can override this function to assist in saving the processor and its 
   parameters
#. `load_from_disk` (optional): if a parameter(s) is not JSON serializable, a 
   user can override this function to assist in loading the processor 

Postprocessor Component
~~~~~~~~~~~~~~~~~~~~~~~

The postprocessor is nearly identical to the preprocessor except it handles 
the output of the model for processing. In order to create your own 
postprocessor component for data  labeling, you can utilize the 
`BaseDataPostprocessor` class from  `dataprofiler.labelers.data_processing` 
and override the abstract class methods.

Reviewing `StructCharPostprocessor` from 
`dataprofiler.labelers.data_processing` illustrates the functions which 
need an override.

#. `__init__`: passing parameters to the base class and executing any 
   extraneous calculations to be saved as parameters
#. `_validate_parameters`: validating parameters given by user during
   setting
#. `process`: takes in the output of the model and processes for output to 
   the user
#. `set_params` (optional): if a parameter requires processing before setting,
   a user can override this function to assist with setting the parameter 
#. `_save_processor` (optional): if a parameter is not JSON serializable, a 
   user can override this function to assist in saving the processor and its 
   parameters
#. `load_from_disk` (optional): if a parameter(s) is not JSON serializable, a 
   user can override this function to assist in loading the processor 
