Evaluation Framework
====================

This repository contains a (software) *evaluation framework* to perform evaluation and comparison on *node embedding techniques*. It can be easily extended by also considering edges into the evaluation. The provided tasks range from Machine Learning (ML) (*classification, regression, and clustering*) and semantic tasks (*entity relatedness and document similarity*) to *semantic analogies*. The framework is designed to be extended with additional tasks. It is  useful both for embedding algorithm developers and users. On one side, when a new embedding algorithm is defined, there is the need to evaluate it upon tasks it was created for. On the other side, users can be interested in performing particular tests and choosing the embedding algorithm that performs best for their application. Our goal is to address both situations providing a ready-to-use framework that can be customized and easily extended. A preliminary overview of the framework and its results can be accessed through `A Configurable Evaluation Framework for Node Embedding Techniques <https://link.springer.com/chapter/10.1007%2F978-3-030-32327-1_31>`_.

We provide the framework as a command-line tool and we are working to the REST API development (you can inspect the *dev* branch).

Please refer https://mariaangelapellegrino.github.io/Evaluation-Framework/ to access the complete readme as a webpage.

Framework structure and extension points
----------------------------------------

It is a diagrammatic representation of the involved actors in the framework and their interactions. The blue boxes represent abstract classes, while the white boxes represent concrete classes. If A `<<extends>>` B, A is the concrete class that extends and makes the abstract behavior of A concrete. If A `<<instantiates>>` B, A creates an instance of B. `<<uses>>` B, A is dependent on B. 

.. image:: https://github.com/mariaangelapellegrino/Evaluation-Framework/blob/master/images/framework.png?raw=true


The starting point of the evaluation is the `Evaluation Manager` which is the orchestrator of the whole evaluation and it is in charge of 

1. verify the correctness of the parameters set by the user, 
2. instantiating the correct data manager according to the data format provided by the user, 
3. determining which task(s) the user asked for, 
4. managing the storage of the results. 

Depending on the file format, the corresponding `data manager` decides how to read the vector file content and how to manage the access to it (e.g., if the whole content has to be load in memory). Each `data manager` has to 

1. manage the reading of the gold standard datasets, 
2. manage the reading of the input file,
3. determine how to merge each gold standard dataset and the input file. 

The behavior of the data manager is modeled by the `abstract data manager`, implemented by a concrete data manager based on the input file format and it refined by a task data manager.

Each task is modeled as a pair of `task manager` and `model`. 
The `task manager` is in charge of 

1) merging the input file and each gold standard file (if more than one is provided) (by exploiting the data manager), 
2) instantiating and training a model for each configuration to test, 
3) collecting and storing results computed by the model. 

Each task can decide if the missing entities (i.e., the entities required into the gold standard file, but absent into the input file) will affect the final result of the task or not. 

To extend the evaluation also to edges, it is enough to create a gold-standard dataset containing edges and related ground truth.

Tasks
-----

The implemented tasks are:

- Machine Learning 

  - Classification 
  - Regression
  - Clustering

- Semantic tasks  

  - Entity Relatedness 
  - Document Similarity 
  - Semantic Analogies
    
Each task follows the same workflow:

1.  the task manager asks data manager to merge each gold standard dataset and the input file and keeps track of both the retrieved vectors and the **missing entities**,  i.e.,  entities  required  by  the  gold  standard  dataset,  but  absent  inthe input file;
2.  a model for each configuration is instantiated and trained;
3.  the missing entities are managed: it is up to the task to decide if they should affect the final result or they can be simply ignored;
4.  the scores are calculated and stored.

We will separately analyse each task, by detailing the gold standard datasets, the configuration of the model(s), and the computed evaluation metrics.

Framework details
------------------
Parameters
----------
The parameters of the evaluation are:

- **vectors\_file** : vector file path, *mandatory*
- **vector\_file\_format** : {TXT, HDF5}, *default TXT*
- **vectors\_size** : numeric value, *default 200*  
- **tasks**: list of *Classification*, *Regression*, *Clustering*, *EntityRelatedness*, *SemanticAnalogies* and *DocumentSimilarity*, *default _all*
- **parallel** : boolean, *default False*
- **debugging\_mode** : boolean, *default False*                                        |           - **similarity\_metric** : `Sklearn affinity metrics <https://scikit-learn.org/stable/modules/classes.html\#module-sklearn.metrics.pairwise>`_, *default Cosine*
- **analogy\_function**  : None (to use the _default\_analogy\_function_), otherwise a function handler. The semantic analogy tasks takes a quadruplet of vectors (a,b,c,d) and it verifies if by manipulating the first three vectors it is possible to predict the last one. The manipulation happens by the analogy function. *def default_analogy_function(a,b,c){return b-a+c}*  Any other handler to function has to take as input 3 vectors and return a predicted vector, which will be compared with the last vector in each semantic quadruplet.
- **top\_k** numeric value, *default 2*. It is used by the semantic analogy task: the predicted vector is compared with the top\_k closest vectors. If the actual _d_ vector is among those, the task is considered correct.
- **compare\_with** : list of run IDs, *default _all*

Vector file format
------------------
The input file can be provided either as a plain text (also called **TXT**) file or as an `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_.

The **TXT** file must be a white-space separated value file with a line for each embedded entity. Each row must contain the IRI of the embedded entity - without angular brackets - and its vector representation. 

The **HDF5** vectors file must be an H5 file with a single `group` called `Vectors`. 
In this group, there must be a `dataset` for each entity with the `base32 encoding` of the entity named as the dataset name and the embedded vector as its value.

Running details
---------------

Users can customize the evaluation settings by: 

1) specifying parameters on the command line (useful when only a few settings must be specified and the user desires to use the default value for most of the parameters);
2) organizing them in an XML file (especially useful when there is the need to define most of the parameters); 
3) passing them to a function that starts the evaluation. 

In the **example** folder of the project on GitHub, there are examples for the different ways to provide the parameters.

To execute one of them you can move the desired *main* file at the top level of the project and then run it.

**Note**: The tasks can be executed sequentially or in parallel. If the code raises MemoryError it means that the tasks need more memory than the one available. In that case, run all the tasks sequentially.

Results storage
---------------

For each task and for each file used as a gold standard, the framework will create 

1) an output file that contains a reference to the file used as a gold standard and all the information related to evaluation metric(s) provided by each task, 
2) a file containing all the **missing** entities, 
3) a log file reporting extra information, occurred problems, and execution time, 
4) information related to the comparison with previous runs. 

In particular, about the comparison, it reports the values effectively considered in the comparison and the ranking of the current run upon the other ones. The results of each run are stored in the directory *results/result\_<starting time of the execution>* generated by the evaluation manager in the local path.