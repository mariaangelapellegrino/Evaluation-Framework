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
