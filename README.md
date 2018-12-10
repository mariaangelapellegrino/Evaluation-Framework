# Evaluation-Framework

This repository contains an evaluation framework to test RDF embedding techniques upon to machine learning (ML) and semantic tasks.
The implemented tasks are:
- Machine Learning
    - Classification
    - Regression
    - Clustering
- Semantic tasks
    - Entity Relatedness
    - Document Similarity
    - Semantic Analogies
    
## Licence
The Apache licence applies to the provided source code. For the datasets, please check the licensing information. 
For example, for the licensing information of the classification and regression datasets, see https://dws.informatik.uni-mannheim.de/en/research/a-collection-of-benchmark-datasets-for-ml/; 
for Entity Relatedness task, see https://old.datahub.io/dataset/kore-50-nif-ner-corpus/resource/840dc999-8451-42d8-baaf-0647f1bc6a20.

## How to run the code? 
Environment: 
- Python version: Python 2.7
- Libraries: (output of _pip freeze_ of my virtual environment)
    - numpy==1.14.0
    - pandas==0.22.0
    - scikit-learn==0.19.2
    - scipy==1.1.0

Parameters:
- --vectors_file, mandatory, Path of the vectors file. 
- --vectors_size, default=200, Length of each vector
- --top_k, default=2, Used in the SemanticAnalogies task: The predicted vector will be compared with the top k closest vectors to establish if the prediction is correct or not

Needed: run _main.py_ providing at least _--vectors_file_ as a parameter.

Note: The tasks can be executed sequentially or in parallel. If the code raises MemoryError it means that the tasks need more memory than the one available. In that case, run all the tasks sequentially.

## Supported vectors file format
The two folders contain two versions of the framework which differ only for the vectors file format. 
In the TXT_version the vectors file must be a TSV - tab or space separated values - with one line for each entity. 
Each line has to contain the entity name - without angular backets - and the embedded vector. 
In the HDF5_version the vectors file must be an H5 with one group in it called "Vectors". 
In this group, there should be one dataset for each entity with the entity name, base32 encoded, as dataset name and the embedded vector as dataset content.

## Project structure
_main.py_ instantiates the distance function to measure the distance between two vectors and the analogy function used in Semantic Analogies task.
It manages the parameters and instantiates the evaluator_manager. 

The _evaluator\_manager.py_ reads the vectors file, runs all the tasks sequentially or in parallel and creates the output directory calling it results\_YYYY-MM-DD\_HH-MM-SS.

Each task is in a separate folder and each of them is constituted by:
    a manager that supervises the work and organizes the output,
    a data\_manager that reads the files used as gold standard and merges them with the actual vectors,
    a model that computes the task and provides the output to the manager.

## How to customize distance and analogy function
You have to redefine your own main.
    
You can use one of the **_distance metric_** accepted by [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).

Your **_analogy function_** has to take 

- 3 vectors or matrices of vectors used to forecast the fourth vector,
- the index (or indices) or these vectors in the data matrix
- the data matrixes that contains all the vectors
- the top_k, i.e., the number of vectors you want to use to check if the predicted vector is close to one in your dataset

and it must return the indices of the top_k closest vector to the predicted one.

## History
To get the Evaluation Framework history, you can follow this link: https://git.rwth-aachen.de/KGEmbedding/evaluationFramework.
