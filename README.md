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
    
## License
The Apache license applies to the provided source code. For the datasets, please check the licensing information. 
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
    - h5py==2.8.0

Parameters:
    - _vectors\_file_ the path of the file where the vectors are stored. It is mandatory
	- _vector\_file\_format_ TXT for text format, HDF5 for HDF5. default: TXT
	- _vectors\_size_ is the length of each vector. default: 200
	- _tasks_ is the list of the tasks to run. default: \_all, i.e., run all tasks.
	- _parallel_ True to run the tasks in parallel, False otherwise.  default: False
	- _debugging\_mode_ True to run the tasks by reporting all the information collected during the run, False otherwise.  default: False
	- _similarit\_metric_ the metric used to compute the distance among vectors. The valid values are provided by [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    default: cosine
	- _analogy\_function_ the function to compute the analogy among vectors. default: None to use the default function
	- _top\_k_ represents a parameter used in the Semantic Analogies task.  default: 2
	- _compare\_with_ the list of the runs to compare the results with.  default: \_all

The parameters can be i) passed by the command line, ii) organized in a XML file which is especially useful for larger runs, or iii) passed programmatically to the function which start the evaluation.

Note: The tasks can be executed sequentially or in parallel. If the code raises MemoryError it means that the tasks need more memory than the one available. In that case, run all the tasks sequentially.

## Supported vectors file format
The TXT_version the vectors file must be a TSV - tab or space separated values - with one line for each entity. 
Each line has to contain the entity name - without angular backets - and the embedded vector. 

The HDF5_version the vectors file must be an H5 with one group in it called "Vectors". 
In this group, there should be one dataset for each entity with the entity name, base32 encoded, as dataset name and the embedded vector as dataset content.

## Project structure
In the _code_ folder there is the whole framework code. 
- code
	- abstract_evaluation: abstract class to supervise the whole evaluation
	- abstract_dataManager: interface to manage a new input file format
	- abstract_taskManager: interface which defines all the methods a new task has to implement
	- abstract_model: interface to define a new model used by the task manager
	- manager: check the correctness of the parameters and start the evaluation
	- evaluationManager: implements the abstrac_evaluation interface
	- txt_dataManager and hdf5_dataManager: concrete dataManager which manage, respectively, the TXT and the HDF5 data format
	
	- one folder for each implemented task which contains
		- data folder with the dataset(s) by the task
		- taskManager which implements the abstract_taskManager interface
		- model which implements the abstract_model interface
            
 - example folder contains several main files showing the different ways to initialize the framework 
    
The evaluation manager instantiates the correct data manager - TXT or HDF5 one - according to the file format provided as input. Then, according to the user settings, the evaluation manager determines which tasks the user asked for and how they have to be run - in sequential or in parallel. For each task, it starts the suitable task manager. 

All the results (ignored data, results, and the output of the comparison) will be stored in results/result\_YYYY-MM-DD\_HH-MM-SS.

## How to customize the analogy function

Your **_analogy function_** has to take 3 vectors and it must return the predicted one.

## History
To get the Evaluation Framework history, you can follow this link: https://git.rwth-aachen.de/KGEmbedding/evaluationFramework.
