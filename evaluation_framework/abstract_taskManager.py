from abc import abstractmethod

"""
It abstracts the behavior of a Task manager. It should be extended by each task manager.
"""
class AbstractTaskManager():
 
    def __init__(self):
        super().__init__()
    
    """
    It evaluates the specific task.
    
    vectors: dataframe which contains the vectors data
    vector_file: path of the vector file
    vector_size: size of the vectors
    result_directory: directory where the results must be stored
    log_dictionary: dictionary to store all the information to store in the log file
    scores_dictionary: dictionary to store all the scores which will be used in the comparison phase
    """
    @abstractmethod
    def evaluate(self, vectors, vector_file, vector_size, result_directory, log_dictionary, scores_dictionary):
        pass