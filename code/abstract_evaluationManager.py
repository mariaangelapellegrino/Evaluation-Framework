from abc import abstractmethod

"""
It abstracts the behavior of a Evaluation manager. It should be extended by the evaluation manager.
"""
class AbstractEvaluationManager():
 
    def __init__(self, debugging_mode):
        super().__init__()
    
    """
    It stores the information related to vectors.
    
    vector_filename: path of the vector file
    vector_size: size of the vectors
    """
    @abstractmethod
    def initialize_vectors(self, vector_file, vec_size):
        pass
    
    """
    It runs the tasks in sequential
    
    tasks: list of the task to run
    similarity_metric: distance metric used as similarity metric
    top_k: parameters of the semantic analogies task
    analogy_function: function to compute the analogy among vectors
    """
    @abstractmethod
    def run_tests_in_sequential(self, tasks, similarity_metric, top_k, analogy_function=None):
        pass

    """
    It runs the tasks in parallel
    
    tasks: list of the task to run
    similarity_metric: distance metric used as similarity metric
    top_k: parameters of the semantic analogies task
    analogy_function: function to compute the analogy among vectors
    """
    @abstractmethod
    def run_tests_in_parallel(self, tasks, similarity_metric, top_k, analogy_function=None):
        pass 

    """
    It creates the result folder.
    """
    @abstractmethod
    def create_result_directory(self):
        pass

    """
    It manages the comparison with previous runs.
    
    compare_with: list of the runs to compare with. Default: _all
    scores_dictionary: dictionary of the scores of all the tasks
    """
    @abstractmethod
    def compare_with(self, compare_with):
        pass