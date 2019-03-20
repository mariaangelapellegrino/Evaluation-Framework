from abc import abstractmethod

class AbstractEvaluationManager():
 
    def __init__(self, debugging_mode):
        super().__init__()
    
    @abstractmethod
    def initialize_vectors(self, vector_file, vec_size):
        pass

    @abstractmethod
    def run_tests_in_sequential(self, tasks, similarity_metric, analogy_function, top_k):
        pass

    @abstractmethod
    def run_tests_in_parallel(self, tasks, similarity_metric, analogy_function, top_k):
        pass 

    @abstractmethod
    def create_result_directory(self):
        pass

    @abstractmethod
    def compare_with(self, compare_with):
        pass