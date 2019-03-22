from abc import abstractmethod

class AbstractTaskManager():
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def evaluate(self, vectors, vector_file, vector_size, result_directory, log_dictionary = None):
        pass