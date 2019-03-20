from abc import abstractmethod

class AbstractTaskManager():
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def evaluate(self, vectors, result_directory):
        pass