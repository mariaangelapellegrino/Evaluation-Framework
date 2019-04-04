from abc import abstractmethod

"""
It abstracts the behavior of a Task model. It should be extended by each task model.
"""
class AbstractModel():
 
    def __init__(self):
        super().__init__()
    
    """
    It trains the models and returns the tesk results.
    """
    @abstractmethod
    def train(self):
        pass