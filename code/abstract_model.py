from abc import abstractmethod

class AbstractModel():
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def train(self):
        pass