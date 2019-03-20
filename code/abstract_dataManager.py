from abc import abstractmethod

class AbstractDataManager():
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def read_vector_file(self, filename, vector_size):
        pass

    @abstractmethod
    def read_file(self, filename, columns):
        pass 

    @abstractmethod
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, goldStandard_data, goldStandard_filename, columns):
        pass 