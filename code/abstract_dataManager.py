from abc import abstractmethod

"""
It abstracts the behavior of a Data manager. It should be extended by all the data manager.
"""
class AbstractDataManager():
 
    def __init__(self):
        super().__init__()
    
    """
    It reads the vectors file or it stores the information to read it.
    
    vector_filename: path of the file provided in input, which contains entities and the related vectors.
    vector_size: size of the vectors
    """
    @abstractmethod
    def inizialize_vectors(self, vector_file, vector_size):
        pass
    
    """
    It reads the vectors file.
    
    vector_filename: path of the file provided in input, which contains entities and the related vectors.
    vector_size: size of the vectors
    """
    @abstractmethod
    def read_vector_file(self, filename, vector_size):
        pass

    """
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    @abstractmethod
    def read_file(self, task, filename, columns):
        pass 

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name
    column_score: column of the dataset used as gold standard which contains the values used as gold standard
    """
    @abstractmethod
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, 
        goldStandard_data, goldStandard_filename, column_key, column_score):
        pass 