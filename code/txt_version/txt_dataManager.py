import pandas as pd

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
    def __init__(self):
        pass

    def read_vector_file(self, vector_filename, vec_size):
        local_vectors = pd.read_csv(vector_filename, "\s+",  names=self.create_header(vec_size),  encoding='utf-8', index_col=False)
        return local_vectors

    def read_file(self, filename, columns):
        pass     

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, goldStandard_data, goldStandard_filename, columns):
        pass 

    def create_header(self, vec_size):
        headers = ['name']
        for i in range(0, vec_size):
            headers.append(i)
        return headers