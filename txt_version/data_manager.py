import pandas as pd

class DataManager:

    @staticmethod
    def create_header(vec_size):
        headers = ['name']
        for i in range(0, vec_size):
            headers.append(i)
        return headers

    @staticmethod
    def read_vector_file(vectors_file, vec_size):
        local_vectors = pd.read_csv(vectors_file, "\s+",  names=DataManager.create_header(vec_size),  encoding='utf-8', index_col=False)
        return local_vectors