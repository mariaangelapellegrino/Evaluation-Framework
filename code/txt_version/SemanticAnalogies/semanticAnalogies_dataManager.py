import codecs
import numpy as np

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
    def __init__(self):
        print("Semantic Analogies data manager initialized")

    def read_vector_file(self, filename, vector_size):
        super.read_vector_file(filename, vector_size)

    def read_file(self, filename, columns):
        pass     

    def intersect_vectors_goldStandard(self, vectors, goldStandard_filename,
        vector_filename = None, vector_size = None, goldStandard_data = None, 
        column_id = 'DBpedia_URI', column_score = 'rating'): 
        
        vocab = self.create_vocab(vectors)

        full_data = []
        file_input_stream = codecs.open(goldStandard_filename, 'r', 'utf-8')
        for line in file_input_stream:
            full_data.append(line.rstrip().split())

        data = [x for x in full_data if all(word in vocab for word in x)]
        
        ignored = [x for x in full_data if not x in data]
        return data, ignored

    def create_vocab(self, vectors):
        words = vectors['name']
        vocab = {w: idx for idx, w in enumerate(words)}

        return vocab

    def normalize_vectors(self, vectors, vec_size, vocab):
        W = np.zeros((len(vectors), vec_size))
        
        for index, row in vectors.iterrows():
            W[vocab[row['name']], :] = row[1:]

        # normalize each word vector to unit length
        #W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm