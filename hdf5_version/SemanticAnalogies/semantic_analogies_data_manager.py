import pandas as pd
import numpy as np
import base64
import h5py

class DataManager:
    def __init__(self):
        print('SemanticAnalogies data manager initialized')

    @staticmethod
    def get_vocab_and_W(vector_filename, vec_size):

        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        words = [key for key in vector_group.keys()]
        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}

        W = np.zeros((len(words), vec_size))
        
        for key in words:
            W[vocab[key], :] = vector_group[key][0]

        return (vocab, W)

    @staticmethod
    def read_data(vector_filename, gold_standard_file):
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        data = list()
        ignored = list()
        
        with open(gold_standard_file, 'r') as f:
            for line in f:
                quadruplet = line.rstrip().split()

                try:
                    key_0 = base64.b32encode(quadruplet[0])
                    key_1 = base64.b32encode(quadruplet[1])
                    key_2 = base64.b32encode(quadruplet[2])
                    key_3 = base64.b32encode(quadruplet[3])

                    vector_group[key_0][0]
                    vector_group[key_1][0]
                    vector_group[key_2][0]
                    vector_group[key_3][0]

                    data.append([key_0, key_1, key_2, key_3])
                except KeyError:
                    ignored.append(quadruplet)
        
        return (data, ignored)