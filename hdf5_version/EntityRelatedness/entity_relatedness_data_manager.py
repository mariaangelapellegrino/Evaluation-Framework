import pandas as pd
import numpy as np
import base64
import h5py
import codecs

class DataManager:
    def __init__(self):
        print("Entity relatedness data manager initialized")

    @staticmethod
    def read_gold_standard_file(filename):
        entities_groups = {}
        related_entities = []
        entities = set()

        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                key = line.rstrip().lstrip()

                entities.add(key)
                if i%21 == 0:           
                    main_entitiy = key
                    related_entities = []

                else :
                    related_entities.append(key)    
                
                if i%21 == 20:
                    entities_groups[main_entitiy] = related_entities

        return (entities, entities_groups)

    @staticmethod
    def create_header(vec_size):
        headers = ['name']
        for i in range(0, vec_size):
            headers.append(i)
        return headers

    @staticmethod
    def merge_data(vector_filename, vector_size, entities_df):
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
        ignored = list()
        
        for row in entities_df.itertuples():
            try:
                encoded_name = base64.b32encode(row.name)
                values = vector_group[encoded_name][0]
                
                new_row = dict(zip(np.arange(vector_size), values))
                new_row['name'] = encoded_name

                merged = merged.append(new_row, ignore_index=True)
            except KeyError:
                ignored.append(row.name)
        
        return (merged, ignored)
