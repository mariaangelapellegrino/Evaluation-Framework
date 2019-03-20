import pandas as pd
import numpy as np
import h5py

class DataManager:
    def __init__(self):
        print("Classification data manager initialized")

    @staticmethod
    def create_header(vec_size):
        headers = ['name', 'label']
        for i in range(0, vec_size):
            headers.append(i)
        return headers

    @staticmethod
    def read_data(vector_filename, vector_size, gold_file, column_id=None, column_score=None):
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]
        
        if column_id is None:
            column_id = 'DBpedia_URI15_Base32'
        if column_score is None:
            column_score = 'label'

        fields = ['DBpedia_URI15', column_id, column_score]
        
        gold = pd.read_csv(gold_file, "\t", usecols=fields, encoding='utf-8')

        gold.rename(columns={column_id: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'label'}, inplace=True)

        merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
        ignored = list()
        
        for row in gold.itertuples():
            try:
                values = vector_group[row.name][0]
                        
                new_row = dict(zip(np.arange(vector_size), values))
                new_row['name'] = row.name
                new_row['label'] = row.label

                merged = merged.append(new_row, ignore_index=True)
            except KeyError:
                ignored.append(row.DBpedia_URI15)

        return (merged, ignored)
