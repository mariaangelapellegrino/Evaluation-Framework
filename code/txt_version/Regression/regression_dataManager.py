import pandas as pd

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode
        if debugging_mode:
            print("Regression data manager initialized")

    def read_vector_file(self, filename, vector_size):
        super.read_vector_file(filename, vector_size)

    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8') 

    def intersect_vectors_goldStandard(self, vectors, goldStandard_filename,
        vector_filename = None, vector_size = None, goldStandard_data = None, 
        column_id = 'DBpedia_URI15', column_score = 'rating'): 
        
        gold = self.read_file(goldStandard_filename, [column_id,column_score])
        
        gold.rename(columns={column_id: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'cluster'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return (merged, ignored)
