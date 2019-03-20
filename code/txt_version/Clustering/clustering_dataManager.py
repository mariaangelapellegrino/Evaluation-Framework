import pandas as pd

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode
        if self.debugging_mode:
            print("Clustering data manager initiated")

    def read_vector_file(self, filename, vector_size):
        super.read_vector_file(filename, vector_size)

    def read_file(self, filename, columns):
        pass     

    def intersect_vectors_goldStandard(self, vectors, goldStandard_filename, 
        vector_filename = None, vector_size = None, goldStandard_data = None,
        column_id = 'DBpedia_URI', column_score = 'cluster'): 
        
        gold = pd.read_csv(goldStandard_filename, usecols=[column_id,column_score], delim_whitespace=True, index_col=False, header=None, names=[column_id,column_score], skipinitialspace=True, skip_blank_lines=True, encoding='utf-8')

        gold.rename(columns={column_id: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'cluster'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return (merged, ignored)