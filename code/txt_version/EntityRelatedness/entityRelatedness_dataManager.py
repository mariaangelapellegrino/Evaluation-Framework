import pandas as pd
import codecs

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
    def __init__(self):
        print("Entity relatedness data manager initialized")

    def read_vector_file(self, filename, vector_size):
        super.read_vector_file(filename, vector_size)

    def read_file(self, filename, columns):
        entities_groups = {}
        related_entities = []
        entities = set()
        f = codecs.open(filename, 'r', 'utf-8')

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

    def intersect_vectors_goldStandard(self, vectors, goldStandard_data,
        vector_filename = None, vector_size = None, goldStandard_filename = None, 
        column_id = None, column_score = None): 
        
        merged = pd.merge(goldStandard_data, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(goldStandard_data, vectors, on='name', how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']
        
        return merged, ignored