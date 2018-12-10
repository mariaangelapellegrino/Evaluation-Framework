import pandas as pd
import codecs

class DataManager:
    def __init__(self):
        print("Entity relatedness data manager initialized")

    @staticmethod
    def read_gold_standard_file(filename):
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

    @staticmethod
    def merge_data(vectors, entities_df):
        merged = pd.merge(entities_df, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(entities_df, vectors, on='name', how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']
        
        return (merged, ignored)
