import pandas as pd
import json
import codecs
import numpy as np

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode

        self.taskDataManager = dict()
        self.taskDataManager['classification'] = ClassificationDataManager
        self.taskDataManager['clustering'] = ClusteringDataManager
        self.taskDataManager['regression'] = RegressionDataManager
        self.taskDataManager['document_similarity'] = DocumentSimilarityDataManager
        self.taskDataManager['entity_relatedness'] = EntityRelatednessDataManager
        self.taskDataManager['semantic_analogies'] = SemanticAnalogiesDataManager

    def initialize_vectors(self, vector_filename, vector_size):
        return self.read_vector_file(vector_filename, vector_size)

    def read_vector_file(self, vector_filename, vec_size):
        local_vectors = pd.read_csv(vector_filename, "\s+",  names=self.create_header(vec_size),  encoding='utf-8', index_col=False)
        return local_vectors

    def create_header(self, vec_size):
        headers = ['name']
        for i in range(0, vec_size):
            headers.append(i)
        return headers

    def read_file(self, filename, columns):
        pass
    
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, goldStandard_filename, goldStandard_data, column_key, column_score):
        pass
    '''
    def read_file(self, task, filename, columns):
        return self.taskDataManager[task].read_file(filename, columns)  
    
    def intersect_vectors_goldStandard(self, task, vectors, vector_filename, vector_size, goldStandard_filename, goldStandard_data, columns):
        return self.taskDataManager[task].intersect_vectors_goldStandard(vectors, vector_filename, vector_size, goldStandard_filename, goldStandard_data, columns)
    '''
    def get_data_manager(self, task):
        if task in self.taskDataManager:
            return self.taskDataManager[task]
        else:
            return self
        
    def add_task_dataManager(self, taskName, dataManager):
        self.dict[taskName] = dataManager
    
class ClassificationDataManager(DataManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('CLassification data manager initialized')
    
    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8')     

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None,
        column_key ='DBpedia_URI15', column_score = 'label'):
        
        gold = self.read_file(goldStandard_filename, [column_key, column_score])

        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'label'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored                                            
    
class ClusteringDataManager(DataManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Clustering data manager initialized')
        
    def read_file(self, filename, columns):
        return pd.read_csv(filename, usecols=columns, delim_whitespace=True, index_col=False, header=None, names=columns, skipinitialspace=True, skip_blank_lines=True, encoding='utf-8') 

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None,
        column_key ='DBpedia_URI15', column_score = 'cluster'): 
        
        gold = self.read_file(goldStandard_filename, [column_key, column_score])

        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'cluster'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored
    
class DocumentSimilarityDataManager(DataManager):  
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Document similarity data manager initialized')
          
    def read_file(self, filename, columns):
        return pd.read_csv(filename, ",",  usecols=columns, index_col=False, skipinitialspace=True, skip_blank_lines=True)

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, 
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None):

        entities = self.get_entities(goldStandard_filename)
        merged = pd.merge(entities, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(entities, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored['name']

    def get_entities(self, filename):
        with open(filename) as f:
            data = json.load(f)

        dict_entities = {}
        doc_list = list()
        entities_list = list()
        weight_list = list()
        i=0
    
        for doc_obj in data:
            i += 1
            for annotation in doc_obj["annotations"]:
                key = annotation['entity']

                doc_list.append(i)
                entities_list.append(key)
                weight_list.append(float(annotation['weight']))

        dict_entities['doc'] = doc_list
        dict_entities['name'] = entities_list
        dict_entities['weight'] = weight_list

        return pd.DataFrame.from_dict(dict_entities)
    
class EntityRelatednessDataManager(DataManager):    
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Entity relatedness data manager initialized')

    def read_file(self, filename, columns = None):
        entities_groups = {}
        related_entities = []
        #entities = set()
        f = codecs.open(filename, 'r', 'utf-8')

        for i, line in enumerate(f):
            key = line.rstrip().lstrip()
 
            #entities.add(key)

            if i%21 == 0:           
                main_entitiy = key
                related_entities = []

            else :
                related_entities.append(key)    
                
            if i%21 == 20:
                entities_groups[main_entitiy] = related_entities

        return entities_groups#entities, entities_groups

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None): 
        
        if goldStandard_data is None:
            goldStandard_data = self.read_file(goldStandard_filename)
        
        merged = pd.merge(goldStandard_data, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(goldStandard_data, vectors, on='name', how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge']=='left_only']
        
        return merged, ignored
    
class RegressionDataManager(DataManager):    
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Regression data manager initialized')
        
    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8') 

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = 'DBpedia_URI15', column_score = 'rating'): 
        
        gold = self.read_file(goldStandard_filename, [column_key, column_score])
        
        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'cluster'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored
    
class SemanticAnalogiesDataManager(DataManager):    
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Semantic analogies data manager initialized')

    def read_file(self, filename, columns):
        pass     

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None): 
        
        vocab = self.create_vocab(vectors)
        
        full_data = []
        file_input_stream = codecs.open(goldStandard_filename, 'r', 'utf-8')
        for line in file_input_stream:
            full_data.append(line.rstrip().split())

        data = [x for x in full_data if all(word in vocab for word in x)]
        
        if len(data)==0:
            ignored = [x for x in full_data]
        else:
            ignored = [x for x in full_data if not x in data]
            
        return data, ignored

    def create_vocab(self, vectors, vector_filename, vector_size):
        words = vectors['name']
        vocab = {w: idx for idx, w in enumerate(words)}

        return vocab

    def normalize_vectors(self, vectors, vector_filename, vec_size, vocab):
        W = np.zeros((len(vectors), vec_size))
        
        for index, row in vectors.iterrows():
            W[vocab[row['name']], :] = row[1:]

        # normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm