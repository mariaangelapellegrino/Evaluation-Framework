import pandas as pd
import json
import codecs
import numpy as np
import base64
import h5py

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
            print('Classification data manager initialized')
    
    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8') 

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None,
        column_key ='DBpedia_URI15_Base32', column_score = 'label'):
        
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]
        
        fields = ['DBpedia_URI15', column_key, column_score]
        
        gold = self.read_file(goldStandard_filename, fields)

        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'label'}, inplace=True)

        merged = pd.DataFrame(columns=self.create_header(vector_size))
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

        return merged, ignored  
    
    def create_header(self, vec_size):
        headers = ['name', 'label']
        for i in range(0, vec_size):
            headers.append(i)
        return headers                                        
    
class ClusteringDataManager(DataManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Clustering data manager initialized')
        
    def read_file(self, filename, columns):
        return pd.read_csv(filename, usecols=columns, delim_whitespace=True, index_col=False, header=None, names=columns, skipinitialspace=True, skip_blank_lines=True, encoding='utf-8') 

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None,
        column_key ='DBpedia_URI_Base32', column_score = 'cluster'): 
        
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        fields = ['DBpedia_URI', column_key, column_score]
        
        gold = self.read_file(goldStandard_filename, fields)
        
        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'cluster'}, inplace=True)

        merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
        ignored = pd.DataFrame(columns= ['name', 'cluster'])

        for row in gold.itertuples():
            try:
                values = vector_group[row.name][0]
                        
                new_row = dict(zip(np.arange(vector_size), values))
                new_row['name'] = row.name
                new_row['cluster'] = row.cluster

                merged = merged.append(new_row, ignore_index=True)
            except KeyError:
                ignored = ignored.append({'name':row.DBpedia_URI, 'cluster':row.cluster}, ignore_index=True)

        return merged, ignored    
    
    def create_header(self, vec_size):
        headers = ['name', 'cluster']
        for i in range(0, vec_size):
            headers.append(i)
        return headers
     
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
        
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
        ignored = list()
        
        entities = self.get_entities(goldStandard_filename)
        
        for row in entities.itertuples():
            try:
                values = vector_group[row.name][0]
                        
                new_row = dict(zip(np.arange(vector_size), values))
                new_row['doc'] = row.doc
                new_row['name'] = row.name
                new_row['weight'] = row.weight

                merged = merged.append(new_row, ignore_index=True)
            except KeyError:
                ignored.append(base64.b32decode(row.name))

        return merged, ignored

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
    
    def create_header(self, vec_size):
        headers = ['doc', 'name', 'weight']
        for i in range(0, vec_size):
            headers.append(i)
        return headers
    
class EntityRelatednessDataManager(DataManager):    
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Entity relatedness data manager initialized')

    def read_file(self, filename, columns = None):
        entities_groups = {}
        related_entities = []

        f = codecs.open(filename, 'r', 'utf-8')

        for i, line in enumerate(f):
            key = line.rstrip().lstrip()
 
            if i%21 == 0:           
                main_entitiy = key
                related_entities = []

            else :
                related_entities.append(key)    
                
            if i%21 == 20:
                entities_groups[main_entitiy] = related_entities

        return entities_groups

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None): 
        
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
        ignored = list()
        
        entities_df = pd.DataFrame()  #TODO
        
        for row in entities_df.itertuples():
            try:
                encoded_name = base64.b32encode(row.name)
                values = vector_group[encoded_name][0]
                
                new_row = dict(zip(np.arange(vector_size), values))
                new_row['name'] = encoded_name

                merged = merged.append(new_row, ignore_index=True)
            except KeyError:
                ignored.append(row.name)
        
        return merged, ignored
    
    def create_header(self, vec_size):
        headers = ['name']
        for i in range(0, vec_size):
            headers.append(i)
        return headers
    
class RegressionDataManager(DataManager):    
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Regression data manager initialized')
        
    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8') 

    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None,
        column_key ='DBpedia_URI15_Base32', column_score = 'rating'):
        
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]
        
        fields = ['DBpedia_URI15', column_key, column_score]
        
        gold = self.read_file(goldStandard_filename, fields)

        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'label'}, inplace=True)

        merged = pd.DataFrame(columns=self.create_header(vector_size))
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
        
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        data = list()
        ignored = list()
        
        with open(goldStandard_filename, 'r') as f:
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
        
        return data, ignored

    def create_vocab(self, vectors, vector_filename, vector_size):
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        words = [key for key in vector_group.keys()]
        vocab = {w: idx for idx, w in enumerate(words)}

        return vocab

    def normalize_vectors(self, vectors, vector_filename, vec_size, vocab):
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        words = [key for key in vector_group.keys()]
        vocab = {w: idx for idx, w in enumerate(words)}

        W = np.zeros((len(words), vec_size))
        
        for key in words:
            W[vocab[key], :] = vector_group[key][0]

        # normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm        