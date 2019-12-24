import pandas as pd
import json
import codecs
import numpy as np
from evaluation_framework.abstract_dataManager import AbstractDataManager

"""
It models how to manage vectors provided in TXT file.
"""
class DataManager(AbstractDataManager):
    """
    It initializes the DataManager for each provided task.
    
    debugging_mode: {TRUE,FALSE}.
    """
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode

        self.taskDataManager = dict()
        self.taskDataManager['classification'] = ClassificationDataManager
        self.taskDataManager['clustering'] = ClusteringDataManager
        self.taskDataManager['regression'] = RegressionDataManager
        self.taskDataManager['document_similarity'] = DocumentSimilarityDataManager
        self.taskDataManager['entity_relatedness'] = EntityRelatednessDataManager
        self.taskDataManager['semantic_analogies'] = SemanticAnalogiesDataManager
        
        if self.debugging_mode:
            print('TXT data manager initialized')

    """
    It reads the vectors file or it stores the information to read it.
    
    vector_filename: path of the file provided in input, which contains entities and the related vectors.
    vector_size: size of the vectors
    """
    def initialize_vectors(self, vector_filename, vector_size):
        return self.read_vector_file(vector_filename, vector_size)

    """
    It reads the vectors file.
    
    vector_filename: path of the file provided in input, which contains entities and the related vectors.
    vector_size: size of the vectors
    """
    def read_vector_file(self, vector_filename, vec_size):
        local_vectors = pd.read_csv(vector_filename, "\s+",  names=self.create_header(vec_size),  encoding='utf-8', index_col=False)
        return local_vectors

    """
    It returns a list which can be used as header, e.g. of a dataframe. 
    
    vec_size: size of the vectors
    """
    def create_header(self, vec_size):
        headers = ['name']
        for i in range(0, vec_size):
            headers.append(i)
        return headers

    """
    Warning: it does nothing.
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    def read_file(self, filename, columns):
        pass
    
    """
    Warning: it does nothing.
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name
    column_score: column of the dataset used as gold standard which contains the values used as gold standard
    """
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, 
        goldStandard_filename, goldStandard_data, column_key, column_score):
        pass
    '''
    def read_file(self, task, filename, columns):
        return self.taskDataManager[task].read_file(filename, columns)  
    
    def intersect_vectors_goldStandard(self, task, vectors, vector_filename, vector_size, goldStandard_filename, goldStandard_data, columns):
        return self.taskDataManager[task].intersect_vectors_goldStandard(vectors, vector_filename, vector_size, goldStandard_filename, goldStandard_data, columns)
    '''
   
    """
    It returns the task dataManager attached to the task provided in input, if any.
    """
    def get_data_manager(self, task):
        if task in self.taskDataManager:
            return self.taskDataManager[task]
        else:
            return self

    """
    It stores a new dataManager attached to the taskName. 
    If the taskName has already an attached dataManager, it will be updated. Otherwise, it will be added.
    """ 
    def add_task_dataManager(self, taskName, dataManager):
        self.dict[taskName] = dataManager
    
"""
DataManager attached to the Classification task
"""
class ClassificationDataManager(DataManager):
    """
    It initializes the DataManager for the classification data manager.
    """
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Classification data manager initialized')
    
    """
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8')     

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name. Default: 'DBpedia_URI15' according to the provided datasets.
    column_score: column of the dataset used as gold standard which contains the values used as gold standard. Default: 'label' according to the provide datasets.
    """
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
"""
DataManager attached to the Clustering task
"""
class ClusteringDataManager(DataManager):
    """
    It initializes the DataManager for the clustering data manager.
    """
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Clustering data manager initialized')
        
    """
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    def read_file(self, filename, columns):
        return pd.read_csv(filename, delim_whitespace=True, usecols=columns, index_col=False, skipinitialspace=True, skip_blank_lines=True, encoding='utf-8')

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name. Default: 'DBpedia_URI15' according to the provided datasets.
    column_score: column of the dataset used as gold standard which contains the values used as gold standard. Default: 'cluster' according to the provide datasets.
    """
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None,
        column_key ='DBpedia_URI', column_score = 'cluster'): 
        
        gold = self.read_file(goldStandard_filename, [column_key, column_score])

        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'cluster'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored
"""
DataManager attached to the DocumentSimilarity task
"""
class DocumentSimilarityDataManager(DataManager):  
    """
    It initializes the DataManager for the document similarity data manager.
    """
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Document similarity data manager initialized')
          
    """
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    def read_file(self, filename, columns):
        return pd.read_csv(filename, ",",  usecols=columns, index_col=False, skipinitialspace=True, skip_blank_lines=True)

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name.
    column_score: column of the dataset used as gold standard which contains the values used as gold standard.
    """
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size, 
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None):

        entities = self.get_entities(goldStandard_filename)
        merged = pd.merge(entities, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(entities, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored

    """
    It reads the file used as gold standard which contains entities attached to the documents.
    """
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
    
"""
DataManager attached to the EntityRelatedness task
"""
class EntityRelatednessDataManager(DataManager): 
    """
    It initializes the DataManager for the entity relatedness data manager.
    """   
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Entity relatedness data manager initialized')

    """
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
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

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name. 
    column_score: column of the dataset used as gold standard which contains the values used as gold standard.
    """
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None): 
        
        if goldStandard_data is None:
            entities = self.read_file(goldStandard_filename)
            goldStandard_data = pd.DataFrame({'name':entities.keys()})
        
        merged = pd.merge(goldStandard_data, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(goldStandard_data, vectors, on='name', how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge']=='left_only']
        
        return merged, ignored
    
"""
DataManager attached to the Regression task
"""
class RegressionDataManager(DataManager):    
    """
    It initializes the DataManager for the regression data manager.
    """
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Regression data manager initialized')
        
    """
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    def read_file(self, filename, columns):
        return pd.read_csv(filename,"\t", usecols=columns, encoding='utf-8') 

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name. Default: 'DBpedia_URI15' according to the provided datasets.
    column_score: column of the dataset used as gold standard which contains the values used as gold standard. Default: 'rating' according to the provide datasets.
    """
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = 'DBpedia_URI15', column_score = 'rating'): 
        
        gold = self.read_file(goldStandard_filename, [column_key, column_score])
        
        gold.rename(columns={column_key: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'label'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return merged, ignored
    
"""
DataManager attached to the SemanticAnalogies task
"""
class SemanticAnalogiesDataManager(DataManager):    
    """
    It initializes the DataManager for the semantic analogies data manager.
    """
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode 
        if self.debugging_mode:
            print('Semantic analogies data manager initialized')

    """
    Warning: it does nothing
    It reads the dataset used as gold standard
    
    filename: path of the dataset
    columns: list of columns to retrieve
    """
    def read_file(self, filename, columns):
        pass     

    """
    It intersects the input file which contains the vectors and the file used as gold standard.
    It returns the merged dataframe and the dataframe of ignored entities.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    goldStandard_filename: path of the dataset used as gold standard
    goldStandard_data: dataframe containing the dataset content
    column_key: column of the dataset used as gold standard which contains the entity name. 
    column_score: column of the dataset used as gold standard which contains the values used as gold standard.
    """
    def intersect_vectors_goldStandard(self, vectors, vector_filename, vector_size,
        goldStandard_filename, goldStandard_data = None, 
        column_key = None, column_score = None): 
        
        vocab = self.create_vocab(vectors, vector_filename, vector_size)
        
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

    """
    It returns a vocabulary containing all the entities of the vector file provided in input.
    It return a dictionary which key is the entity name and the value is a progressive value.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    """
    def create_vocab(self, vectors, vector_filename, vector_size):
        words = vectors['name']
        vocab = {w: idx for idx, w in enumerate(words)}

        return vocab
    
    """
    It normalizes the vector provided in input.
    
    vectors: dataframe containing the vectors
    vector_filename: path of the input file which contains the vectors provided in input
    vector_size: size of the vectors
    vocab: dictionary which key is the entity name and the value is a progressive value
    """
    def normalize_vectors(self, vectors, vector_filename, vec_size, vocab):
        W = np.zeros((len(vectors), vec_size))
        
        for index, row in vectors.iterrows():
            W[vocab[row['name']], :] = row[1:]

        # normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm