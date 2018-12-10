import numpy as np
import os
import time
import datetime
from multiprocessing import Process

from data_manager import DataManager
from ClassificationAndRegression.classification_and_regression_evaluation import Evaluator as Classification_Regression_evaluator
from Clustering.clustering_evaluation import Evaluator as Clustering_evaluator
from DocumentSimilarity.document_similarity_evaluation import Evaluator as Doc_Similarity_evaluator
from EntityRelatedness.entity_relatedness_evaluation import Evaluator as Entity_Relatedness_evaluator
from SemanticAnalogies.semantic_analogies_evaluation import Evaluator as Semantic_Analogies_evaluator

vectors_size = 0
distance_metric = ''
result_directory = ''
top_k = 2

class EvaluationManager():
    def __init__(self):
        print('Start evaluation...')

    def evaluate(self, vectors_filename, vec_size, distance_metric_string, analogy_func, k):
        global vectors_size
        global distance_metric
        global analogy_function
        global top_k 

        vectors_size = vec_size
        distance_metric = distance_metric_string
        analogy_function = analogy_func
        top_k = k

        self.create_result_directory()

        global vectors
        vectors = DataManager.read_vector_file(vectors_filename, vectors_size)
        print('Vectors read '+ str(len(vectors)))

        if len(vectors) == 0:
            raise Exception('Empty vectors file')

        self.run_tests_in_sequential()
        #self.run_tests_in_parallel()

    def create_result_directory(self):
        global result_directory
        result_directory = "result"+datetime.datetime.fromtimestamp(time.time()).strftime('_%Y-%m-%d_%H-%M-%S')

        try:  
            os.mkdir(result_directory)
        except OSError:  
            print ("Creation of the directory %s failed" % result_directory)

    def run_tests_in_sequential(self):
        Classification_Regression_evaluator.evaluate(vectors, result_directory)
        Clustering_evaluator.evaluate(vectors, distance_metric, result_directory)
        Doc_Similarity_evaluator.evaluate(vectors, distance_metric, result_directory)
        Entity_Relatedness_evaluator.evaluate(vectors, distance_metric, result_directory)
        Semantic_Analogies_evaluator.evaluate(vectors, vectors_size, analogy_function, top_k, result_directory)

    def run_tests_in_parallel(self):
        processes = []
        
        p1 = Process(target=Classification_Regression_evaluator.evaluate, args=(vectors, result_directory))
        p1.start()
        processes.append(p1)

        p2 = Process(target=Clustering_evaluator.evaluate, args=(vectors, distance_metric, result_directory))
        p2.start()
        processes.append(p2)

        p3 = Process(target=Doc_Similarity_evaluator.evaluate, args=(vectors, distance_metric, result_directory))
        p3.start()
        processes.append(p3)

        p4 = Process(target=Entity_Relatedness_evaluator.evaluate, args=(vectors, distance_metric, result_directory))
        p4.start()
        processes.append(p4)

        p5 = Process(target=Semantic_Analogies_evaluator.evaluate, args=(vectors, vectors_size, analogy_function, top_k, result_directory))
        p5.start()
        processes.append(p5)

        for p in processes:
            p.join()






   
