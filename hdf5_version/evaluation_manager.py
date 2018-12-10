import numpy as np
import os
import time
import datetime
from multiprocessing import Process

from ClassificationAndRegression.classification_and_regression_evaluation import Evaluator as Classification_Regression_evaluator
from Clustering.clustering_evaluation import Evaluator as Clustering_evaluator
from DocumentSimilarity.document_similarity_evaluation import Evaluator as Doc_Similarity_evaluator
from EntityRelatedness.entity_relatedness_evaluation import Evaluator as Entity_Relatedness_evaluator
from SemanticAnalogies.semantic_analogies_evaluation import Evaluator as Semantic_Analogies_evaluator

vector_filename = ''
vector_size = 0
distance_metric = ''
result_directory = ''
top_k = 2

class EvaluationManager():
    def __init__(self):
        print('Start evaluation...')

    def evaluate(self, vec_filename, vec_size, distance_metric_string, analogy_func, k):
        global vector_filename 
        global vector_size
        global distance_metric
        global analogy_function
        global top_k 

        vector_filename = vec_filename
        vector_size = vec_size
        distance_metric = distance_metric_string
        analogy_function = analogy_func
        top_k = k

        self.create_result_directory()
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
        Semantic_Analogies_evaluator.evaluate(vector_filename, vector_size, analogy_function, top_k, result_directory)
        Classification_Regression_evaluator.evaluate(vector_filename, vector_size, result_directory)
        Clustering_evaluator.evaluate(vector_filename, vector_size, distance_metric, result_directory)
        Doc_Similarity_evaluator.evaluate(vector_filename, vector_size, distance_metric, result_directory)
        Entity_Relatedness_evaluator.evaluate(vector_filename, vector_size, distance_metric, result_directory)

    def run_tests_in_parallel(self):
        processes = []
        
        p1 = Process(target=Classification_Regression_evaluator.evaluate, args=(vector_filename, vector_size, result_directory))
        p1.start()
        processes.append(p1)

        p2 = Process(target=Clustering_evaluator.evaluate, args=(vector_filename, vector_size, distance_metric, result_directory))
        p2.start()
        processes.append(p2)

        p3 = Process(target=Doc_Similarity_evaluator.evaluate, args=(vector_filename, vector_size, distance_metric, result_directory))
        p3.start()
        processes.append(p3)

        p4 = Process(target=Entity_Relatedness_evaluator.evaluate, args=(vector_filename, vector_size, distance_metric, result_directory))
        p4.start()
        processes.append(p4)

        p5 = Process(target=Semantic_Analogies_evaluator.evaluate, args=(vector_filename, vector_size, analogy_function, top_k, result_directory))
        p5.start()
        processes.append(p5)

        for p in processes:
            p.join()






   
