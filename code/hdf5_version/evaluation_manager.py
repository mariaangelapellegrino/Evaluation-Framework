import numpy as np
import os
import time
import datetime
from multiprocessing import Process

from abstract_evaluationManager import AbstractEvaluationManager

from ClassificationAndRegression.classification_and_regression_evaluation import Evaluator as Classification_Regression_evaluator
from Clustering.clustering_evaluation import Evaluator as Clustering_evaluator
from DocumentSimilarity.document_similarity_evaluation import Evaluator as Doc_Similarity_evaluator
from EntityRelatedness.entity_relatedness_evaluation import Evaluator as Entity_Relatedness_evaluator
from SemanticAnalogies.semantic_analogies_evaluation import Evaluator as Semantic_Analogies_evaluator

class EvaluationManager(AbstractEvaluationManager):
    def __init__(self):
        print('Created HDF5 evaluation manager')

    def manage_vector_file(self, vector_filename, vector_size):
        global vector_filename 
        global vector_size

        self.vector_filename = vector_filename
        self.vector_size = vector_size

    def run_tests_in_sequential(self, tasks=tasks, similarity_metric=similarity_metric, analogy_function=analogy_function, top_k=top_k, debugging_mode=debugging_mode):
        for task in tasks:
            if task=='Classification':
                Classification_evaluator.evaluate(vector_filename, vector_size, result_directory, debugging_mode)
            elif task=='Regression':
                Regression_evaluator.evaluate(vector_filename, vector_size, result_directory, debugging_mode)
            elif task=='Clustering':
                Clustering_evaluator.evaluate(vector_filename, vector_size, similarity_metric, result_directory, debugging_mode)
            elif task=='DocumentSimilarity':
                Doc_Similarity_evaluator.evaluate(vector_filename, vector_size, similarity_metric, result_directory, debugging_mode)
            elif task=='EntityRelatedness':
                Entity_Relatedness_evaluator.evaluate(vector_filename, vector_size, similarity_metric, result_directory, debugging_mode)     
            elif task=='SemanticAnalogies':
                Semantic_Analogies_evaluator.evaluate(vector_filename, vector_size, analogy_function, similarity_metric, top_k, result_directory, debugging_mode)
            else:
                raise Exception('The task ' + task + ' is not supported')

    def run_tests_in_parallel(self, tasks=tasks, similarity_metric=similarity_metric, analogy_function=analogy_function, top_k=top_k, debugging_mode=debugging_mode):
        processes = []

        for task in tasks:
            if task=='Classification':
                p1 = Process(target=Classification_evaluator.evaluate, args=(vector_filename, vector_size, result_directory, debugging_mode))
                p1.start()
                processes.append(p1)
            elif task=='Regression':
                p2 = Process(target=Regression_evaluator.evaluate, args=(vector_filename, vector_size, result_directory, debugging_mode))
                p2.start()
                processes.append(p2)
            elif task=='Clustering':
                p3 = Process(target=Clustering_evaluator.evaluate, args=(vector_filename, vector_size, similarity_metric, result_directory, debugging_mode))
                p3.start()
                processes.append(p3)
            elif task=='DocumentSimilarity':
                p4 = Process(target=Doc_Similarity_evaluator.evaluate, args=(vector_filename, vector_size, similarity_metric, result_directory, debugging_mode))
                p4.start()
                processes.append(p4)
            elif task=='EntityRelatedness':
                p5 = Process(target=Entity_Relatedness_evaluator.evaluate, args=(vector_filename, vector_size, similarity_metric, result_directory, debugging_mode))
                p5.start()
                processes.append(p5)
            elif task=='SemanticAnalogies':
                p6 = Process(target=Semantic_Analogies_evaluator.evaluate, args=(vector_filename, vector_size, analogy_function, similarity_metric, top_k, result_directory, debugging_mode))
                p6.start()
                processes.append(p6)
            else:
                raise Exception('The task ' + task + ' is not supported')

        for p in processes:
            p.join()

    def create_result_directory(self):
        global result_directory
        result_directory = "result"+datetime.datetime.fromtimestamp(time.time()).strftime('_%Y-%m-%d_%H-%M-%S')

        try:  
            os.mkdir(result_directory)
        except OSError:  
            print ("Creation of the directory %s failed" % result_directory)