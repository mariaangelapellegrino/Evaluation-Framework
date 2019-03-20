import os
import time
import datetime
from multiprocessing import Process

from code.abstract_evaluationManager import AbstractEvaluationManager

from txt_dataManager import DataManager
from Classification.classification_taskManager import ClassificationManager as Classification_evaluator
from Regression.regression_taskManager import RegressionManager as Regression_evaluator
from Clustering.clustering_taskManager import ClusteringManager as Clustering_evaluator
from DocumentSimilarity.documentSimilarity_taskManager import DocumentSimilarityManager as Doc_Similarity_evaluator
from EntityRelatedness.entityRelatedness_taskManager import EntityRelatednessManager as Entity_Relatedness_evaluator
from SemanticAnalogies.semanticAnalogies_taskManager import SemanticAnalogiesManager as Semantic_Analogies_evaluator

class EvaluationManager(AbstractEvaluationManager):
    def __init__(self, debugging_mode):
        self.start_time = time.time()
        self.debugging_mode = debugging_mode
        self.data_manager = DataManager()
        if self.debugging_mode:
            print('Created TXT evaluation manager')

    def initialize_vectors(self, vector_filename, vector_size):
        self.vectors = self.data_manager.read_vector_file(vector_filename, vector_size)
        if self.debugging_mode: 
            print('Vectors read '+ str(len(self.vectors)))

        if len(self.vectors) == 0:
            raise Exception('Empty vectors file')

    def run_tests_in_sequential(self, tasks, similarity_metric, analogy_function, top_k):
        for task in tasks:
            if task=='Classification':
                try:
                    classification_evaluator = Classification_evaluator(self.debugging_mode)
                    log_errors = classification_evaluator.evaluate(self.vectors, self.result_directory)
                    self.log_file.write(str(log_errors))
                except Exception as e:
                    self.log_file.write("Classification: " + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Classification execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='Regression':
                try:
                    regression_evaluator = Regression_evaluator(self.debugging_mode)
                    log_errors = regression_evaluator.evaluate(self.vectors, self.result_directory)
                    self.log_file.write(str(log_errors))
                except Exception as e:
                    self.log_file.write("Regression:" + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Regression execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='Clustering':
                try:
                    clustering_evaluator = Clustering_evaluator(similarity_metric, self.debugging_mode)
                    log_errors = clustering_evaluator.evaluate(self.vectors, self.result_directory)
                    self.log_file.write(str(log_errors))
                except Exception as e:
                    self.log_file.write("Clustering: " + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Clustering execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='DocumentSimilarity':
                try:
                    doc_similarity_evaluator = Doc_Similarity_evaluator(similarity_metric, self.debugging_mode)
                    log_errors = doc_similarity_evaluator.evaluate(self.vectors, self.result_directory)
                    self.log_file.write(str(log_errors))
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Document Similarity execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='EntityRelatedness':
                try:
                    entity_Relatedness_evaluator = Entity_Relatedness_evaluator(similarity_metric, self.debugging_mode)
                    log_errors = entity_Relatedness_evaluator.evaluate(self.vectors, self.result_directory)    
                    self.log_file.write(str(log_errors)) 
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Entity Relatedness execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='SemanticAnalogies':
                try:
                    semantic_Analogies_evaluator = Semantic_Analogies_evaluator(similarity_metric, analogy_function, top_k, self.debugging_mode)
                    log_errors = semantic_Analogies_evaluator.evaluate(self.vectors, self.result_directory)
                    self.log_file.write(str(log_errors)) 
                except Exception as e:
                    self.log_file.write(e)
                else:
                    end_time = time.time()
                    self.log_file.write("Semantic Analogies execution time: " + str(end_time-self.start_time) + " seconds\n")
            else:
                #raise Exception('The task ' + task + ' is not supported')
                print('The task ' + task + ' is not supported')

    def run_tests_in_parallel(self, tasks, similarity_metric, analogy_function, top_k):
        processes = {}

        for task in tasks:
            if task=='Classification':
                classification_evaluator = Classification_evaluator(self.debugging_mode)
                p1 = Process(target=classification_evaluator.evaluate, args=(self.vectors, self.result_directory))
                p1.start()
                processes['Classification'] = p1
            elif task=='Regression':
                regression_evaluator = Regression_evaluator(self.debugging_mode)
                p2 = Process(target=regression_evaluator.evaluate, args=(self.vectors, self.result_directory))
                p2.start()
                processes['Regression'] = p2
            elif task=='Clustering':
                clustering_evaluator = Clustering_evaluator(similarity_metric, self.debugging_mode)
                p3 = Process(target=clustering_evaluator.evaluate, args=(self.vectors, self.result_directory))
                p3.start()
                processes['Clustering'] = p3
            elif task=='DocumentSimilarity':
                doc_similarity_evaluator = Doc_Similarity_evaluator(similarity_metric, self.debugging_mode)
                p4 = Process(target=doc_similarity_evaluator.evaluate, args=(self.vectors, self.result_directory))
                p4.start()
                processes['DocumentSimilarity'] = p4
            elif task=='EntityRelatedness':
                entity_relatedness_evaluator = Entity_Relatedness_evaluator(similarity_metric, self.debugging_mode)
                p5 = Process(target=entity_relatedness_evaluator.evaluate, args=(self.vectors, self.result_directory))
                p5.start()
                processes['EntityRelatedness'] = p5
            elif task=='SemanticAnalogies':
                semantic_analogies_evaluator = Semantic_Analogies_evaluator(analogy_function, similarity_metric, top_k, self.debugging_mode)
                p6 = Process(target=semantic_analogies_evaluator.evaluate, args=(self.vectors, self.result_directory))
                p6.start()
                processes['EntityRelatedness'] = p6
            else:
                print('The task ' + task + ' is not supported')

        for process_name, process in processes:
            process.join()
            if process.exception:
                self.log_file.write(process.exception)
            else:
                end_time = time.time()
                self.log_file.write(process_name + ": " + str(end_time-self.start_time))


    def create_result_directory(self):
        self.result_directory = '../results'
        if not os.path.exists(self.result_directory):
            try:  
                os.mkdir(self.result_directory)
            except OSError:  
                self.result_directory = "" 
            else:
                self.result_directory += "/" 
        else:
            self.result_directory += "/" 
            
        self.result_directory += "result"+datetime.datetime.fromtimestamp(time.time()).strftime('_%Y-%m-%d_%H-%M-%S')

        try:  
            os.mkdir(self.result_directory)
        except OSError:  
            raise Exception("Creation of the directory %s failed" % self.result_directory)
        else:
            self.log_file = open(os.path.join(self.result_directory, 'log.txt'), "w") 