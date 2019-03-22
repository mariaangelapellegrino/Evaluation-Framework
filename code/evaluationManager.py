import os
import time
import datetime
from multiprocessing import Process
import multiprocessing

from code.abstract_evaluationManager import AbstractEvaluationManager

from Classification.classification_taskManager import ClassificationManager as Classification_evaluator
from Regression.regression_taskManager import RegressionManager as Regression_evaluator
from Clustering.clustering_taskManager import ClusteringManager as Clustering_evaluator
from DocumentSimilarity.documentSimilarity_taskManager import DocumentSimilarityManager as Doc_Similarity_evaluator
from EntityRelatedness.entityRelatedness_taskManager import EntityRelatednessManager as Entity_Relatedness_evaluator
from SemanticAnalogies.semanticAnalogies_taskManager import SemanticAnalogiesManager as Semantic_Analogies_evaluator

class EvaluationManager(AbstractEvaluationManager):
    def __init__(self, data_manager, debugging_mode):
        self.start_time = time.time()
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        if self.debugging_mode:
            print('Created evaluation manager')

    def initialize_vectors(self, vector_filename, vector_size):
        self.vector_filename = vector_filename
        self.vector_size = vector_size
        self.vectors = self.data_manager.initialize_vectors(vector_filename, vector_size)

    def run_tests_in_sequential(self, tasks, similarity_metric, analogy_function, top_k):
        for task in tasks:
            if task=='Classification':
                try:
                    classification_dataManager = self.data_manager.get_data_manager('classification')(self.debugging_mode)
                    classification_evaluator = Classification_evaluator(classification_dataManager, self.debugging_mode)
                    log_errors = classification_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory)
                    self.log_file.write(str(log_errors))
                    print('Classification finished')
                except Exception as e:
                    self.log_file.write("Classification: " + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Classification execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='Regression':
                try:
                    regression_dataManager = self.data_manager.get_data_manager('regression')(self.debugging_mode)
                    regression_evaluator = Regression_evaluator(regression_dataManager, self.debugging_mode)
                    log_errors = regression_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory)
                    self.log_file.write(str(log_errors))
                    print('Regression finished')
                except Exception as e:
                    self.log_file.write("Regression:" + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Regression execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='Clustering':
                try:
                    clustering_dataManager = self.data_manager.get_data_manager('clustering')(self.debugging_mode)
                    clustering_evaluator = Clustering_evaluator(clustering_dataManager, similarity_metric, self.debugging_mode)
                    log_errors = clustering_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory)
                    self.log_file.write(str(log_errors))
                    print('Clustering finished')
                except Exception as e:
                    self.log_file.write("Clustering: " + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Clustering execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='DocumentSimilarity':
                try:
                    documentSimilarity_dataManager = self.data_manager.get_data_manager('document_similarity')(self.debugging_mode)
                    doc_similarity_evaluator = Doc_Similarity_evaluator(documentSimilarity_dataManager, similarity_metric, self.debugging_mode)
                    log_errors = doc_similarity_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory)
                    self.log_file.write(str(log_errors))
                    print('Document similarity finished')
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Document Similarity execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='EntityRelatedness':
                try:
                    entityRelatedness_dataManager = self.data_manager.get_data_manager('entity_relatedness')(self.debugging_mode)
                    entity_Relatedness_evaluator = Entity_Relatedness_evaluator(entityRelatedness_dataManager, similarity_metric, self.debugging_mode)
                    log_errors = entity_Relatedness_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory)    
                    self.log_file.write(str(log_errors)) 
                    print('Entity Relatedness finished')
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Entity Relatedness execution time: " + str(end_time-self.start_time) + " seconds\n")
            elif task=='SemanticAnalogies':
                try:
                    semanticAnalogies_dataManager = self.data_manager.get_data_manager('semantic_analogies')(self.debugging_mode)
                    semantic_Analogies_evaluator = Semantic_Analogies_evaluator(semanticAnalogies_dataManager, similarity_metric, analogy_function, top_k, self.debugging_mode)
                    log_errors = semantic_Analogies_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory)
                    self.log_file.write(str(log_errors)) 
                    print('Semantic Analogies finished')
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Semantic Analogies execution time: " + str(end_time-self.start_time) + " seconds\n")
            else:
                #raise Exception('The task ' + task + ' is not supported')
                print('The task ' + task + ' is not supported')

    def run_tests_in_parallel(self, tasks, similarity_metric, analogy_function, top_k):
        processing_manager = multiprocessing.Manager()
        log_dictionary = processing_manager.dict()
        processes = {}
        for task in tasks:
            if task=='Classification':
                classification_dataManager = self.data_manager.get_data_manager('classification')(self.debugging_mode)
                classification_evaluator = Classification_evaluator(classification_dataManager, self.debugging_mode)
                p1 = Process(target=classification_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary))
                p1.start()
                processes['Classification'] = p1
            elif task=='Regression':
                regression_dataManager = self.data_manager.get_data_manager('regression')(self.debugging_mode)
                regression_evaluator = Regression_evaluator(regression_dataManager, self.debugging_mode)
                p2 = Process(target=regression_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary))
                p2.start()
                processes['Regression'] = p2
            elif task=='Clustering':
                clustering_dataManager = self.data_manager.get_data_manager('clustering')(self.debugging_mode)
                clustering_evaluator = Clustering_evaluator(clustering_dataManager, similarity_metric, self.debugging_mode)
                p3 = Process(target=clustering_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary))
                p3.start()
                processes['Clustering'] = p3
            elif task=='DocumentSimilarity':
                documentSimilarity_dataManager = self.data_manager.get_data_manager('document_similarity')(self.debugging_mode)
                doc_similarity_evaluator = Doc_Similarity_evaluator(documentSimilarity_dataManager, similarity_metric, self.debugging_mode)
                p4 = Process(target=doc_similarity_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary))
                p4.start()
                processes['DocumentSimilarity'] = p4
            elif task=='EntityRelatedness':
                entityRelatedness_dataManager = self.data_manager.get_data_manager('entity_relatedness')(self.debugging_mode)
                entity_relatedness_evaluator = Entity_Relatedness_evaluator(entityRelatedness_dataManager, similarity_metric, self.debugging_mode)
                p5 = Process(target=entity_relatedness_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary))
                p5.start()
                processes['EntityRelatedness'] = p5
            elif task=='SemanticAnalogies':
                semanticAnalogies_dataManager = self.data_manager.get_data_manager('semantic_analogies')(self.debugging_mode)
                semantic_analogies_evaluator = Semantic_Analogies_evaluator(semanticAnalogies_dataManager, analogy_function, similarity_metric, top_k, self.debugging_mode)
                p6 = Process(target=semantic_analogies_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary))
                p6.start()
                processes['SemanticAnalogies'] = p6
            else:
                print('The task ' + task + ' is not supported')

        for process_name in processes:
            process = processes[process_name]
            process.join() 
                        
            if process_name in log_dictionary:
                self.log_file.write(log_dictionary[process_name])
                
            if hasattr(process, 'exception'):
                self.log_file.write(str(process.exception))
            else:
                print(process_name + ' is finished')
                end_time = time.time()
                self.log_file.write(process_name + " execution time: " + str(end_time-self.start_time) + ' seconds \n')


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