import os
import time
import datetime
from multiprocessing import Process
import multiprocessing
import pandas as pd

from evaluation_framework.abstract_evaluationManager import AbstractEvaluationManager

from evaluation_framework.Classification.classification_taskManager import ClassificationManager as Classification_evaluator
from evaluation_framework.Regression.regression_taskManager import RegressionManager as Regression_evaluator
from evaluation_framework.Clustering.clustering_taskManager import ClusteringManager as Clustering_evaluator
from evaluation_framework.DocumentSimilarity.documentSimilarity_taskManager import DocumentSimilarityManager as Doc_Similarity_evaluator
from evaluation_framework.EntityRelatedness.entityRelatedness_taskManager import EntityRelatednessManager as Entity_Relatedness_evaluator
from evaluation_framework.SemanticAnalogies.semanticAnalogies_taskManager import SemanticAnalogiesManager as Semantic_Analogies_evaluator

"""
It coordinates the execution of the tasks. 
"""
class EvaluationManager(AbstractEvaluationManager):
    """
    It initialize the evaluation manager.
    
    data_manager: data manager related to the specific file format
    debugging_mode: {True, False}
    """
    def __init__(self, data_manager, debugging_mode):
        self.start_time = time.time()
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        if self.debugging_mode:
            print('Created evaluation manager')

    """
    It stores the information related to vectors.
    
    vector_filename: path of the vector file
    vector_size: size of the vectors
    """
    def initialize_vectors(self, vector_filename, vector_size):
        self.vector_filename = vector_filename
        self.vector_size = vector_size
        self.vectors = self.data_manager.initialize_vectors(vector_filename, vector_size)
        
        self.log_file.write("TESTED CONFIGURATION\n")
        self.log_file.write("Vector filename: " + vector_filename+"\n")
        self.log_file.write("Vector size:" + str(vector_size)+"\n")

    """
    It runs the tasks in sequential
    
    tasks: list of the task to run
    similarity_metric: distance metric used as similarity metric
    top_k: parameters of the semantic analogies task
    analogy_function: function to compute the analogy among vectors
    """
    def run_tests_in_sequential(self, tasks, similarity_metric, top_k, analogy_function = None):
        self.log_file.write("Distance metric:" + similarity_metric+"\n\n")
        
        self.similarity_metric = similarity_metric
        self.top_k = top_k
        self.tasks = tasks
                
        log_dictionary = dict()
        scores_dictionary = dict()
        for task in tasks:
            if task==Classification_evaluator.get_task_name():
                try:
                    classification_dataManager = self.data_manager.get_data_manager('classification')(self.debugging_mode)
                    classification_evaluator = Classification_evaluator(classification_dataManager, self.debugging_mode)
                    classification_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary)
                    self.log_file.write(log_dictionary[task])
                    print('Classification finished')
                except Exception as e:
                    self.log_file.write("Classification: " + str(e))
                else:
                    end_time = time.time()
                    seconds = end_time-self.start_time
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    execution_time = str(h) + ":" + str(round(m, 2)) + ":" + str(round(s, 2))
                    self.log_file.write("Classification execution time: " + execution_time + " seconds\n")
                    print "%d:%02d:%02d" % (h, m, s)

            elif task==Regression_evaluator.get_task_name():
                try:
                    regression_dataManager = self.data_manager.get_data_manager('regression')(self.debugging_mode)
                    regression_evaluator = Regression_evaluator(regression_dataManager, self.debugging_mode)
                    regression_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary)
                    self.log_file.write(log_dictionary[task])
                    print('Regression finished')
                except Exception as e:
                    self.log_file.write("Regression:" + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Regression execution time: " + str(round(end_time-self.start_time, 2)) + " seconds\n")
            elif task==Clustering_evaluator.get_task_name():
                try:
                    clustering_dataManager = self.data_manager.get_data_manager('clustering')(self.debugging_mode)
                    clustering_evaluator = Clustering_evaluator(clustering_dataManager, similarity_metric, self.debugging_mode)
                    clustering_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary)
                    self.log_file.write(log_dictionary[task])
                    print('Clustering finished')
                except Exception as e:
                    self.log_file.write("Clustering: " + str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Clustering execution time: " + str(round(end_time-self.start_time, 2)) + " seconds\n")
            elif task==Doc_Similarity_evaluator.get_task_name():
                try:
                    documentSimilarity_dataManager = self.data_manager.get_data_manager('document_similarity')(self.debugging_mode)
                    doc_similarity_evaluator = Doc_Similarity_evaluator(documentSimilarity_dataManager, similarity_metric, self.debugging_mode)
                    doc_similarity_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary)
                    self.log_file.write(log_dictionary[task])
                    print('Document similarity finished')
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Document Similarity execution time: " + str(round(end_time-self.start_time, 2)) + " seconds\n")
            elif task==Entity_Relatedness_evaluator.get_task_name():
                try:
                    entityRelatedness_dataManager = self.data_manager.get_data_manager('entity_relatedness')(self.debugging_mode)
                    entity_Relatedness_evaluator = Entity_Relatedness_evaluator(entityRelatedness_dataManager, similarity_metric, self.debugging_mode)
                    entity_Relatedness_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary)    
                    self.log_file.write(log_dictionary[task]) 
                    print('Entity Relatedness finished')
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Entity Relatedness execution time: " + str(round(end_time-self.start_time, 2)) + " seconds\n")
            elif task==Semantic_Analogies_evaluator.get_task_name():
                try:
                    semanticAnalogies_dataManager = self.data_manager.get_data_manager('semantic_analogies')(self.debugging_mode)
                    semantic_Analogies_evaluator = Semantic_Analogies_evaluator(semanticAnalogies_dataManager, top_k, self.debugging_mode, analogy_function)
                    semantic_Analogies_evaluator.evaluate(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary)
                    self.log_file.write(log_dictionary[task]) 
                    print('Semantic Analogies finished')
                except Exception as e:
                    self.log_file.write(str(e))
                else:
                    end_time = time.time()
                    self.log_file.write("Semantic Analogies execution time: " + str(round(end_time-self.start_time, 2)) + " seconds\n")
            else:
                #raise Exception('The task ' + task + ' is not supported')
                print('The task ' + task + ' is not supported')

        return scores_dictionary

    """
    It runs the tasks in parallel
    
    tasks: list of the task to run
    similarity_metric: distance metric used as similarity metric
    top_k: parameters of the semantic analogies task
    analogy_function: function to compute the analogy among vectors
    """
    def run_tests_in_parallel(self, tasks, similarity_metric, top_k, analogy_function = None):
        self.similarity_metric = similarity_metric
        self.top_k = top_k
        self.tasks = tasks
        
        processing_manager = multiprocessing.Manager()
        log_dictionary = processing_manager.dict()
        scores_dictionary = processing_manager.dict()
        processes = {}
        for task in tasks:
            if task==Classification_evaluator.get_task_name():
                classification_dataManager = self.data_manager.get_data_manager('classification')(self.debugging_mode)
                classification_evaluator = Classification_evaluator(classification_dataManager, self.debugging_mode)
                p1 = Process(target=classification_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary))
                p1.start()
                processes[Classification_evaluator.get_task_name()] = p1
            elif task==Regression_evaluator.get_task_name():
                regression_dataManager = self.data_manager.get_data_manager('regression')(self.debugging_mode)
                regression_evaluator = Regression_evaluator(regression_dataManager, self.debugging_mode)
                p2 = Process(target=regression_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary))
                p2.start()
                processes[Regression_evaluator.get_task_name()] = p2
            elif task==Clustering_evaluator.get_task_name():
                clustering_dataManager = self.data_manager.get_data_manager('clustering')(self.debugging_mode)
                clustering_evaluator = Clustering_evaluator(clustering_dataManager, similarity_metric, self.debugging_mode)
                p3 = Process(target=clustering_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary))
                p3.start()
                processes[Clustering_evaluator.get_task_name()] = p3
            elif task==Doc_Similarity_evaluator.get_task_name():
                documentSimilarity_dataManager = self.data_manager.get_data_manager('document_similarity')(self.debugging_mode)
                doc_similarity_evaluator = Doc_Similarity_evaluator(documentSimilarity_dataManager, similarity_metric, self.debugging_mode)
                p4 = Process(target=doc_similarity_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary))
                p4.start()
                processes[Doc_Similarity_evaluator.get_task_name()] = p4
            elif task==Entity_Relatedness_evaluator.get_task_name():
                entityRelatedness_dataManager = self.data_manager.get_data_manager('entity_relatedness')(self.debugging_mode)
                entity_relatedness_evaluator = Entity_Relatedness_evaluator(entityRelatedness_dataManager, similarity_metric, self.debugging_mode)
                p5 = Process(target=entity_relatedness_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary))
                p5.start()
                processes[Entity_Relatedness_evaluator.get_task_name()] = p5
            elif task==Semantic_Analogies_evaluator.get_task_name():
                semanticAnalogies_dataManager = self.data_manager.get_data_manager('semantic_analogies')(self.debugging_mode)
                semantic_analogies_evaluator = Semantic_Analogies_evaluator(semanticAnalogies_dataManager, top_k, self.debugging_mode, analogy_function)
                p6 = Process(target=semantic_analogies_evaluator.evaluate, args=(self.vectors, self.vector_filename, self.vector_size, self.result_directory, log_dictionary, scores_dictionary))
                p6.start()
                processes[Semantic_Analogies_evaluator.get_task_name()] = p6
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
                self.log_file.write(process_name + " execution time: " + str(round(end_time-self.start_time, 2)) + ' seconds \n')

        return scores_dictionary
    
    """
    It creates the result folder.
    """
    def create_result_directory(self):
        script_dir = os.path.dirname(__file__)

        curr_dir = os.getcwd()
        rel_path = curr_dir+'/results'
        self.result_directory = os.path.join(script_dir, rel_path)
        
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
            
    """
    It manages the comparison with previous runs.
    
    compare_with: list of the runs to compare with. Default: _all
    scores_dictionary: dictionary of the scores of all the tasks
    """
    def compare_with(self, compare_with, scores_dictionary):   
        #read data for the comparison             
        script_dir = os.path.dirname(__file__)

        curr_dir = os.getcwd()
        rel_path = curr_dir+'/comparison.csv'
        
        self.comparison_filename = os.path.join(script_dir, rel_path)
        exists = os.path.isfile(self.comparison_filename)
        if exists:
            results_df = pd.read_csv(self.comparison_filename, "\s+",  names=['test_name', 'task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value'],  encoding='utf-8', header=0)
        else:
            results_df = pd.DataFrame(columns=['test_name', 'task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value'])
            
        #define the test name
        partial_test_name = os.path.splitext(self.vector_filename)[0]+'_'+str(self.vector_size)+'_'+self.similarity_metric+'_'+str(self.top_k)
        test_name = partial_test_name + '_1'     
            
        test_names = set(results_df['test_name'])
        
        if test_name in test_names:
            values = [value for value in test_names if value.startswith(partial_test_name)]
            values.sort(reverse=True)
            splitted_values = values[0].split("_")
            last_progressive = int(splitted_values[len(splitted_values)-1])
            test_name = partial_test_name + '_' + str(last_progressive+1)
        
        #the score dictionary is converted into score dataframe and the stored results are updated
        scores_dataframe = pd.DataFrame(columns=['test_name', 'task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value'])
        for (task, current_score_dataframe) in scores_dictionary.items():
            scores_dataframe = pd.concat([scores_dataframe, current_score_dataframe])
        scores_dataframe['test_name'] = test_name
        
        new_results_df = pd.concat([results_df, scores_dataframe])
        
        new_results_df.to_csv(self.comparison_filename, sep=' ', columns= ['test_name', 'task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value' ], index = False)  

        #start the comparison
        effective_comparison_df = pd.DataFrame(columns = ['test_name', 'task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value'])
        rating_dataframe = pd.DataFrame(columns=['task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'ranking', 'absolute_total', 'relative_total'])

        if compare_with == '_all':
            compare_with = test_names
            
        comparison_df = scores_dataframe
        for comparing_test_name in compare_with:
            comparison_df = pd.concat([new_results_df[new_results_df['test_name']==comparing_test_name], comparison_df])
                   
        if not comparison_df.empty:
            for task in self.tasks:
                task_dataframe = comparison_df[comparison_df['task_name']==task]
                effective_comparison_df = pd.concat([effective_comparison_df, task_dataframe])
                
                to_filter_0 = scores_dataframe[scores_dataframe['task_name']==task]
                gold_standard_list = list(to_filter_0['gold_standard_file'].drop_duplicates())                
    
                for gold_standard_file in gold_standard_list:
                    
                    to_filter_1 = to_filter_0[to_filter_0['gold_standard_file']==gold_standard_file]
                    models = list(to_filter_1['model'].drop_duplicates())
                    
                    for model in models:
                        to_filter_2 = to_filter_1[to_filter_1['model']==model]
                        model_configurations = list(to_filter_2['model_configuration'].drop_duplicates())
                    
                        for model_configuration in model_configurations:
                            to_filter_3 = to_filter_2[to_filter_2['model_configuration']==model_configuration]
                            metrics = list(to_filter_3['metric'].drop_duplicates())
                    
                            for metric in metrics:
                                to_filter_4 = to_filter_3[to_filter_3['metric']==metric]
                                value_to_find = list(to_filter_4['score_value'])[0]

                                to_sort = task_dataframe[(task_dataframe['gold_standard_file']==gold_standard_file) &
                                                         (task_dataframe['model']==model) &
                                                         (task_dataframe['model_configuration']==model_configuration) &
                                                         (task_dataframe['metric']==metric)]
                                to_sort = list(to_sort['score_value'])
                                
                                sorted_metric_results = sorted(to_sort, reverse = True)
                                if task==Regression_evaluator.get_task_name():
                                    sorted_metric_results = sorted(to_sort)  
                                    
                                ranking = sorted_metric_results.index(value_to_find)

                                rating_dataframe = rating_dataframe.append({'task_name':task,
                                                         'gold_standard_file':gold_standard_file,
                                                         'model':model,
                                                         'model_configuration':model_configuration,
                                                         'metric':metric,
                                                         'ranking':ranking, 
                                                         'absolute_total':len(sorted_metric_results), 
                                                         'relative_total':len(set(sorted_metric_results))}, ignore_index=True)

            effective_comparison_df.to_csv(self.result_directory+'/comparison_values.csv', sep=' ', columns= ['test_name', 'task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value' ], index=False)        
            rating_dataframe.to_csv(self.result_directory+'/comparison_ranking.csv', sep=' ', columns= ['task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'ranking', 'absolute_total', 'relative_total' ], index=False)
        