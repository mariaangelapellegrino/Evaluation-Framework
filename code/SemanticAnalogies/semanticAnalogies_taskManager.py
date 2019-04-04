import csv
from semanticAnalogies_model import SemanticAnalogiesModel as Model
import os
import pandas as pd
from code.abstract_taskManager import AbstractTaskManager

task_name = "SemanticAnalogies"

class SemanticAnalogiesManager (AbstractTaskManager):
    def __init__(self, data_manager, similarity_metric, top_k, debugging_mode, analogy_function= None):
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        self.similarity_metric = similarity_metric
        self.analogy_function = analogy_function
        self.top_k = top_k
        if debugging_mode:
            print('SemanticAnalogies task manager initialized')
        
    @staticmethod
    def get_task_name():
        return task_name

    def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary, scores_dictionary):
        log_errors = ""
        
        vocab = self.data_manager.create_vocab(vectors, vector_file, vector_size)
        W_norm = self.data_manager.normalize_vectors(vectors, vector_file, vector_size, vocab)

        gold_standard_filenames = self.get_gold_standard_file()

        scores = list()

        for gold_standard_filename in gold_standard_filenames:
            script_dir = os.path.dirname(__file__)
            rel_path = "data/"+gold_standard_filename+'.txt'
            gold_standard_file = os.path.join(script_dir, rel_path)
            
            data, ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, gold_standard_file)
            self.storeIgnored(results_folder, gold_standard_filename, ignored)

            if len(data) == 0:
                log_errors += 'SemanticAnalogies : Problems in merging vector with gold standard ' + gold_standard_filename + '\n'
                if self.debugging_mode:
                    print('SemanticAnalogies : Problems in merging vector with gold standard ' + gold_standard_filename)
            else:
                model = Model(self.similarity_metric, self.top_k, self.debugging_mode, self.analogy_function)

                result = model.train(vocab, data, W_norm)
                result['gold_standard_file'] = gold_standard_filename
                scores.append(result)

                self.storeResults(results_folder, gold_standard_file, scores)
                
                results_df = self.resultsAsDataFrame(scores)
                scores_dictionary[task_name] = results_df
                
        log_dictionary[task_name] = log_errors

    def storeIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Semantic analogies:'+ str(len(ignored))+' ignored quadruples')

        with open(results_folder+'/semanticAnalogies_'+gold_standard_filename+'_ignoredData.csv', 'w') as file_result:
            fieldnames = ['entity_1', 'entity_2', 'entity_3', 'entity_4']
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()

            for ignored_quadruplet in ignored:
                if self.debugging_mode:
                    print('Semantic analogies: Ignored quadruplet ' + str(ignored_quadruplet))
                
                entities = []
                
                for i in range(4):
                    entity = ignored_quadruplet[i]
                    if isinstance(entity, str):
                        entity = unicode(entity, "utf-8")
                    entities.append(entity.encode("utf-8"))
                    
                writer.writerow({'entity_1':entities[0], 
                                 'entity_2':entities[1], 
                                 'entity_3':entities[2], 
                                 'entity_4':entities[3]})
            
    def storeResults(self, results_folder, gold_standard_file, scores):
        with open(results_folder+'/semanticAnalogies_results.csv', 'wb') as file_result:
            fieldnames = ['task_name', 'gold_standard_file', 'top_k_value', 'right_answers', 'tot_answers', 'accuracy']
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()
            for score in scores:
                writer.writerow(score)
                if self.debugging_mode:
                    print(score)  
                    
    def resultsAsDataFrame(self, scores):
        data_dict = dict()
        data_dict['task_name'] = list()
        data_dict['gold_standard_file'] = list()
        data_dict['model'] = list()
        data_dict['model_configuration'] = list()
        data_dict['metric'] = list()
        data_dict['score_value'] = list()
                
        metrics = self.get_metric_list()
        
        for score in scores:
            for metric in metrics:
                data_dict['task_name'].append(score['task_name'])
                data_dict['gold_standard_file'].append(score['gold_standard_file'])
                data_dict['model'].append('')
                data_dict['model_configuration'].append('')
                data_dict['metric'].append(metric)
                data_dict['score_value'].append(score[metric])
        
        results_df = pd.DataFrame(data_dict, columns = ['task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value'])
        return results_df
    
    @staticmethod
    def get_gold_standard_file():
        return ['capital_country_entities', 'all_capital_country_entities','currency_entities', 'city_state_entities']
    
    @staticmethod
    def get_metric_list():
        return ['accuracy']