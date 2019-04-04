# -*- coding: utf-8 -*-

import pandas as pd
from entityRelatedness_model import EntityRelatednessModel as Model
import unicodecsv as csv
#import csv
import os
from numpy import mean

from code.abstract_taskManager import AbstractTaskManager

task_name = 'EntityRelatedness'

class EntityRelatednessManager (AbstractTaskManager):
    def __init__(self, data_manager, distance_metric, debugging_mode):
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        self.distance_metric = distance_metric
        if self.debugging_mode:
            print("Entity relatedness task manager initialized")
            
    @staticmethod
    def get_task_name():
        return task_name

    def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary, scores_dictionary): 
        log_errors = ""              
        gold_standard_filename = "KORE"
        script_dir = os.path.dirname(__file__)
        rel_path = "data/"+gold_standard_filename+".txt"
        gold_standard_file = os.path.join(script_dir, rel_path)
        
        groups = self.data_manager.read_file(gold_standard_file)

        scores = list()

        left_entities_df = pd.DataFrame({'name':groups.keys()})
        left_merged, left_ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, gold_standard_file, left_entities_df)

        self.storeIgnored(results_folder, gold_standard_filename, left_ignored)

        if left_merged.size==0:
            log_errors += 'EntityRelatedeness : no left entities of '+gold_standard_filename+' in vectors \n'
            if self.debugging_mode:
                print('EntityRelatedeness : no left entities of '+gold_standard_filename+' in vectors')
        else:
            right_merged_list = list()
            right_ignored_list = list()
        
            for key in groups.keys():
                right_entities_df = pd.DataFrame({'name': groups[key]})
                right_merged, right_ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, gold_standard_file, goldStandard_data=right_entities_df)
                right_ignored['related_to'] = key
                right_merged_list.append(right_merged)
                right_ignored_list.append(right_ignored)
                
                self.storeIgnored(results_folder, gold_standard_filename, right_ignored)

            model = Model(self.distance_metric, self.debugging_mode)
            scores = model.train(left_merged, left_ignored, right_merged_list, right_ignored_list, groups)
            
            for score in scores:
                score['gold_standard_file'] = gold_standard_filename
                
            self.storeResults(results_folder, gold_standard_filename, scores)

            results_df = self.resultsAsDataFrame(scores)
            scores_dictionary[task_name] = results_df
        
        log_dictionary[task_name] = log_errors

    def storeIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Entity relatedness: Ignored data: ' + str(len(ignored)))
        
        with open(results_folder+'/entityRelatedness_'+gold_standard_filename+'_ignoredData.csv', "a+") as csv_file:
            fieldnames = ['entity', 'related_to']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for ignored_tuple in ignored.itertuples():
                if 'related_to' in ignored.columns:
                    related_to_value = getattr(ignored_tuple,'related_to')
                else:
                    related_to_value = ''
                    
                value = getattr(ignored_tuple,'name')
            
                '''
                if isinstance(value, str):
                    value = unicode(value, "utf-8").encode(encoding='UTF-8', errors='ignore')
                    
                if isinstance(related_to_value, str):
                    related_to_value = unicode(related_to_value, "utf-8").encode(encoding='UTF-8', errors='ignore')
                '''
                try:
                    writer.writerow({'entity':value, 'related_to':related_to_value})
                except UnicodeEncodeError:
                    print(value)
                    print(related_to_value)
                
                if self.debugging_mode:
                    print('Entity relatedness : Ignored data: ' + value.encode(encoding='UTF-8', errors='ignore'))
                
                
    def storeResults(self, results_folder, gold_standard_filename, scores):
        with open(results_folder+'/entityRelatedness_'+gold_standard_filename+'_results.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'gold_standard_file', 'entity_name', 'kendalltau_correlation', 'kendalltau_pvalue']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for score in scores:
                writer.writerow(score)
                if self.debugging_mode:
                    print('Entity Relatedness score: ' +   score)   
                    
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
        return ['KORE']
    
    @staticmethod
    def get_metric_list():
        return ['kendalltau_correlation']