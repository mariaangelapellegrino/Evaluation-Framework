import pandas as pd
from entityRelatedness_model import EntityRelatednessModel as Model
import csv
import os

from code.abstract_taskManager import AbstractTaskManager

class EntityRelatednessManager (AbstractTaskManager):
    def __init__(self, data_manager, distance_metric, debugging_mode):
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        self.distance_metric = distance_metric
        self.task_name = 'entity_relatedness'
        if self.debugging_mode:
            print("Entity relatedness task manager initialized")

    def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary= None): 
        log_errors = ""              
        gold_standard_filename = "KORE"
        script_dir = os.path.dirname(__file__)
        rel_path = "data/"+gold_standard_filename+".txt"
        gold_standard_file = os.path.join(script_dir, rel_path)
        
        groups = self.data_manager.read_file(gold_standard_file)

        scores = list()

        left_entities_df = pd.DataFrame({'name':groups.keys()})
        left_merged, left_ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, gold_standard_file, goldStandard_data=left_entities_df)

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

            self.storeResults(results_folder, gold_standard_filename, scores)
        
        if not log_dictionary is None:
            log_dictionary['Entity relatedness'] = log_errors
        return log_errors

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
                writer.writerow({'entity':getattr(ignored_tuple,'name'), 'related_to':related_to_value})
                
                if self.debugging_mode:
                    print('Entity relatedness : Ignored data: ' + getattr(ignored_tuple,'name'))
                
                
    def storeResults(self, results_folder, gold_standard_filename, scores):
        with open(results_folder+'/entityRelatedness_'+gold_standard_filename+'_results.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'entity_name', 'kendalltau_correlation', 'kendalltau_pvalue']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for score in scores:
                writer.writerow(score)
                if self.debugging_mode:
                    print('Entity Relatedness score: ' +   score)   