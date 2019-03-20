import pandas as pd
from entityRelatedness_dataManager import DataManager
from entityRelatedness_model import EntityRelatednessModel as Model
import csv

from code.abstract_taskManager import AbstractTaskManager

class EntityRelatednessManager (AbstractTaskManager):
    def __init__(self, distance_metric, debugging_mode):
        self.data_manager = DataManager()
        self.distance_metric = distance_metric
        self.debugging_mode = debugging_mode
        if debugging_mode:
            print("Entity relatedness task Manager init")

    @staticmethod
    def evaluate(self, vectors, results_folder):        
        gold_standard_filename = "EntityRelatedness/data/KORE.txt"
        entities, groups = self.data_manager.read_gold_standard_file(gold_standard_filename)

        entities_df = pd.DataFrame(list(entities), columns = ['name'])

        scores = list()

        file_ignored = open(results_folder+'/entityRelatedness_KORE_ignoredData.csv',"w") 
        fieldnames = ['entity', 'related_to']
        self.writer = csv.DictWriter(file_ignored, fieldnames=fieldnames, encoding="utf-8")
        self.writer.writeheader()

        left_entities_df = pd.DataFrame({'name':groups.keys()})
        left_merged, left_ignored = self.data_manager.intersect_vectors_goldStandard(vectors= vectors, goldStandard_data = left_entities_df)
        left_ignored['related_to'] = [''] * left_ignored.shape[0]

        self.manageIgnored(results_folder, 'KORE', left_ignored)

        if left_merged.size == 0:
            print('EntityRelatedeness : no left entities of KORE in vectors')
        else:
            right_merged_list = list()
            right_ignored_list = list()
        
            for key in groups.keys():
                right_entities_df = pd.DataFrame({'name': groups[key]})
                right_merged, right_ignored = self.data_manager.merge_data(vectors, right_entities_df)
                right_ignored['related_to'] = key
                right_merged_list.append(right_merged)
                right_ignored_list.append(right_ignored)
                
                self.manageIgnored(results_folder, 'KORE', right_ignored)

            model = Model(self.distance_metric, self.debugging_mode)
            scores = model.train(left_merged, left_ignored, right_merged_list, right_ignored_list, groups)

        file_ignored.close()

        self.storeResults(results_folder, 'KORE', scores)

    def manageIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Entity relatedness: Ignored data: ' + str(len(ignored)))
        for ignored_tuple in ignored.itertuples():
            if self.debugging_mode:
                print('Entity relatedness: Ignored data: ' + getattr(ignored_tuple,'name'))
            self.writer.writerow({'entity':getattr(ignored_tuple,'name'), 'related_to':getattr(ignored_tuple,'related_to')})

    def storeResults(self, results_folder, gold_standard_filename, scores):
        with open(results_folder+'/entityRelatedness_KORE_results.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'entity_name', 'kendalltau_correlation', 'kendalltau_pvalue']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for score in scores:
                writer.writerow(score)