import pandas as pd
import numpy as np
import entity_relatedness_data_manager as data_manager
import entity_relatedness_model as model
import csv
import codecs

class Evaluator:
    def __init__(self):
        print("Entity relatedness evaluator init")

    @staticmethod
    def evaluate(vector_filename, vector_size, distance_metric, results_folder):
        gold_standard_filename = "EntityRelatedness/data/KORE.txt"
        entities, groups = data_manager.DataManager.read_gold_standard_file(gold_standard_filename)
        
        entities_df = pd.DataFrame(list(entities), columns = ['name'])

        scores = list()

        file_ignored = codecs.open(results_folder+'/entityRelatedness_KORE_ignoredData.csv',"w") 
        fieldnames = ['entity', 'related_to']
        writer = csv.DictWriter(file_ignored, fieldnames=fieldnames)
        writer.writeheader()
		
        left_entities_df = pd.DataFrame({'name':groups.keys()})
        left_merged, left_ignored = data_manager.DataManager.merge_data(vector_filename, vector_size, left_entities_df)

        print('Entity relatedness: Ignored data: ' + str(len(left_ignored)))
        for ignored_data in left_ignored:
            print('Entity relatedness: Ignored data: ' + ignored_data)
            writer.writerow({'entity':ignored_data, 'related_to':''})

        if left_merged.size == 0:
            print('EntityRelatedeness : no left entities of KORE in vectors')
        else:
            right_merged_list = list()
            right_ignored_list = list()
        
            for key in groups.keys():
                right_entities_df = pd.DataFrame({'name': groups[key]})
                right_merged, right_ignored = data_manager.DataManager.merge_data(vector_filename, vector_size, right_entities_df)
                right_merged_list.append(right_merged)
                right_ignored_list.append(right_ignored)
                
                print('Entity relatedness: Ignored ' + str(len(right_ignored)) + ' entities related to '+ key)
                for ignored_data in right_ignored:
                    print('Entity relatedness: Ignored ' + ignored_data+ ' entity related to '+ key)
                    writer.writerow({'entity':ignored_data, 'related_to':key})

            predicted_rank_list = model.Model.compute_relatedness(left_merged, left_ignored, right_merged_list, right_ignored_list, distance_metric)
            gold_rank_list = np.tile(np.arange(1, 21), (21, 1))
            scores = model.Model.evaluate_ranking(groups.keys(), gold_rank_list, predicted_rank_list)

        file_ignored.close()

        with open(results_folder+'/entityRelatedness_KORE_results.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'entity_name', 'kendalltau_correlation', 'kendalltau_pvalue']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for score in scores:
                writer.writerow(score)