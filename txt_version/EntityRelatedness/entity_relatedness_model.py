import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import kendalltau

class Model:

    def __init__(self):
        print('Entity relatedness model initialized')

    @staticmethod
    def compute_relatedness(left_merged, left_ignored, right_merged_list, right_ignored_list, distance_metric):
        predicted_rank_list = list()       

        for i in range(len(left_merged)):

            distances = []

            if right_merged_list[i].size>0:
                distances = distance.cdist(left_merged.iloc[[i], 1:], right_merged_list[i].iloc[:, 1:], metric= distance_metric).flatten() 
            
            ignored_distances = np.array([1 for j in range(len(right_ignored_list[i]))]) #max dist??            
            distances = np.concatenate((distances, ignored_distances))

            predicted_rank_list.append(list(np.argsort(distances)))

        for i in range(len(left_ignored['name'])):
            predicted_rank_list.append(list(np.argsort([1 for i in range(20)])))

        return predicted_rank_list

    @staticmethod
    def evaluate_ranking(entities_list, gold_ranking_list, predicted_ranking_list):
        score_list = list()

        for i in range(len(entities_list)):
            print('Entity Relatedness : ' +  entities_list[i])
            print(gold_ranking_list[i])
            print(predicted_ranking_list[i])
            kendalltau_correlation, kendalltau_pvalue = kendalltau(gold_ranking_list[i], predicted_ranking_list[i])
            print('Entity Relatedness : ' +  entities_list[i] + ' kendall tau correlation ' + str(kendalltau_correlation) + ' kendall tau pvalue ' + str(kendalltau_pvalue))
            score_list.append({'task_name' : 'Entity Relatedness', 'entity_name' : entities_list[i], 'kendalltau_correlation': kendalltau_correlation, 'kendalltau_pvalue': kendalltau_pvalue})

        return score_list