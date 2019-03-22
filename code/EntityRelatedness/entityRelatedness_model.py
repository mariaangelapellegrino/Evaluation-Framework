import numpy as np
from scipy.spatial import distance
from scipy.stats import kendalltau

from code.abstract_model import AbstractModel

class EntityRelatednessModel(AbstractModel):

    def __init__(self, distance_metric, debugging_mode = False):
        self.debugging_mode = debugging_mode
        self.distance_metric = distance_metric
        if self.debugging_mode:
            print('Entity relatedness model initialized')

    def train(self, left_merged, left_ignored, right_merged_list, right_ignored_list, groups):
        predicted_rank_list = self.compute_relatedness(left_merged, left_ignored, right_merged_list, right_ignored_list)
        gold_rank_list = np.tile(np.arange(1, 21), (21, 1))
        return self.evaluate_ranking(groups.keys(), gold_rank_list, predicted_rank_list)

    def compute_relatedness(self, left_merged, left_ignored, right_merged_list, right_ignored_list):
        predicted_rank_list = list()       

        for i in range(len(left_merged)):

            distances = []

            if right_merged_list[i].size>0:
                distances = distance.cdist(left_merged.iloc[[i], 1:], right_merged_list[i].iloc[:, 1:], metric= self.distance_metric).flatten() 
            
            ignored_distances = np.array([1 for j in range(len(right_ignored_list[i]))]) #max dist??            
            distances = np.concatenate((distances, ignored_distances))

            predicted_rank_list.append(list(np.argsort(distances)))

        for i in range(len(left_ignored['name'])):
            predicted_rank_list.append(list(np.argsort([1 for i in range(20)])))

        return predicted_rank_list

    def evaluate_ranking(self, entities_list, gold_ranking_list, predicted_ranking_list):
        score_list = list()

        for i in range(len(entities_list)):
            if self.debugging_mode:
                print('Entity Relatedness : ' +  entities_list[i])
                print(gold_ranking_list[i])
                print(predicted_ranking_list[i])
            kendalltau_correlation, kendalltau_pvalue = kendalltau(gold_ranking_list[i], predicted_ranking_list[i])
            if self.debugging_mode: 
                print('Entity Relatedness : ' +  entities_list[i] + ' kendall tau correlation ' + str(kendalltau_correlation) + ' kendall tau pvalue ' + str(kendalltau_pvalue))
            score_list.append({'task_name' : 'Entity Relatedness', 'entity_name' : entities_list[i], 'kendalltau_correlation': kendalltau_correlation, 'kendalltau_pvalue': kendalltau_pvalue})

        return score_list