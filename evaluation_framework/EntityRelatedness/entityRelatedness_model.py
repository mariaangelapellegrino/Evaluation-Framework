# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance
from scipy.stats import kendalltau
from evaluation_framework.abstract_model import AbstractModel

float_precision = 15

"""
Model of the entity relatedness task
"""
class EntityRelatednessModel(AbstractModel):

    """
    It initialize the model of the entity relatedness task
    
    task_name: name of the task
    distance_metric: distance metric which will be used to calculate the similarity score
    debugging_mode: {TRUE, FALSE}, TRUE to run the model by reporting all the errors and information; FALSE otherwise
    """
    def __init__(self, task_name, distance_metric, debugging_mode):
        self.debugging_mode = debugging_mode
        self.distance_metric = distance_metric
        self.task_name = task_name
        if self.debugging_mode:
            print('Entity relatedness model initialized')
    
    """
    It trains the model based on the provided data
    
    left_merged: dataframe containing main entities and the related vectors
    left_ignored: dataframe containing the missing main entities, i.e. the entities which are in the dataset but not in the input file
    right_merged_list: list of dataframes containing the right entities and the related vectors
    right_ignored_list: list of dataframes containing the missing right entities, i.e. the entities which are in the dataset but not in the input file
    groups: dictionary containing the main entities and the attached right entities
    
    It returns the result object reporting the task name, the model name and its configuration - if any -, and the evaluation metric values.
    """
    def train(self, left_merged, left_ignored, right_merged_list, right_ignored_list, groups):
        predicted_rank_list = self.compute_relatedness(left_merged, left_ignored, right_merged_list, right_ignored_list)
        gold_rank_list = np.tile(np.arange(1, 21), (21, 1))
        return self.evaluate_ranking(groups.keys(), gold_rank_list, predicted_rank_list)

    """
    It computes the relatedness among main and right entities.
    
    left_merged: dataframe containing main entities and the related vectors
    left_ignored: dataframe containing the missing main entities, i.e. the entities which are in the dataset but not in the input file
    right_merged_list: list of dataframes containing the right entities and the related vectors
    right_ignored_list: list of dataframes containing the missing right entities, i.e. the entities which are in the dataset but not in the input file
    
    It returns the predicted ranked list.
    """
    def compute_relatedness(self, left_merged, left_ignored, right_merged_list, right_ignored_list):
        predicted_rank_list = list()       

        for i in range(len(left_merged['name'])):
            distances = []

            if right_merged_list[i].size>0:
                distances = distance.cdist(left_merged.iloc[[i], 1:], right_merged_list[i].iloc[:, 1:], metric= self.distance_metric).flatten() 
                        
            ignored_distances = np.array([1 for j in range(len(right_ignored_list[i]))]) #max dist?? 
            distances = np.concatenate((distances, ignored_distances))

            predicted_rank_list.append(list(np.argsort(distances)))

        for i in range(len(left_ignored['name'])):
            predicted_rank_list.append(list(np.argsort([1 for i in range(20)])))

        return predicted_rank_list

    """
    It evaluates the ranking comparing the predicted and the gold one.
    
    entities_list: it keeps the relation among the main entities and the right entities
    gold_ranking_list: list of the ranking used as gold standard
    predicted_ranking_list: list of the predicted ranking
    
    It returns the evaluation of the predicted ranking.
    """
    def evaluate_ranking(self, entities_list, gold_ranking_list, predicted_ranking_list):
        score_list = list()

        for i in range(len(entities_list)):
            if self.debugging_mode:
                print('Entity Relatedness : ' +  entities_list[i])
                print(gold_ranking_list[i])
                print(predicted_ranking_list[i])
            kendalltau_correlation, kendalltau_pvalue = kendalltau(gold_ranking_list[i], predicted_ranking_list[i])
            if self.debugging_mode: 
                print('Entity Relatedness : ' +  entities_list[i] + ' kendalltau correlation ' + str(kendalltau_correlation) + ' kendall tau pvalue ' + str(kendalltau_pvalue))
            score_list.append({'task_name' : self.task_name, 'entity_name' : entities_list[i], 'kendalltau_correlation': round(kendalltau_correlation,float_precision), 'kendalltau_pvalue': round(kendalltau_pvalue, float_precision)})

        return score_list