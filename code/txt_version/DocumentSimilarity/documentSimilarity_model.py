from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from code.abstract_model import AbstractModel

class DocumentSimilarityModel(AbstractModel):

	def __init__(self, distance_metric, with_weights, debugging_mode):
		self.debugging_mode = debugging_mode
		self.distance_metric = distance_metric
		self.with_weights = with_weights
		if self.debugging_mode:
			print('Document similarity model initialized')

	def train(self, data, stats):
		doc_similarity, log_info = self.compute_doc_distance(data, self.distance_metric)
		gold_similarity_score, similarity_score = self.get_gold_and_actual_score(stats, doc_similarity)
		pearson_score, spearman_score, harmonic_mean = self.evaluate_document_similarity(gold_similarity_score, similarity_score)	
			
		if self.with_weights:
			conf = 'without_weights'
		else:
			conf = 'with weights'

		return {'task_name' : 'Document Similarity', 'conf': conf, 'pearson_score': pearson_score, 'spearman_score' : spearman_score, 'harmonic_mean' : harmonic_mean}

	def compute_doc_distance(self, data, distance_metric):
		log_info = ""
		
		min_distance_score1 = 0
		min_distance_score2 = 0

		result_dict = {}
		doc1_list = list()
		doc2_list = list()
		doc_similarity = list()

		for i in range(1, 51):
			#1. extract entities of document i and j
			set1 = self.extractEntities(i, data)
			weight1 = set1[2]
			if len(set1)==0:
				log_info += "No entities in doc " + i
				continue
			
			for j in range(i, 51):
				set2 = self.extractEntities(j, data)
				weight2 = set2[2]
				if len(set2)==0:
					log_info += "No entities in doc " + i
					continue
			
				#1. DONE : set 1 and set 2 contain entities of document i and j
				
				#2. compute the similarity for each pair of entities in d_i and d_j
				# similarity is interpreted as the opposite of distance 
				distance_score1 = self.compute_distance(set1, set2)
				
				max_sim1 = list()
				for k in range(len(distance_score1)):
					weight1 = weight1.iloc[2]
					weight2 = set2.iloc[index_min_distance, 2]
					max_sim1.append(self.compute_max_similarity(distance_score1[k, :]))

				


				distance_score2 = pairwise_distances(set2.iloc[:, 3:], set1.iloc[:, 3:], metric=distance_metric)
			
				min_distance_score_21_list = list()
				max_similarity_score_21_list = list()
				for k in range(len(distance_score2)):
					index_min_distance = np.argmin(distance_score2[k, :])
					min_distance_score2 = distance_score2[k, index_min_distance]

					if self.with_weights:
						weight2 = set2.iloc[k, 2]
						weight1 = set1.iloc[index_min_distance, 2]
						min_distance_score2 = min_distance_score2 / (weight1 * weight2)
			
					min_distance_score_21_list.append(min_distance_score2)
					
				max_distance = max(min_distance_score_21_list)
				for distance in min_distance_score_21_list:
					normalized_distance = distance/max_distance
					similarity_score = 1-normalized_distance
					max_similarity_score_21_list.append(similarity_score)
			
				document_distance = (sum(max_similarity_score_12_list)+sum(max_similarity_score_21_list))/(len(set1)+len(set2))
				if self.debugging_mode:
					print("Doc " + str(i) + " - Doc " + str(j) + " : distance score " + str(document_distance))

				doc1_list.append(i)
				doc2_list.append(j)
				doc_similarity.append(document_distance)

		result_dict['doc1'] = doc1_list
		result_dict['doc2'] = doc2_list
		result_dict['similarity'] = doc_similarity

		return pd.DataFrame(result_dict)

	def get_gold_and_actual_score(self, gold_stats, actual_stats):
		merged = pd.merge(gold_stats, actual_stats, on=['doc1', 'doc2'], how='inner')
		return (merged.iloc[:, 2], merged.iloc[:, 3])

	def evaluate_document_similarity(self, gold_similarity_score, similarity_score):
		spearman_score, _value = spearmanr(gold_similarity_score, similarity_score)
		pearson_score, _value = pearsonr(gold_similarity_score, similarity_score)
		harmonic_mean = (2*pearson_score*spearman_score)/(pearson_score+spearman_score)
		return pearson_score, spearman_score, harmonic_mean
	
	def extract_entities(self, documentID, data):
		set1 = data[data['doc']==documentID]
		if len(set1) == 0:
			if self.debugging_mode:
				print('No entities in doc ' + str(documentID))
		else:	
			set1 = set1.sort_values('weight', ascending=False).drop_duplicates(subset='name', keep='first')
		return set1
	
	def compute_distance(self, set1, set2):
		return pairwise_distances(set1.iloc[:, 3:], set2.iloc[:, 3:], metric=self.distance_metric)
	
	def compute_max_similarity(self, distance_list, weight1, weight_list):
		index_min_distance = np.argmin(distance_list)
		min_distance_score = distance_list[index_min_distance]
	
		if self.with_weights:
			
			min_distance_score1 = min_distance_score1 / (weight1 * weight2)

					min_distance_score_12_list.append(min_distance_score1)

				max_distance = max(min_distance_score_12_list)
				for distance in min_distance_score_12_list:
					normalized_distance = distance/max_distance
					similarity_score = 1-normalized_distance
					max_similarity_score_12_list.append(similarity_score)