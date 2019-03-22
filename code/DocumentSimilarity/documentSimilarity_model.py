from scipy.stats import spearmanr
from scipy.stats import pearsonr
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
		doc_similarity, log_info = self.compute_doc_distance(data)
		gold_similarity_score, similarity_score = self.get_gold_and_actual_score(stats, doc_similarity)
		pearson_score, spearman_score, harmonic_mean = self.evaluate_document_similarity(gold_similarity_score, similarity_score)	
			
		if self.with_weights:
			conf = 'without_weights'
		else:
			conf = 'with weights'

		return {'task_name' : 'Document Similarity', 'conf': conf, 'pearson_score': pearson_score, 'spearman_score' : spearman_score, 'harmonic_mean' : harmonic_mean}, log_info

	def compute_doc_distance(self, data):
		log_info = ""
		
		result_dict = {}
		doc1_list = list()
		doc2_list = list()
		doc_similarity = list()

		for i in range(1, 51):
			#1. extract entities of document i and j
			set1 = self.extract_entities(i, data)
			weight1 = set1['weight']
			if len(set1)==0:
				log_info += "Document Similarity: No entities in doc " + str(i) + "\n"
				continue
			
			for j in range(i, 51):
				set2 = self.extract_entities(j, data)
				weight2 = set2['weight']
				if len(set2)==0:
					log_info += "Document Similarity: No entities in doc " + str(j) + "\n"
					continue
			
				#1. DONE : set 1 and set 2 contain entities of document i and j
				
				#2. compute the similarity for each pair of entities in d_i and d_j
				# similarity is interpreted as the opposite of distance 
				distance_score1 = self.compute_distance(set1, set2)
				distance_score2 = self.compute_distance(set2, set1)

				#3. for each entity in d_i identify the maximum similarity to an entity in d2, and viceversa				
				max_sim1 = list()
				for k in range(len(distance_score1)):
					weight = weight1.iloc[k]
					max_sim1.append(self.compute_max_similarity(distance_score1[k, :], weight, weight2))

			
				max_sim2 = list()
				for k in range(len(distance_score2)):
					weight = weight2.iloc[k]
					max_sim2.append(self.compute_max_similarity(distance_score2[k, :], weight, weight1))
			
				#4. calculate document similarity
				sum_max_sim1 = sum(max_sim1)
				sum_max_sim2 = sum(max_sim2)
			
				document_similarity = (sum_max_sim1+sum_max_sim2)/(len(set1)+len(set2))

				if self.debugging_mode:
					print("Doc " + str(i) + " - Doc " + str(j) + " : distance similarity " + str(document_similarity))

				doc1_list.append(i)
				doc2_list.append(j)
				doc_similarity.append(document_similarity)

		result_dict['doc1'] = doc1_list
		result_dict['doc2'] = doc2_list
		result_dict['similarity'] = doc_similarity

		return pd.DataFrame(result_dict), log_info

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
	
		max_distance = max(distance_list)
		if max_distance!=0:
			normalized_distance = min_distance_score/max_distance
		else:
			normalized_distance = min_distance_score
		similarity_score = 1-normalized_distance
		
		if self.with_weights:
			weight2 = weight_list.iloc[index_min_distance]
			similarity_score = similarity_score * (weight1 * weight2)

		return similarity_score