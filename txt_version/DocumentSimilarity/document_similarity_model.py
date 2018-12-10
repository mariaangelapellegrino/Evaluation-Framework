from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.spatial import distance
import pandas as pd
import numpy as np

class Model:

	def __init__(self):
		print('Document similarity model initialized')

	@staticmethod
	def compute_doc_distance(data, distance_metric, with_weights=False):
		min_distance_score1 = 0
		min_distance_score2 = 0

		result_dict = {}
		doc1_list = list()
		doc2_list = list()
		doc_distance = list()

		for i in range(1, 51):
			set1 = data[data['doc']==i]
			if len(set1) == 0:
				print('No entities in doc ' + str(i))
				continue

			for j in range(i, 51):
				set2 = data[data['doc']==j]
				if len(set2) == 0:
					print('No entities in doc ' + str(j))
					continue

				distance_score1 = distance.cdist(set1.iloc[:, 3:], set2.iloc[:, 3:], metric=distance_metric)

				for k in range(len(distance_score1)):
					index_max = np.argmax(-distance_score1[k, :])
					min_distance_score1 = distance_score1[k, index_max]
	
					if with_weights:
						weight1 = set1.iloc[k, 2]
						weight2 = set2.iloc[index_max, 2]
						min_distance_score1 = min_distance_score1 / (weight1 * weight2)

				distance_score2 = distance.cdist(set2.iloc[:, 3:], set1.iloc[:, 3:], metric=distance_metric)
			
				for k in range(len(distance_score2)):
					index_max = np.argmax(-distance_score2[k, :])
					min_distance_score2 = distance_score2[k, index_max]

					if with_weights:
						weight2 = set2.iloc[k, 2]
						weight1 = set1.iloc[index_max, 2]
						min_distance_score2 = min_distance_score2 / (weight1 * weight2)

			
				document_similarity = (min_distance_score1+min_distance_score2)/(len(set1)+len(set2))
				print("Doc " + str(i) + " - Doc " + str(j) + " : similarity score " + str(document_similarity))

				doc1_list.append(i)
				doc2_list.append(j)
				doc_distance.append(document_similarity)

		result_dict['doc1'] = doc1_list
		result_dict['doc2'] = doc2_list
		result_dict['similarity'] = doc_distance

		return pd.DataFrame(result_dict)

	@staticmethod
	def get_gold_and_actual_score(gold_stats, actual_stats):
		merged = pd.merge(gold_stats, actual_stats, on=['doc1', 'doc2'], how='inner')
		return (merged.iloc[:, 2], merged.iloc[:, 3])

	@staticmethod
	def evaluate_document_similarity(gold_similarity_score, similarity_score):
		spearman_score, pval = spearmanr(gold_similarity_score, similarity_score, axis=None)
		pearson_score, prob = pearsonr(gold_similarity_score, similarity_score)
		harmonic_mean = (2*pearson_score*spearman_score)/(pearson_score+spearman_score)
		return (pearson_score, spearman_score, harmonic_mean)
