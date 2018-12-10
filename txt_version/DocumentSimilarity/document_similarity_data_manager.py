import pandas as pd
import json

class DataManager:
	def __init__(self):
		print("Document similarity data manager initialized")

	@staticmethod
	def read_stats(stats_filename):
		fields = ['Document1', 'Document2', 'Similarity']
		stats = pd.read_csv(stats_filename, ",",  usecols=fields, index_col=False, skipinitialspace=True, skip_blank_lines=True)
		return stats

	@staticmethod
	def read_stats_average_scores(stats_filename):
		fields = ['doc1', 'doc2', 'average']
		stats = pd.read_csv(stats_filename, ",",  usecols=fields, index_col=False, skipinitialspace=True, skip_blank_lines=True)
		return stats
   
	@staticmethod
	def read_entities(document_entities_filename):

		with open(document_entities_filename) as f:
			data = json.load(f)

		dict_entities = {}
		doc_list = list()
		entities_list = list()
		weight_list = list()
		i=0
	
		for doc_obj in data:
			i += 1
			for annotation in doc_obj["annotations"]:
				key = annotation['entity']

				doc_list.append(i)
				entities_list.append(key)
				weight_list.append(float(annotation['weight']))
	
		dict_entities['doc'] = doc_list
		dict_entities['name'] = entities_list
		dict_entities['weight'] = weight_list

		return pd.DataFrame.from_dict(dict_entities)

	@staticmethod
	def merge_entities_vectors(entities, vectors):
		merged = pd.merge(entities, vectors, on='name', how='inner')
		outputLeftMerge = pd.merge(entities, vectors, how='outer', indicator=True)
		ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

		return (merged, ignored)
