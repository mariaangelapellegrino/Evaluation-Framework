import pandas as pd
import json

from code.abstract_dataManager import AbstractDataManager

class DataManager(AbstractDataManager):
	def __init__(self, debugging_mode):
		self.debugging_mode = debugging_mode
		if self.debugging_mode:
			print("Document similarity data manager initialized")

	def read_vector_file(self, filename, vector_size):
		super.read_vector_file(filename, vector_size)

	def read_file(self, filename, columns):
		stats = pd.read_csv(filename, ",",  usecols=columns, index_col=False, skipinitialspace=True, skip_blank_lines=True)
		return stats

	def intersect_vectors_goldStandard(self, vectors, goldStandard_filename, 
		vector_filename = None, vector_size = None, goldStandard_data = None, 
		column_id = None, column_score = None):

		entities = self.get_entities(goldStandard_filename)
		merged = pd.merge(entities, vectors, on='name', how='inner')
		outputLeftMerge = pd.merge(entities, vectors, how='outer', indicator=True)
		ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

		return merged, ignored['name']

	def get_entities(self, filename):
		with open(filename) as f:
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