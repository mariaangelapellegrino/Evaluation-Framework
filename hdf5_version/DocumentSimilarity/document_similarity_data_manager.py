import pandas as pd
import numpy as np
import json
import base64
import h5py

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
				doc_list.append(i)
				entities_list.append(base64.b32encode(annotation['entity']))
				weight_list.append(float(annotation['weight']))

		dict_entities['doc'] = doc_list
		dict_entities['name'] = entities_list
		dict_entities['weight'] = weight_list

		return pd.DataFrame.from_dict(dict_entities)

	@staticmethod
	def create_header(vec_size):
		headers = ['doc', 'name', 'weight']
		for i in range(0, vec_size):
			headers.append(i)
		return headers

	@staticmethod
	def merge_entities_vectors(vector_filename, vector_size, entities):
		vector_file = h5py.File(vector_filename, 'r')
		vector_group = vector_file["Vectors"]

		merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
		ignored = list()
        
		for row in entities.itertuples():
			try:
				values = vector_group[row.name][0]
                        
				new_row = dict(zip(np.arange(vector_size), values))
				new_row['doc'] = row.doc
				new_row['name'] = row.name
				new_row['weight'] = row.weight

				merged = merged.append(new_row, ignore_index=True)
			except KeyError:
				ignored.append(base64.b32decode(row.name))

		return (merged, ignored)
