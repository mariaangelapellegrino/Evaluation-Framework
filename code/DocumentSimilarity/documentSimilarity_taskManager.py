from documentSimilarity_model import DocumentSimilarityModel as Model
import csv
import os
from collections import defaultdict

from code.abstract_taskManager import AbstractTaskManager

class DocumentSimilarityManager (AbstractTaskManager):
	def __init__(self, data_manager, distance_metric, debugging_mode):
		self.debugging_mode = debugging_mode
		self.data_manager = data_manager
		self.distance_metric = distance_metric
		self.task_name = 'document_similarity'
		if self.debugging_mode:
			print("Document Similarity task manager initialized")

	def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary= None):
		log_errors = ""
		
		stats_filename = "LP50_averageScores.csv"

		script_dir = os.path.dirname(__file__)
		rel_path = "data/"+stats_filename
		stats_file = os.path.join(script_dir, rel_path)
		
		stats = self.data_manager.read_file(stats_file, ['doc1', 'doc2', 'average'])

		document_entities_filename = "LP50_entities.json"	
		
		script_dir = os.path.dirname(__file__)
		rel_path = "data/"+document_entities_filename
		document_entities_file = os.path.join(script_dir, rel_path)
				
		data, ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, document_entities_file)

		self.storeIgnored(results_folder, 'LP50', ignored)

		scores = list()

		if data.size == 0:
			log_errors += 'Document similarity : Problems in merging vector with gold standard ' + document_entities_file + '\n'
			if self.debugging_mode:
				print('Document similarity : Problems in merging vector with gold standard ' + document_entities_file)
		else:
			try:
				
				scores = defaultdict(list)
				with_weights = False
				model = Model(self.distance_metric, with_weights, self.debugging_mode)
				result, log_info = model.train(data, stats)
				scores['without_weights'] = result
				#log_errors += log_info
				
				with_weights = True
				model = Model(self.distance_metric, with_weights, self.debugging_mode)
				result, log_info = model.train(data, stats)
				scores['with_weights'] = result
				log_errors += log_info

				self.storeResults(results_folder, 'LP50', scores)
			except Exception as e:
				log_errors += 'File used as gold standard: ' + document_entities_file + '\n'
				log_errors += 'Document similarity, with weights: ' + str(with_weights) + '\n'
				log_errors += str(e) + '\n'
	
		if not log_dictionary is None:
			log_dictionary['Document Similarity'] = log_errors
		return log_errors
		

	def storeIgnored(self, results_folder, gold_standard_filename, ignored):
		ignored = ignored.drop_duplicates()

		if self.debugging_mode: 
			print('Document similarity: Ignored data : ' + str(len(ignored)))

		file_ignored = open(results_folder+'/documentSimilarity_'+gold_standard_filename+'_ignoredData.txt',"w") 
		for ignored_tuple in ignored.itertuples():
			if self.debugging_mode:
				print('Document similarity : Ignored data: ' + getattr(ignored_tuple,'name'))
			file_ignored.write(getattr(ignored_tuple,'name').encode('utf-8')+'\n')
		file_ignored.close()

	def storeResults(self, results_folder, gold_standard_filename, scores):
		with open(results_folder+'/documentSimilarity_'+gold_standard_filename+'_results.csv', "wb") as csv_file:
			fieldnames = ['task_name', 'conf', 'pearson_score', 'spearman_score', 'harmonic_mean']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			
			for (method, score) in scores.items():
				writer.writerow(score)
				if self.debugging_mode:
					print('Document Similarity ' + method + ' score: ' +   str(score))     