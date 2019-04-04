from documentSimilarity_model import DocumentSimilarityModel as Model
import csv
import os
import pandas as pd
from collections import defaultdict
from numpy import mean

from code.abstract_taskManager import AbstractTaskManager

task_name = 'DocumentSimilarity'

class DocumentSimilarityManager (AbstractTaskManager):
	def __init__(self, data_manager, distance_metric, debugging_mode):
		self.debugging_mode = debugging_mode
		self.data_manager = data_manager
		self.distance_metric = distance_metric
		if self.debugging_mode:
			print("Document Similarity task manager initialized")
			
	@staticmethod
	def get_task_name():
		return task_name

	def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary, scores_dictionary):
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
				model = Model(task_name, self.distance_metric, with_weights, self.debugging_mode)
				result, log_info = model.train(data, stats)
				result['gold_standard_file'] = 'LP50'
				scores['without_weights'] = result
				#log_errors += log_info
				
				with_weights = True
				model = Model(task_name, self.distance_metric, with_weights, self.debugging_mode)
				result, log_info = model.train(data, stats)
				result['gold_standard_file'] = 'LP50'
				scores['with_weights'] = result
				log_errors += log_info

				self.storeResults(results_folder, 'LP50', scores)
				results_df = self.resultsAsDataFrame(scores)
				scores_dictionary[task_name] = results_df
			except Exception as e:
				log_errors += 'File used as gold standard: ' + document_entities_file + '\n'
				log_errors += 'Document similarity, with weights: ' + str(with_weights) + '\n'
				log_errors += str(e) + '\n'
	
		log_dictionary[task_name] = log_errors		

	def storeIgnored(self, results_folder, gold_standard_filename, ignored):
		ignored = ignored.drop_duplicates()

		if self.debugging_mode: 
			print('Document similarity: Ignored data : ' + str(len(ignored)))

		file_ignored = open(results_folder+'/documentSimilarity_'+gold_standard_filename+'_ignoredData.txt',"w") 
		for ignored_tuple in ignored.itertuples():
			value = getattr(ignored_tuple,'name')
			if self.debugging_mode:
				print('Document similarity : Ignored data: ' + value.encode(encoding='UTF-8', errors='ignore'))

			if isinstance(value, str):
				value = unicode(value, "utf-8").encode(encoding='UTF-8', errors='ignore')
			file_ignored.write(value+'\n')
			
		file_ignored.close()

	def storeResults(self, results_folder, gold_standard_filename, scores):
		with open(results_folder+'/documentSimilarity_'+gold_standard_filename+'_results.csv', "wb") as csv_file:
			fieldnames = ['task_name', 'gold_standard_file', 'conf', 'pearson_score', 'spearman_score', 'harmonic_mean']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			
			for (method, score) in scores.items():
				writer.writerow(score)
				if self.debugging_mode:
					print('Document Similarity ' + method + ' score: ' +   str(score))     
					
	def resultsAsDataFrame(self, scores):
		data_dict = dict()
		data_dict['task_name'] = list()
		data_dict['gold_standard_file'] = list()
		data_dict['model'] = list()
		data_dict['model_configuration'] = list()
		data_dict['metric'] = list()
		data_dict['score_value'] = list()
		
		metrics = self.get_metric_list()
		
		for (key, scoresForMethod) in scores.items():
			for metric in metrics:
				
				metric_scores = list()
				for score in scoresForMethod:
					metric_scores.append(score[metric])
				metric_score = mean(metric_scores)

				score = scoresForMethod[0]

				data_dict['task_name'].append(score['task_name'])
				data_dict['gold_standard_file'].append(score['gold_standard_file'])
				data_dict['model'].append(score['conf'])
				data_dict['model_configuration'].append('-')
				data_dict['metric'].append(metric)
				data_dict['score_value'].append(metric_score)


		results_df = pd.DataFrame(data_dict, columns = ['task_name', 'gold_standard_file', 'model', 'model_configuration', 'metric', 'score_value'])
		return results_df
	
	@staticmethod
	def get_gold_standard_file():
		return ['LP50']

	@staticmethod
	def get_metric_list():
		return ['pearson_score', 'spearman_score', 'harmonic_mean']