import document_similarity_data_manager as data_manager
import document_similarity_model as model
import csv

class Evaluator:
	def __init__(self):
		print("Classification and regression evaluator init")

	@staticmethod
	def evaluate(vectors, distance_metric, results_folder):
		stats_filename = "DocumentSimilarity/data/LP50_averageScores.csv"
		stats = data_manager.DataManager.read_stats_average_scores(stats_filename)

		document_entities_filename = "DocumentSimilarity/data/LP50_entities.json"	
		doc_entities = data_manager.DataManager.read_entities(document_entities_filename)
				
		data, ignored = data_manager.DataManager.merge_entities_vectors(doc_entities, vectors)

		print('Document similarity: Ignored data : ' + str(len(ignored)))

		file_ignored = open(results_folder+'/documentSimilarity_LP50_ignoredData.txt',"w") 
		for ignored_tuple in ignored.itertuples():
			print('Document similarity : Ignored data: ' + getattr(ignored_tuple,'name'))
			file_ignored.write(getattr(ignored_tuple,'name').encode('utf-8')+'\n')
		file_ignored.close()

		scores = list()

		if data.size == 0:
			print('Document similarity : Problems in merging vector with gold standard LP50')
		else:
			doc_similarity = model.Model.compute_doc_distance(data, distance_metric, False)
			gold_similarity_score, similarity_score = model.Model.get_gold_and_actual_score(stats, doc_similarity)
			pearson_score, spearman_score, harmonic_mean = model.Model.evaluate_document_similarity(gold_similarity_score, similarity_score)	
			
			scores.append({'task_name' : 'Document Similarity', 'conf': 'ignoring weight', 'pearson_score': pearson_score, 'spearman_score' : spearman_score, 'harmonic_mean' : harmonic_mean})

			doc_similarity = model.Model.compute_doc_distance(data, distance_metric, True)
			gold_similarity_score, similarity_score = model.Model.get_gold_and_actual_score(stats, doc_similarity)
			pearson_score, spearman_score, harmonic_mean = model.Model.evaluate_document_similarity(gold_similarity_score, similarity_score)	
			
			scores.append({'task_name' : 'Document Similarity', 'conf': 'with weight', 'pearson_score': pearson_score, 'spearman_score' : spearman_score, 'harmonic_mean' : harmonic_mean})

		with open(results_folder+'/documentSimilarity_LP50_results.csv', "wb") as csv_file:
			fieldnames = ['task_name', 'conf', 'pearson_score', 'spearman_score', 'harmonic_mean']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			
			for score in scores:
				writer.writerow(score)