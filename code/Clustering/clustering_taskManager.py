from collections import defaultdict
from clustering_model import ClusteringModel as Model
import csv
import os

from code.abstract_taskManager import AbstractTaskManager

class ClusteringManager (AbstractTaskManager):
    def __init__(self, data_manager, distance_metric, debugging_mode):
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        self.distance_metric = distance_metric
        self.task_name = 'clustering'
        if self.debugging_mode:
            print("Clustering task manager initialized")

    def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary= None):
        log_errors = ""
        
        gold_standard_filenames = ['citiesAndCountries_cluster', 'cities2000AndCountries_cluster', 'citiesMoviesAlbumsCompaniesUni_cluster', 'teams_cluster']
        n_clusters_list = [2, 2, 5, 2]

        for i in range(len(gold_standard_filenames)):
            gold_standard_filename = gold_standard_filenames[i]
            
            script_dir = os.path.dirname(__file__)
            rel_path = "data/"+gold_standard_filename+'.tsv'
            gold_standard_file = os.path.join(script_dir, rel_path)
            
            n_clusters = n_clusters_list[i]

            clustering_models = ["DB", "KMeans", "AC", "WHC"]

            data, ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, gold_standard_file)
            
            self.storeIgnored(results_folder, gold_standard_filename, ignored)

            scores = defaultdict(list)
            if data.size == 0:
                log_errors += 'Clustering : Problems in merging vector with gold standard ' + gold_standard_file + '\n'
                if self.debugging_mode:
                    print('Clustering : Problems in merging vector with gold standard ' + gold_standard_file)
            else:               
                for model_name in clustering_models:
                    model = Model(model_name, self.distance_metric, n_clusters, self.debugging_mode)

                    try:                    
                        result = model.train(data, ignored)
                        scores[model_name].append(result)
                                
                        self.storeResults(results_folder, gold_standard_filename, scores)
                    except Exception as e:
                        log_errors += 'File used as gold standard: ' + gold_standard_filename + '\n'
                        log_errors += 'Clustering method: ' + model_name + '\n'
                        log_errors += str(e) + '\n'
        if not log_dictionary is None:
            log_dictionary['Clustering'] = log_errors
        return log_errors
    
    def storeIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Clustering: Ignored data : ' + str(len(ignored)))

        file_ignored = open(results_folder+'/clustering_'+gold_standard_filename+'_ignoredData.txt',"w") 
        for ignored_tuple in ignored.itertuples():
            if self.debugging_mode:
                print('Clustering : Ignored data: ' + getattr(ignored_tuple,'name'))
            file_ignored.write(getattr(ignored_tuple,'name').encode('utf-8')+'\n')
        file_ignored.close()

    def storeResults(self, results_folder, gold_standard_filename, scores):
        with open(results_folder+'/clustering_'+gold_standard_filename+'_results.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'model_name', 'model_configuration', 'num_clusters', 'adjusted_rand_index', 'adjusted_mutual_info_score', 
        'homogeneity_score', 
        'completeness_score', 'v_measure_score'] #'fowlkes_mallows_score', 
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for (method, scoresForMethod) in scores.items():
                for score in scoresForMethod:
                    writer.writerow(score)
                    if self.debugging_mode:
                        print('Clustering ' + method + ' score: ' +   score)      