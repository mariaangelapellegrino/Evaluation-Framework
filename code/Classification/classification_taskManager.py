from classification_model import ClassificationModel as Model
import csv
from collections import defaultdict
import codecs
import os

from code.abstract_taskManager import AbstractTaskManager

class ClassificationManager (AbstractTaskManager):    
    def __init__(self, data_manager, debugging_mode):
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        if self.debugging_mode:
            print("Classification task manager initialized")

    def evaluate(self, vectors, vector_file, vector_size, results_folder, log_dictionary = None):
        log_errors = ""
        
        gold_standard_filenames = ['Cities', 'MetacriticMovies', 'MetacriticAlbums', 'AAUP', 'Forbes']
        
        for gold_standard_filename in gold_standard_filenames:
            script_dir = os.path.dirname(__file__)
            rel_path = "data/"+gold_standard_filename+'.tsv'
            gold_standard_file = os.path.join(script_dir, rel_path)
            
            classification_model_names = ["NB", "KNN", "C45"]
            SVM_configurations = [pow(10, -3), pow(10, -2), 0.1, 1.0, 10.0, pow(10, 2), pow(10, 3)]

            scores = defaultdict(list)
            data, ignored = self.data_manager.intersect_vectors_goldStandard(vectors, vector_file, vector_size, gold_standard_file)

            self.storeIgnored(results_folder, gold_standard_filename, ignored)

            if data.size == 0:
                log_errors += 'Classification : Problems in merging vector with gold standard ' + gold_standard_file + '\n'
                if self.debugging_mode:
                    print('Classification : Problems in merging vector with gold standard ' + gold_standard_file)
            else:
                for i in range(10):
                    data = data.sample(frac=1, random_state=i).reset_index(drop=True)
                    for model_name in classification_model_names:
                        # initialize the model
                        model = Model(model_name, self.debugging_mode)
                        # train and print score
                        try:
                            result = model.train(data)
                            scores[model_name].append(result)
                            
                            self.storeResults(results_folder, gold_standard_filename, scores)
                        except Exception as e:
                            log_errors += 'File used as gold standard: ' + gold_standard_filename + '\n'
                            log_errors += 'Classification method: ' + model_name + '\n'
                            log_errors += str(e) + '\n'
                            
                    for conf in SVM_configurations:
                        # initialize the model
                        model = Model("SVM", conf)
                        # train and print score
                        try:
                            result = model.train(data)
                            scores["SVM"].append(result)
                            
                            self.storeResults(results_folder, gold_standard_filename, scores)
                        except Exception as e:
                            log_errors += 'File used as gold standard: ' + gold_standard_filename + '\n'
                            log_errors += 'Classification method: SVM ' + str(conf) + '\n'
                            log_errors += str(e) + '\n'

        if not log_dictionary is None:
            log_dictionary["Classification"] = log_errors
            
        return log_errors
                
    def storeIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Classification : Ignored data: ' + str(len(ignored)))
        
        file_ignored = codecs.open(results_folder+'/classification_'+gold_standard_filename+'_ignoredData.txt',"w", 'utf-8') 
        for ignored_tuple in ignored.itertuples():        
            value = getattr(ignored_tuple,'name')
            if self.debugging_mode:
                print('Classification : Ignored data: ' + value.encode(encoding='UTF-8', errors='ignore'))
            
            if isinstance(value, str):
                value = unicode(value, "utf-8").encode(encoding='UTF-8', errors='ignore')
            file_ignored.write(value+'\n')
        
        file_ignored.close()  

    def storeResults(self, results_folder, gold_standard_filename, scores):
        with open(results_folder+'/classification_'+gold_standard_filename+'_results.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'model_name', 'model_configuration', 'score_type', 'score_value']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for (method, scoresForMethod) in scores.items():
                for score in scoresForMethod:
                    writer.writerow(score)
                    if self.debugging_mode:
                        print('Classification ' + method + ' score: ' +   score)      

                        
