from code.txt_version.Regression.regression_dataManager import DataManager
from code.txt_version.Regression.regression_model import RegressionModel as Model
import csv
from collections import defaultdict
import codecs
import os

from code.abstract_taskManager import AbstractTaskManager

class RegressionManager (AbstractTaskManager):
    def __init__(self, debugging_mode):
        self.debugging_mode = debugging_mode
        self.data_manager = DataManager(self.debugging_mode)
        if debugging_mode:
            print("Regression task Manager initialized")

    def evaluate(self, vectors, results_folder):
        log_errors = ""
        
        gold_standard_filenames = ['Cities', 'MetacriticMovies', 'MetacriticAlbums', 'AAUP', 'Forbes']

        for gold_standard_filename in gold_standard_filenames:
            script_dir = os.path.dirname(__file__)
            rel_path = "data/"+gold_standard_filename+'.tsv'
            gold_standard_file = os.path.join(script_dir, rel_path)

            regression_model_names = ["LR", "KNN", "M5"]

            scores = defaultdict(list)

            data, ignored = self.data_manager.intersect_vectors_goldStandard(vectors, gold_standard_file)

            self.storeIgnored(results_folder, gold_standard_filename, ignored)

            if data.size == 0:
                log_errors += 'Regression : Problems in merging vector with gold standard ' + gold_standard_file + '\n'
                if self.debugging_mode:
                    print('Regression : Problems in merging vector with gold standard ' + gold_standard_file)
            else:
                for i in range(10):
                    data = data.sample(frac=1, random_state=i).reset_index(drop=True)

                    for model_name in regression_model_names:
                        # initialize the model
                        model = Model(model_name, self.debugging_mode)
                        # train and print score
                        try:
                            result = model.train(data)
                            scores[model_name].append(result)
                            
                            self.storeResults(results_folder, gold_standard_filename, scores)
                        except Exception as e:
                            log_errors += 'File used as gold standard: ' + gold_standard_filename + '\n'
                            log_errors += 'Regression method: ' + model_name + '\n'
                            log_errors += str(e) + '\n'

        return log_errors

    def storeIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Regression : Ignored data: ' + str(len(ignored)))
        
        file_ignored = codecs.open(results_folder+'/regression_'+gold_standard_filename+'_ignoredData.txt',"w", 'utf-8') 
        for ignored_tuple in ignored.itertuples():
            if self.debugging_mode:
                print('Regression : Ignored data: ' + getattr(ignored_tuple,'name'))
            file_ignored.write(getattr(ignored_tuple,'name')+'\n')
        file_ignored.close() 
                
    def storeResults(self, results_folder, gold_standard_filename, scores):
        with open(results_folder+'/regression_'+gold_standard_filename+'_results.csv', "a+") as csv_file:
            fieldnames = ['task_name', 'model_name', 'model_configuration', 'score_type', 'score_value']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for (method, scoresForMethod) in scores.items():
                for score in scoresForMethod:
                    writer.writerow(score)
                    if self.debugging_mode:
                        print('Regression ' + method + ' score: ' +   score)  
           

            

                        
