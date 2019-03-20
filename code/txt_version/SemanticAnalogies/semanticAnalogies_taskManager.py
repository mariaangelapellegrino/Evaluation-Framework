import csv
from code.txt_version.SemanticAnalogies.semanticAnalogies_dataManager import DataManager
from code.txt_version.SemanticAnalogies.semanticAnalogies_model import SemanticAnalogiesModel as Model

from code.abstract_taskManager import AbstractTaskManager

class SemanticAnalogiesManager (AbstractTaskManager):
    def __init__(self, similarity_metric, analogy_function, top_k, debugging_mode):
        self.data_manager = DataManager()
        self.similarity_metric = similarity_metric
        self.analogy_function = analogy_function
        self.top_k = top_k
        self.debugging_mode = debugging_mode
        if debugging_mode:
            print('SemanticAnalogies task Manager initialized')

    def evaluate(self, vectors, results_folder):
        vec_size = vectors.shape[1]

        vocab = self.data_manager.create_vocab(vectors)
        W_norm = self.data_manager.normalize_vectors(vectors, vec_size, vocab)

        gold_standard_filenames = ['capital_country_entities', 'all_capital_country_entities','currency_entities', 'city_state_entities']

        scores = list()

        for gold_standard_filename in gold_standard_filenames:
            gold_standard_file = 'SemanticAnalogies/data/'+gold_standard_filename+'.txt'

            data, ignored = self.data_manager.read_data(vocab, gold_standard_file)
            self.manageIgnored(results_folder, gold_standard_filename, ignored)
            if len(data) == 0:
                print('SemanticAnalogies : Problems in merging vector with gold standard ' + gold_standard_file)
            else:
                model = Model()
                #add similarity function
                result = model.train(vocab, data, W_norm, self.analogy_function, self.top_k, self.debugging_mode)
                result['gold_standard_file'] = gold_standard_file
                scores.append(result)

        self.storeResults(results_folder, gold_standard_file, scores)

    def manageIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print('Semantic analogies: Ignored quadruplet: ' + str(len(ignored)))

        with open(results_folder+'/semanticAnalogies_'+gold_standard_filename+'_ignoredData.csv', 'wb') as file_result:
            fieldnames = ['entity_1', 'entity_2', 'entity_3', 'entity_4']
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()

            for ignored_quadruplet in ignored:
                if self.debugging_mode:
                    print('Semantic analogies: Ignored quadruplet: ' + str(ignored_quadruplet))
                writer.writerow({'entity_1':str(ignored_quadruplet[0]), 'entity_2':str(ignored_quadruplet[1]), 'entity_3':str(ignored_quadruplet[2]), 'entity_4':str(ignored_quadruplet[3])})
            
    def storeResults(self, results_folder, gold_standard_file, scores):
        with open(results_folder+'/semanticAnalogies_results.csv', 'wb') as file_result:
            fieldnames = ['task_name', 'gold_standard_file', 'top_k_value', 'right_answers', 'tot_answers', 'accuracy']
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()
            for score in scores:
                writer.writerow(score)
