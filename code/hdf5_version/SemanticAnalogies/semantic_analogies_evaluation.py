import numpy as np
import csv
import semantic_analogies_data_manager as data_manager
import semantic_analogies_model as model


class Evaluator:
    def __init__(self):
        print('SemanticAnalogies evaluator initialized')

    @staticmethod
    def evaluate(vector_filename, vector_size, analogy_function, top_k, results_folder):
        vocab, W = data_manager.DataManager.get_vocab_and_W(vector_filename, vector_size)
        W_norm = model.Model.normalize_vectors(W)

        gold_standard_filenames = ['capital_country_entities', 'all_capital_country_entities',
           'currency_entities', 'city_state_entities']

        scores = list()

        for gold_standard_filename in gold_standard_filenames:
            gold_standard_file = 'SemanticAnalogies/data/'+gold_standard_filename+'.txt'

            data, ignored = data_manager.DataManager.read_data(vector_filename, gold_standard_file)

            print('Semantic analogies: Ignored quadruplet: ' + str(len(ignored)))

            with open(results_folder+'/semanticAnalogies_'+gold_standard_filename+'_ignoredData.csv', 'wb') as file_result:
                fieldnames = ['entity_1', 'entity_2', 'entity_3', 'entity_4']
                writer = csv.DictWriter(file_result, fieldnames=fieldnames)
                writer.writeheader()

                for ignored_quadruplet in ignored:
                    print('Semantic analogies: Ignored quadruplet: ' + str(ignored_quadruplet))
                    writer.writerow({'entity_1':str(ignored_quadruplet[0]), 'entity_2':str(ignored_quadruplet[1]), 'entity_3':str(ignored_quadruplet[2]), 'entity_4':str(ignored_quadruplet[3])})

            if len(data) == 0:
                print('SemanticAnalogies : Problems in merging vector with gold standard ' + gold_standard_file)
            else:
                right_answers, tot_answers, accuracy = model.Model.compute_semantic_analogies(vocab, data, W_norm, analogy_function, top_k)
                scores.append({'task_name':'Semantic Analogies', 'gold_standard_file': gold_standard_filename, 'top_k_value':top_k, 'right_answers':right_answers, 'tot_answers':tot_answers, 'accuracy':accuracy})

        with open(results_folder+'/semanticAnalogies_results.csv', 'wb') as file_result:
            fieldnames = ['task_name', 'gold_standard_file', 'top_k_value', 'right_answers', 'tot_answers', 'accuracy']
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()
            for score in scores:
                writer.writerow(score)

