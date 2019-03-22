import numpy as np

from code.abstract_model import AbstractModel

def default_analogy_function(a, b, c):
    return np.array(b) - np.array(a) + np.array(c)

class SemanticAnalogiesModel(AbstractModel):
    def __init__(self, similarity_metric, top_k, debugging_mode, analogy_function = None):
        self.debugging_mode = debugging_mode
        self.similarity_metric = similarity_metric
        
        if analogy_function is None:
            self.analogy_function = default_analogy_function
        else:
            self.analogy_function = analogy_function
            
        self.top_k = top_k
        if debugging_mode:
            print('SemanticAnalogies model initialized')

    def train(self, vocab, data, W, top_k, analogy_function = default_analogy_function):
       
        split_size = 100

        correct_sem = 0; 
        count_sem = 0; 

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),top_k))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))
            predictions[subset] = analogy_function(W[ind1[subset], :], W[ind2[subset], :], W[ind3[subset], :], ind1[subset], ind2[subset], ind3[subset],  W, top_k)

        max_val = np.zeros(0) # correct predictions
        for pred_index in range(top_k):
            val = (ind4 == predictions[:,pred_index]) 
            if sum(val)>sum(max_val):
                max_val = val

        count_sem = count_sem + len(ind1)
        correct_sem = correct_sem + sum(max_val)

        num_right_answers = np.sum(max_val)
        num_tot_answers = len(max_val)
        accuracy = np.mean(max_val) * 100

        if num_tot_answers == 0 :
            if self.debugging_mode:
                print('SemanticAnalogies : No data to check')
        else:
            if self.debugging_mode:
                print('SemanticAnalogies : ACCURACY TOP %d: %.2f%% (%d/%d)' % (top_k, accuracy, num_right_answers, num_tot_answers))
        
        return {'task_name':'Semantic Analogies', 'top_k_value':top_k, 'right_answers':num_right_answers, 'tot_answers':num_tot_answers, 'accuracy':accuracy}
