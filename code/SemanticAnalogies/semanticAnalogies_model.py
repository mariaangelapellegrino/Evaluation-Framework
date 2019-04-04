import numpy as np

from code.abstract_model import AbstractModel

float_precision = 15

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

    def train(self, vocab, data, W):
        correct_sem = 0; 
        count_sem = 0; 

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices), self.top_k))
      
        for j in range(len(indices)):
            pred_vec = self.analogy_function(W[ind1[j], :], W[ind2[j], :], W[ind3[j], :])
            dist = np.dot(W, pred_vec.T)

            dist[ind1[j]] = -np.Inf
            dist[ind2[j]] = -np.Inf
            dist[ind3[j]] = -np.Inf

            predictions[j] = np.argsort(-dist, axis=0)[:self.top_k].T
            
        max_val = np.zeros(0) # correct predictions
        for pred_index in range(self.top_k):
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
                print('SemanticAnalogies : ACCURACY TOP %d: %.2f%% (%d/%d)' % (self.top_k, accuracy, num_right_answers, num_tot_answers))
        
        return {'task_name':'Semantic Analogies', 'top_k_value':self.top_k, 'right_answers':num_right_answers, 'tot_answers':num_tot_answers, 'accuracy':round(accuracy, float_precision)}
