import numpy as np

class Model:
    def __init__(self):
        print('SemanticAnalogies model initialized')

    @staticmethod
    def normalize_vectors(vectors, vec_size, vocab):
        W = np.zeros((len(vectors), vec_size))
        
        for index, row in vectors.iterrows():
            W[vocab[row['name']], :] = row[1:]

        # normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm

    @staticmethod
    def compute_semantic_analogies(vocab, data, W, analogy_function, top_k):
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
            print('SemanticAnalogies : No data to check')
        else:
            print('SemanticAnalogies : ACCURACY TOP %d: %.2f%% (%d/%d)' % (top_k, accuracy, num_right_answers, num_tot_answers))
        
        return (num_right_answers, num_tot_answers, accuracy)