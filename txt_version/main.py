import argparse
from evaluation_manager import EvaluationManager
import numpy as np

def default_analogy_function(a, b, c, index_a, index_b, index_c, data, top_k):
    pred_vec = np.array(b) - np.array(a) + np.array(c)

    dist = np.dot(data, pred_vec.T)

    for k in range(len(a)):
        dist[index_a[k], k] = -np.Inf
        dist[index_b[k], k] = -np.Inf
        dist[index_c[k], k] = -np.Inf

    return np.argsort(-dist, axis=0)[:top_k].T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation framework for RDF embedding methods')
    parser.add_argument('--vectors_file', type=str, required=True, help='Path of the file where your vectors are stored. File format: one line for each entity with entity and vector')
    parser.add_argument('--vectors_size', default=200, type=int, help='Length of each vector. Default : 200')
    parser.add_argument('--top_k', default=2, type=int, help='Used in SemanticAnalogies : The predicted vector will be compared with the top k closest vectors to establish if the prediction is correct or not. Default : 2')
    args = parser.parse_args()
    evaluation_manager = EvaluationManager()
    evaluation_manager.evaluate(args.vectors_file, args.vectors_size, 'cosine', default_analogy_function, args.top_k)
