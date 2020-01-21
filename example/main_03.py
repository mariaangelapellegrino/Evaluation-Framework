# this example specifies only the vector file and the "custom" analogy function (which is the same of the default value). It uses default values for all the other parameters

import argparse
import numpy as np

from evaluation_framework.manager import FrameworkManager

def default_analogy_function(a, b, c):
    return np.array(b) - np.array(a) + np.array(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation framework for RDF embedding methods')
    parser.add_argument('--vectors_file', type=str, required=True, help='Path of the file where your vectors are stored. File format: one line for each entity with entity and vector')
    args = parser.parse_args()
    evaluation_manager = FrameworkManager()
    evaluation_manager.evaluate(vector_file=args.vectors_file)
