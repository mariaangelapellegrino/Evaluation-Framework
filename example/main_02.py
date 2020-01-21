# this example specifies only the vector file. It uses default values for all the other parameters

import argparse

from evaluation_framework.manager import FrameworkManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation framework for RDF embedding methods')
    parser.add_argument('--vector_file', type=str, required=True, help='Path of the file where your vectors are stored. File format: one line for each entity with entity and vector')
    args = parser.parse_args()
    evaluation_manager = FrameworkManager()
    evaluation_manager.evaluate(args.vector_file)
