# this example specifies the parameters by a xml file and the "custom" analogy function (which is the same of the default value). It uses default values for all the other parameters

import argparse
import numpy as np

import sys
sys.path.append("../code/")
from manager import EvaluationManager

def default_analogy_function(a, b, c):
    return np.array(b) - np.array(a) + np.array(c)

if __name__ == "__main__":
    evaluation_manager = EvaluationManager()
    parameters_dict = evaluation_manager.get_parameters_xmlFile('parameters.xml')
    evaluation_manager.evaluate(parameters_dict, analogy_function=default_analogy_function)
