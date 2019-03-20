# this example specifies the parameters by a xml file

import argparse
import numpy as np

import sys
sys.path.append("../code/")
from manager import EvaluationManager

if __name__ == "__main__":
    evaluation_manager = EvaluationManager()
    parameters_dict = evaluation_manager.get_parameters_xmlFile('parameters.xml')
    evaluation_manager.evaluate(parameters_dict)