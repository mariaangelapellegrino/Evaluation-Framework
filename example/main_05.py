# this example specifies the parameters by a xml file and the "custom" analogy function (which is the same of the default value). It uses default values for all the other parameters

import numpy as np
from evaluation_framework.manager import FrameworkManager

def default_analogy_function(a, b, c):
    return np.array(b) - np.array(a) + np.array(c)

if __name__ == "__main__":
    evaluation_manager = FrameworkManager()
    parameters_dict = evaluation_manager.get_parameters_xmlFile('parameters.xml')
    evaluation_manager.evaluate(parameters_dict['vector_filename'], 
                                vector_file_format = parameters_dict['vector_file_format'], 
                                vector_size = parameters_dict['vector_size'], 
                                parallel = parameters_dict['parallel'], 
                                tasks = parameters_dict['tasks'], 
                                similarity_metric = parameters_dict['similarity_function'], 
                                analogy_function= default_analogy_function, 
                                top_k = parameters_dict['top_k'], 
                                compare_with = parameters_dict['compare_with'], 
                                debugging_mode = parameters_dict['debugging_mode'])
