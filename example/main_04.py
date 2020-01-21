# this example specifies the parameters by a xml file

from evaluation_framework.manager import FrameworkManager

if __name__ == "__main__":
    evaluation_manager = FrameworkManager()
    parameters_dict = evaluation_manager.get_parameters_xmlFile('parameters.xml')
    evaluation_manager.evaluate(parameters_dict['vector_filename'], 
                                vector_file_format = parameters_dict['vector_file_format'], 
                                vector_size = parameters_dict['vector_size'], 
                                parallel = parameters_dict['parallel'], 
                                tasks = parameters_dict['tasks'], 
                                similarity_metric = parameters_dict['similarity_function'],  
                                top_k = parameters_dict['top_k'], 
                                compare_with = parameters_dict['compare_with'], 
                                debugging_mode = parameters_dict['debugging_mode'])