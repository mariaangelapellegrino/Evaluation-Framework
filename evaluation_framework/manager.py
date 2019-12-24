import xml.etree.ElementTree as ET

from evaluation_framework.evaluationManager import EvaluationManager
from evaluation_framework.txt_dataManager import DataManager as TxtDataManager
from evaluation_framework.hdf5_dataManager import DataManager as Hdf5DataManager

available_tasks = ['Classification', 'Regression', 'Clustering', 
    'DocumentSimilarity', 'EntityRelatedness', 'SemanticAnalogies']
available_file_formats = ['txt', 'hdf5']

"""
It checks the parameters of the evaluation and starts it.
"""
class FrameworkManager():
    def __init__(self):
        print('Start evaluation...')

    """
    It checks the parameters of the evaluation and starts it.
    
    vector_filename: path of the vector file provided in input
    vector_file_format: {txt, hdf5}. Default: txt
    vector_size: size of the vectors. Default: 200
    parallel: {True, False}, True to run the tasks in parallel, False otherwise. Default: False
    tasks: list of the tasks to run
    similarity_metric: metric used to compute the distance among vectors. Default: cosine
    top_k: parameter used in the SemanticAnalogies task. Default: 2
    compare_with: list of the technique to compare the results with. Default: _all
    debugging_mode: {True, False}, True to run the tasks by reporting all the information collected during the run, False otherwise. Default: False
    analogy_function: function to compute the analogy among vectors. Default: None to use the default function.
    """
    def evaluate(self, vector_filename, vector_file_format='txt', vector_size=200, 
        parallel=False, tasks=available_tasks, similarity_metric='cosine', top_k=2, 
        compare_with='_all', debugging_mode=False, analogy_function=None ):

        self.vector_filename = vector_filename
        self.vector_file_format = vector_file_format
        self.vector_size = vector_size
        self.parallel = parallel
        self.tasks = tasks
        self.similarity_metric = similarity_metric
        self.analogy_function = analogy_function
        self.top_k = top_k
        self.compare_with = compare_with
        self.debugging_mode = debugging_mode

        self.check_parameters()

        if vector_file_format == 'txt':
            self.dataManager = TxtDataManager(self.debugging_mode)
        elif vector_file_format == 'hdf5':
            self.dataManager = Hdf5DataManager(self.debugging_mode)
                    
        self.evaluation_manager = EvaluationManager(self.dataManager, self.debugging_mode)

        self.evaluation_manager.create_result_directory()

        self.evaluation_manager.initialize_vectors(vector_filename, vector_size)

        if parallel:
            scores_dictionary = self.evaluation_manager.run_tests_in_parallel(tasks, similarity_metric, top_k, analogy_function)
        else:
            scores_dictionary = self.evaluation_manager.run_tests_in_sequential(tasks, similarity_metric, top_k, analogy_function)        
        
        self.evaluation_manager.compare_with(compare_with, scores_dictionary)
    
    """
    It checks if the parameters are all valid. 
    If no problem occurs, the evaluation will start.
    """
    def check_parameters(self):
        if self.vector_filename==None:
            raise Exception('The vector filename is a mandatory parameter.')

        if not self.vector_file_format in available_file_formats:
            raise Exception('Not supported file format. The managed file format are: ' + available_file_formats)
        
        if self.vector_size < 0:
            raise Exception('The vector size must be not negative.')
        
        if self.parallel!=True and self.parallel!=False:
            raise Exception('The parameter PARALLEL is boolean.')
        
        if self.tasks!='_all':
            for task in self.tasks:
                if not task in available_tasks:
                    raise Exception(task + ' is not a supported task. The managed tasks are ' + ', '.join(available_tasks) + ' or \'_all\'.')
        
        #similarity_metric TODO
        if self.top_k < 0:
            raise Exception('The top_k value must be not negative.')
        
        #compare_with TODO

        if self.debugging_mode!=True and self.debugging_mode!=False:
            raise Exception('The parameter DEBUGGING_MODE is boolean.') 

    """
    It reads a xml_file and it recovers the parameters which can be passed in input to the evaluate() function.
    It returns a dictionary containing all the read parameters.
    
    xml_file: path of the xml file.
    """
    def get_parameters_xmlFile(self, xml_file):
        parameters_dict = {}
        
        tree = ET.parse(xml_file)
        root = tree.getroot()

        string_tags = ['vector_filename', 'vector_file_format', 'similarity_function']

        for tag in string_tags:
            actual_tag = root.find(tag)
            if not actual_tag is None:
                parameters_dict[tag] = actual_tag.text
                
        int_tags = ['vector_size', 'top_k']

        for tag in int_tags:
            actual_tag = root.find(tag)
            if not actual_tag is None:
                parameters_dict[tag] = int(actual_tag.text)
                
        boolean_tags = ['parallel', 'debugging_mode']

        for tag in boolean_tags:
            actual_tag = root.find(tag)
            if not actual_tag is None:
                parameters_dict[tag] = bool(actual_tag.text)

        tags = ['tasks', 'compare_with']
        for tag in tags:
            tag_values_list = []
            actual_tag_list = root.find(tag)
            if not actual_tag_list is None:
                for actual_tag in actual_tag_list.findall('value'):
                    tag_values_list.append(actual_tag.text)
                parameters_dict[tag] = tag_values_list
                
        

        return parameters_dict