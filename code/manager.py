from xml.dom import minidom

from txt_version.txt_evaluationManager import EvaluationManager as TxtEvaluationManager
#from hdf5_version.evaluation_manager import EvaluationManager as Hdf5EvaluationManager

available_tasks = ['Classification', 'Regression', 'Clustering', 
    'DocumentSimilarity', 'EntityRelatedness', 'SemanticAnalogies']
available_file_formats = ['txt', 'hdf5']

class EvaluationManager():
    def __init__(self):
        print('Start evaluation...')

    def evaluate(self, vector_filename, vector_file_format='txt', vector_size=200, 
        parallel=False, tasks=available_tasks, similarity_metric='cosine', 
        analogy_function=None, top_k=2, 
        compare_with='_all', debugging_mode=False ):

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
            self.evaluation_manager = TxtEvaluationManager(debugging_mode = debugging_mode)
        #elif vector_file_format == 'hdf5':
        #    self.evaluation_manager = Hdf5EvaluationManager(debugging_mode = debugging_mode)

        self.evaluation_manager.create_result_directory()

        self.evaluation_manager.initialize_vectors(vector_filename, vector_size)

        if parallel:
            self.evaluation_manager.run_tests_in_parallel(tasks=tasks, similarity_metric=similarity_metric, analogy_function=analogy_function, top_k=top_k)
        else:
            self.evaluation_manager.run_tests_in_sequential(tasks=tasks, similarity_metric=similarity_metric, analogy_function=analogy_function, top_k=top_k)

        self.evaluation_manager.compare_with(compare_with)
    
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
                    raise Exception(task + ' is not a supported task. The managed tasks are ' + available_tasks + ' or \'_all\'.')
        
        #similarity_metric TODO
        if self.top_k < 0:
            raise Exception('The top_k value must be not negative.')
        
        #compare_with TODO

        if self.debugging_mode!=True and self.debugging_mode!=False:
            raise Exception('The parameter DEBUGGING_MODE is boolean.') 

    def get_parameters_xmlFile(self, xml_file):
        parameters_dict = {}

        # parse an xml file by name
        mydoc = minidom.parse(xml_file)

        parameters = mydoc.getElementsByTagName('parameters')
        tags = ['vector_filename', 'vector_file_format', 'vector_size', 
            'similarity_function', 'top_k',
            'parallel', 'debugging_mode']

        for tag in tags:
            actual_tag = parameters.find(tag)
            if not actual_tag is None:
                parameters_dict[tag] = actual_tag.value

        tags = ['tasks', 'compare_with']
        tag_values_list = []
        for tag in tags:
            actual_tag_list = parameters.find(tag)
            if not actual_tag_list is None:
                for actual_tag in actual_tag_list.getElementsByTagName('value'):
                    tag_values_list.append(actual_tag.value)
                parameters_dict[tag] = tag_values_list

        return parameters_dict