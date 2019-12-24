from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import tree
import numpy as np
from evaluation_framework.abstract_model import AbstractModel

float_precision = 15
    
"""
Model of the classification task
"""
class ClassificationModel(AbstractModel):
    """
    It initialize the model of the classification task
    
    task_name: name of the task
    modelName: name of the Classificator to train
    debugging_mode: {TRUE, FALSE}, TRUE to run the model by reporting all the errors and information; FALSE otherwise
    C_value: it is read only if the modelName is SVM
    """
    def __init__(self, task_name, modelName, debugging_mode, C_value=None):
        self.name = modelName
        self.configuration = None
        self.debugging_mode = debugging_mode
        self.task_name = task_name

        #create the model
        if modelName == "NB":
            self.model = GaussianNB()
        elif modelName == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.configuration = "K=3"
        elif modelName == "SVM":
            if C_value is None:
                raise Exception("For SVM, the C value has to be specified. The default used to be 1.0")
            self.model = SVC(C=C_value)
            self.configuration = "C=" + str(C_value)
        elif modelName == "C45":
            self.model = tree.DecisionTreeClassifier()
        else:
            print("YOU CHOSE WRONG MODEL FOR CLASSIFICATION!")
            
        if self.debugging_mode:
            print('Classification model initialized')
    
    """
    It trains the model based on the provided data
    
    data: dataframe with entity name as first column, class label as second column and the vectors starting from the third column
    
    It returns the result object reporting the task name, the model name and its configuration - if any -, and the accuracy as evaluation metric.
    """
    def train(self, data):
        if self.debugging_mode:
            print("Classification training...")
        scoring = 'accuracy'
        n_splits = 10
        n_samples = data.shape[0]
        if n_splits > n_samples:
            raise ValueError(
                ("Classification : Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: {1}.").format(n_splits, n_samples) + "\n")
            
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["label"], cv=n_splits, scoring=scoring)
        scoring_value = np.mean(scores)
        if self.debugging_mode:
            print('Classification', self.name, self.configuration, scoring, scoring_value)
        return {'task_name':self.task_name, 'model_name':self.name, 'model_configuration':self.configuration, scoring:round(scoring_value, float_precision)}          