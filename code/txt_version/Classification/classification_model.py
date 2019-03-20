from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import tree
import numpy as np

from code.abstract_model import AbstractModel
    
class ClassificationModel(AbstractModel):
    def __init__(self, modelName, C_value=None, debugging_mode=False):
        self.name = modelName
        self.configuration = None
        self.debugging_mode = debugging_mode

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
        return {'task_name':'Classification', 'model_name':self.name, 'model_configuration':self.configuration, 'score_type':scoring, 'score_value':scoring_value}          