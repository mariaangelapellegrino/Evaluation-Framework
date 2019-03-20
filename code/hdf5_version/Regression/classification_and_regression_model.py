from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import numpy as np


class AbstractModel:
    def __init__(self, modelName):
        self.name = modelName
        self.configuration = None

    
class ClassificationModel (AbstractModel):
    def __init__(self, modelName, C_value=None):
        AbstractModel.__init__(self, modelName)
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
    
    def train(self,data):
        print("training...")
        scoring = 'accuracy'
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["label"], cv=10, scoring=scoring)
        scoring_value = np.mean(scores)
        taskName = "Classification"
        print(taskName, self.name, self.configuration, scoring, scoring_value)
        return {'task_name':taskName, 'model_name':self.name, 'model_configuration':self.configuration, 'score_type':scoring, 'score_value':scoring_value}          


class RegressionModel (AbstractModel):
    def __init__(self, modelName, C_value=None):
        AbstractModel.__init__(self, modelName)
        #create the model
        if modelName == "LR":
            self.model = linear_model.LinearRegression()
        elif modelName == "M5":
            self.model = tree.DecisionTreeRegressor()
        elif modelName == "KNN":
            self.model = KNeighborsRegressor(n_neighbors=3)
            self.configuration = "K=3"
        else:
            print("YOU CHOSE WRONG MODEL FOR REGRESSION!")
    
    def train(self,data):
        print("training...")
        scoring="neg_mean_squared_error"
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["label"], cv=10, scoring=scoring)
        scoring = "root mean squared error"
        scoring_value = np.mean(np.sqrt(np.abs(scores)))
        taskName = "Regression"
        print(taskName, self.name, self.configuration, scoring, scoring_value)

        return {'task_name':taskName, 'model_name':self.name, 'model_configuration':self.configuration, 'score_type':scoring, 'score_value':scoring_value}          
