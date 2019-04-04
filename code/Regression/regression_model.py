from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from code.abstract_model import AbstractModel

float_precision = 15

class RegressionModel (AbstractModel):
    def __init__(self, modelName, debugging_mode):
        self.name = modelName
        self.configuration = None
        self.debugging_mode = debugging_mode
        
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
            
        if self.debugging_mode:
            print('Regression model initialized')
    
    def train(self, data):
        if self.debugging_mode:
            print("Regression training...")
        scoring="neg_mean_squared_error"
        n_splits = 10
        n_samples = data.shape[0]
        if n_splits > n_samples:
            raise ValueError(
                ("Regression : Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: {1}.").format(n_splits, n_samples) + "\n")
            
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["label"], cv=n_splits, scoring=scoring)
        scoring = "root_mean_squared_error"
        scoring_value = np.mean(np.sqrt(np.abs(scores)))
        if self.debugging_mode:
            print(self.name, self.configuration, scoring, scoring_value)

        return {'task_name':'Regression', 'model_name':self.name, 'model_configuration':self.configuration, scoring:round(scoring_value, float_precision)}          
