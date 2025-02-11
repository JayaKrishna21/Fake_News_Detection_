import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test,y_test, models, param):
    try:
        report = {}
        
        model = list(models.values())[0]

        #Setting parameters
        para = param[list(models.keys())[0]]


        
        grid_search = GridSearchCV(model,param_grid= para, cv =5, verbose = True)
        grid_search.fit(X_train, y_train)

        model.set_params(**grid_search.best_params_)
            


        model.fit(X_train,y_train) # Training the model

        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)

        train_model_score = r2_score(y_train,y_train_pred)

        test_model_score = r2_score(y_test,y_test_pred)

        report[list(models.keys())[0]] = (test_model_score) # entering test data score into reports
        return report
        

    except Exception as e:
        raise CustomException(e,sys)