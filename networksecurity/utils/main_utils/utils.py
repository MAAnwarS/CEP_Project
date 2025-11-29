import os, sys
import yaml
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path) -> dict:
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def write_yaml_file(file_path, content, replace=False) -> None:
    try:
        if replace:
            if os.path.exist(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)


def save_object(file_path, object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(object, file) 
    except Exception as e:
        raise NetworkSecurityException(e, sys)

    

def save_numpy_array_data(file_path, array) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array) 
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The filepath: {file_path} does not exist")
        
        with open(file_path, "rb") as file:
            print(file)
            return pickle.load(file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)


def load_numpy_array_data(file_path):
    try:
        with open(file_path, "rb") as file:
            return np.load(file)   # Correct order
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = models[model_name]
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score


        return report



    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

## Alternate
# def evaluate_models(X_train, y_train, X_test, y_test, models, params):
#     try:
#         result ={}

#         for model_name in models:
#             model = models[model_name]
#             param = params[model_name]

#             # Tune model using GridSearchCV
#             grid_search = GridSearchCV(model, param, cv=3)
#             grid_search.fit(X_train, y_train)

#             # Update model with best parameters
#             best_model = grid_search.best_estimator_

#             # Train the best model
#             best_model.fit(X_train, y_train)

#             # Make predictions
#             train_pred = best_model.predict(X_train)
#             test_pred = best_model.predict(X_test)

#             # Calculate R2 scores
#             train_score = r2_score(y_train, train_pred)
#             test_score = r2_score(y_test, test_pred)

#             # Save test score
#             result[model_name] = test_score

#         return result

#     except Exception as e:
#         raise NetworkSecurityException(e,sys)