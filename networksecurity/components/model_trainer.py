import os
import sys
import mlflow
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import (
    save_object, load_object, load_numpy_array_data, evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score

import dagshub
# dagshub.init(repo_owner='muhammadabdullah', repo_name='CEP_Project', mlflow=True)




class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Initializing ModelTrainer...")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("ModelTrainer initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classification_metric):
        logging.info("Starting MLflow tracking...")

        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            logging.info(f"Logging metrics to MLflow: F1 Score={f1_score}, Precision={precision_score}, Recall={recall_score}")
            mlflow.log_metric("f1 score", f1_score)
            mlflow.log_metric("precision score", precision_score)
            mlflow.log_metric("recall score", recall_score)

            logging.info("Logging model to MLflow.")
            mlflow.sklearn.log_model(best_model, "model")

        logging.info("MLflow tracking completed.")

        

    def train_model(self, x_train, y_train, x_test, y_test):
        logging.info("Starting model training with hyperparameter tuning.")

        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "KNN Classifier": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoosting": AdaBoostClassifier()
        }

        params = {
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
            "KNN Classifier": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ['uniform', 'distance'],
            },
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
            "Logistic Regression": {
                "C": [0.1, 1.0, 10.0],
                "solver": ['lbfgs', 'saga'],
                "max_iter": [100, 200, 500]
            },
            "AdaBoosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            }
        }

        logging.info("Evaluating all models with GridSearchCV...")
        model_report: dict = evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test,
                                             y_test=y_test, models=models, params=params)

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        logging.info(f"Best Model Selected: {best_model_name} with score: {best_model_score}")

        y_train_pred = best_model.predict(x_train)
        classification_train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        logging.info("Training performance metrics calculated.")

        y_test_pred = best_model.predict(x_test)
        classification_test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        logging.info("Testing performance metrics calculated.")

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        logging.info(f"Model directory created at: {model_dir_path}")

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, object=network_model)
        logging.info(f"Trained model saved to: {self.model_trainer_config.trained_model_file_path}")

        ## Tracking with MLFLOW
        self.track_mlflow(best_model=best_model, classification_metric=classification_train_metrics)
        self.track_mlflow(best_model=best_model, classification_metric=classification_test_metrics)


        save_object("final_models/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metrics,
            test_metric_artifact=classification_test_metrics
        )

        logging.info(f"Model Trainer artifact created: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer process...")

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading transformed train and test data.")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info("Starting training using train_model method.")
            model_trainer_artfiact = self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            logging.info("Model training completed successfully.")
            return model_trainer_artfiact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
