import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            logging.info("Initializing DataTransformation class.")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls):
        try:
            logging.info("Creating KNNImputer and building pipeline.")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor = Pipeline([("imputer", imputer)])
            logging.info("Data transformer (pipeline) created successfully.")
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process.")

            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Successfully read train and test datasets.")

            # Separating input features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            logging.info("Separated input and target features. Target -1 values replaced with 0.")

            # Preprocessing
            preprocessor_object = self.get_data_transformer_object()
            logging.info("Fitting and transforming training data.")
            transformed_input_feature_train_df = preprocessor_object.fit_transform(input_feature_train_df)

            logging.info("Transforming testing data.")
            transformed_input_feature_test_df = preprocessor_object.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[transformed_input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test_df, np.array(target_feature_test_df)]

            logging.info("Combined transformed features with target arrays.")

            # Save the numpy arrays and preprocessor object
            save_numpy_array_data(self.data_transformation_config.transformed_trained_file_path, array=train_arr)
            logging.info(f"Saved transformed training array to {self.data_transformation_config.transformed_trained_file_path}")

            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info(f"Saved transformed test array to {self.data_transformation_config.transformed_test_file_path}")

            save_object(self.data_transformation_config.transformed_object_file_path, object=preprocessor_object)
            logging.info(f"Saved preprocessor object to {self.data_transformation_config.transformed_object_file_path}")

            save_object("final_models/preprocessor.pkl", preprocessor_object)

            # Creating artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_trained_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            logging.info("DataTransformationArtifact created successfully.")
            return data_transformation_artifact

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise NetworkSecurityException(e, sys)
