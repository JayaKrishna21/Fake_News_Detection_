import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        # Responsible for Data Transformation ( Standardisation / Encoding )
        try:
            

            # categorical_columns = []
            '''
            Creating Pipeline

            For Numerical Columns

            - Filling missing values with median
            - Standardisation

            For Categorical Columns

            - Filling missing values with mode
            - OneHot Encoding
            - Standardisation
            
            '''
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")), 
                    ("Scaler",StandardScaler())

                ]
            )

            # cat_pipeline = Pipeline(
            #     steps=[
            #         ("Imputer",SimpleImputer(strategy="most_frequent")), 
            #         ("One_Hot_Encoder",OneHotEncoder())
            #         ("Scaler",StandardScaler())

            #     ]
            # )

            logging.info("Numerical columns Standardisation completed")
            logging.info(f"Numerical columns:{numerical_columns}")

            # logging.info("Categorical columns OneHotEncoding completed")

            preprocessor = ColumnTransformer(
                [("Num_pipeline",num_pipeline,numerical_columns)
                #  ("Cat_pipeline",cat_pipeline,categorical_columns)
                 

                ]
                
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path) #Data Transformation is done for splitted train,test datasets in artifacts

            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Outcome"

            numerical_columns = ['Pregnancies', 
                'Glucose', 
                'BloodPressure', 
                'SkinThickness',
                'Insulin',
                'BMI', 
                'DiabetesPedigreeFunction',
                'Age'
            ]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) #fit_transform for train dataset
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) #transform for test data
            

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                # used to save pickle file
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)










def preprocess_text(df, text_columns):
    """Preprocess multiple text columns in the DataFrame."""
    for col in text_columns:
        # Lowercase the text
        df[col] = df[col].str.lower()
        
        # Remove punctuation
        df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # Remove extra whitespace
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    return df