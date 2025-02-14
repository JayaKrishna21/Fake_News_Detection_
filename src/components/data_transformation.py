import sys
from dataclasses import dataclass
import re
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

import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
from nltk import PorterStemmer
import scipy.sparse

nltk.download('stopwords')
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))        
        self.stemmer = PorterStemmer()
        self.data_transformation_config = DataTransformationConfig()

        
    '''    
    def text_cleaning(self,text):
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Remove stopwords
        words = [word for word in text.split() if word not in self.stop_words]
        
        # Apply stemming
        cleaned_text = ' '.join([self.stemmer.stem(word) for word in words])
        
        return cleaned_text
    '''

    def get_data_transformer_object(self):

        # Responsible for Data Transformation ( Standardisation / Encoding )
        try:
            
            
            text_pipeline = Pipeline(
                steps=[
                    
                    ("Vectorising",TfidfVectorizer(stop_words="english",lowercase=True))

                ]
            )
            logging.info("TF-IDF applied")


            
            return text_pipeline
        

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path) #Data Transformation is done for splitted train,test datasets in artifacts

            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            # Ensure subject column does not exist
            
            if "subject" in train_df.columns:
                train_df = train_df.drop(columns=["subject"])
            if "subject" in test_df.columns:
                test_df = test_df.drop(columns=["subject"])

            target_column_name = "label"
            # Separate features and target
            X_train = train_df["news"]
            y_train = train_df[target_column_name]

            X_test = test_df["news"]
            y_test = test_df[target_column_name]

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            # Transform data
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Convert labels to sparse matrices
            y_train_sparse = scipy.sparse.csr_matrix(y_train.values.reshape(-1, 1))
            y_test_sparse = scipy.sparse.csr_matrix(y_test.values.reshape(-1, 1))

            # Combine transformed inputs with target labels
            train_arr_sparse = scipy.sparse.hstack([X_train_transformed, y_train_sparse])
            test_arr_sparse = scipy.sparse.hstack([X_test_transformed, y_test_sparse])

            logging.info("✅ Data transformation completed and saved")

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

        
            logging.info(f"Saved preprocessing object.")


            return(
                train_arr_sparse,
                test_arr_sparse,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)
    

    