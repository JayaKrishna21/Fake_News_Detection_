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
    
    def text_cleaning(self,text):
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Remove stopwords
        words = [word for word in text.split() if word not in self.stop_words]
        
        # Apply stemming
        cleaned_text = ' '.join([self.stemmer.stem(word) for word in words])
        
        return cleaned_text

        
    

    def get_data_transformer_object(self):

        # Responsible for Data Transformation ( Standardisation / Encoding )
        try:
            
            '''
            cat_col = ['subject']

            output = ['label']

            
            train_df['news'] = (train_df['title'] + ' ' + train_df['text']).apply(self.text_cleaning)
            test_df['news'] = (test_df['title'] + ' ' + test_df['text']).apply(self.text_cleaning)

            logging.info("Text Cleaning is done")




            text_col = ['news']       
            '''

            text_pipeline = Pipeline(
                steps=[
                    
                    ("Vectorising",TfidfVectorizer(preprocessor=self.text_cleaning))

                ]
            )
            logging.info("TF-IDF applied")

            cat_pipeline = Pipeline(
                 steps=[
                     ("Imputer",SimpleImputer(strategy="most_frequent")), 
                     ("One_Hot_Encoder",OneHotEncoder()),
                     ("Scaler",StandardScaler(with_mean=False))

                 ]
            )
            logging.info("Subject column is treated with One Hot Encoding")

            logging.info("Subject column Standardisation completed")
            

            # logging.info("Categorical columns OneHotEncoding completed")

            preprocessor = ColumnTransformer(
                [("Text_pipeline", text_pipeline, "news"),
                 ("Cat_pipeline", cat_pipeline, ['subject'])]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    



    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path) #Data Transformation is done for splitted train,test datasets in artifacts

            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            train_df['news'] = train_df['title'] + " " + train_df['text']
            test_df['news'] = test_df['title'] + " " + test_df['text']

            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
            test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

            train_df = train_df.drop(columns=['date', 'title', 'text'])
            test_df = test_df.drop(columns=['date', 'title', 'text'])

            

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "label"

            
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) #fit_transform for train dataset
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) #transform for test data
            
            

            input_feature_train_arr = input_feature_train_arr
            input_feature_test_arr = input_feature_test_arr

            # Reshape target labels to a column vector
            target_feature_train_df_reshaped = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_df_reshaped = target_feature_test_df.values.reshape(-1, 1)

            target_feature_train_sparse = scipy.sparse.csr_matrix(target_feature_train_df_reshaped)
            target_feature_test_sparse = scipy.sparse.csr_matrix(target_feature_test_df_reshaped)

            # Concatenate sparse feature matrix with the target labels
            train_arr_sparse = scipy.sparse.hstack([input_feature_train_arr, target_feature_train_sparse])
            test_arr_sparse = scipy.sparse.hstack([input_feature_test_arr, target_feature_test_sparse])

            logging.info(f"Saved preprocessing object.")

            save_object(
                # used to save pickle file
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr_sparse,
                test_arr_sparse,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)
    

    