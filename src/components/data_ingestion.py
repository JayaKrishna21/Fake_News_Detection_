

# print(sys.path)

#sys.path.append('D:/Codes/PIMA diabetes - End to End ML Proj/src')
# os.chdir('..')
# from src.utils import logging
import sys
import os
from src.logger import logging
from src.exception import CustomException
from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#from src.components.data_loader import get_combined_data

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass

class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv') # os.path.join(path where the file needs to be stored, the file that needs to be stored)
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            data_set_name = "ErfanMoosaviMonazzah/fake-news-detection-dataset-English"
            
            df = load_dataset(data_set_name)
            
            #df = pd.DataFrame(df)
            logging.info('Read the dataset as dataframe from Hugging Face and using only training dataset ')
            #os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_set = pd.DataFrame(df['train'])
            test_set = pd.DataFrame(df['test'])

            # Merge 'title' and 'text' into a single 'news' column
            train_set["news"] = train_set["title"] + " " + train_set["text"]
            test_set["news"] = test_set["title"] + " " + test_set["text"]

            # Keep only the necessary columns
            train_set = train_set[['news', 'label']]
            test_set = test_set[['news', 'label']]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("✅ Train and test datasets saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":

    #Data Ingestion
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()


    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data) # train,test datasets are transformed into arrays using coulmn transformer

    #Model Training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    # giving the model : transformed train,test arrays to get better accuracy    