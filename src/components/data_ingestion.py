

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
'''
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformatidironConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
'''

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

            logging.info("Train Test Split initiated")

            train_set = pd.DataFrame(df['train'])

            test_set = pd.DataFrame(df['test'])

            

            #train_set,test_set = train_test_split(df,test_size = 0.2 , random_state = 65)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Train Dataset loaded from HuggingFace")

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Test Dataset loaded from HuggingFace")

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

    