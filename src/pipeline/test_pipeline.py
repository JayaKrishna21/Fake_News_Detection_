import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # to load pickle file
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features): 
        try:
            
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:

    
    # responsible to map the input values to pickle file

    def __init__(self,
                title:str,
                news:str):
        
        self.title = title

        self.news = news


    def get_data_as_df(self):
        #mapping data to respective keys

        try:
            custom_data_input_dict = {
                "news": [self.title + " " + self.news]
            }
            return pd.DataFrame(custom_data_input_dict)
         
        except Exception as e:
            raise CustomException(e,sys)

