#to Log errors and execution
import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log" #Name of the log_file that needs to be stored for each time, the file is being exectued
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)  #Path where the log_file needs to be stoored at
os.makedirs(log_path,exist_ok=True) #storing in the specified path

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    
    level = logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")
