import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

# Create Data Ingestion Class 

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Process Started')

        try:

            # Read data from notebook folder
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info("Data read as Dataframe")

            # Create artifacts folder if not already
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)

            # Saving raw data to artifacts   
            df.to_csv(self.ingestion_config.raw_data_path, header= True, index= False)

            logging.info('Raw Data File saved ib artifacts folder')

            # Train test split

            logging.info('Train test split started')

            train_set, test_set = train_test_split(df, test_size= 0.30, random_state= 42)

            train_set.to_csv(self.ingestion_config.train_data_path,header = True, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path,header = True, index = False)

            logging.info('Data Ingestion Complete')

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info("Error occured in data ingestion")
            raise CustomException(e,sys)