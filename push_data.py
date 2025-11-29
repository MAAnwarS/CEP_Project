import os
import sys
import json
import pymongo
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import certifi
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException


load_dotenv()
ca = certifi.where()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
## print(MONGO_DB_URL)


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_to_mongodb(self, database, collection, records):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(records)
            
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

if __name__ == "__main__":
    FILE_PATH = "Network_data\phisingData.csv"
    DATABASE = "AbdullahAI"
    collection = "NetworkData"
    network_obj = NetworkDataExtract()
    records = network_obj.csv_to_json_converter(file_path=FILE_PATH)
    print(records)
    no_of_records = network_obj.insert_data_to_mongodb(records=records, database=DATABASE, collection=collection)
    print(no_of_records)