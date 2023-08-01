import pandas as pd
import datetime
import sys
import os
import contextlib
import csv
import requests
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from Utils.utils import FileOperation

class Data_Getter:
    """
    This class shall be used for obtaining the data from the source for training.
    Written By: Rachit More
    Version: 1.0
    Revisions: None
    """
    def __init__(self):
        try:
            print("Data Ingestion has been started")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            self.filepath = f"artifacts/logs/Stage1_Dataingestion/{current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileOperation = FileOperation()
            self.fileOperation.delete_files_in_directory("artifacts/data")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

    # Data ingestion from specific path in csv file
    def ingest_from_csv(self,path = "artifacts/data/raw/adult.csv"):
        try:
            self.logging.log("Data ingestion started from specific path in csv")
            df = pd.read_csv(path)
            self.fileOperation.save_data_to_csv(df,"artifacts/data/raw.csv")
            self.logging.log("Data ingested successfully from specific path in csv")

        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

    # Data ingestion from url
    def ingest_from_url(self, url):
        try:
            self.logging.log("Data ingestion started from url")
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                self.fileOperation.save_data_to_csv(df,"artifacts/data/raw.csv")
                self.logging.log("Data ingested successfully from url")
                return None
            else:
                self.logging.log(f"Failed to fetch data from the API. Status code: {response.status_code}")
                return None

        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        
    def ingest_from_database(self):
        self.logging.log("Data ingestion started")
        self.logging.log("Connecting to the database server")
        try:
            cloud_config= {
            'secure_connect_bundle': 'artifacts\secure-connect-census.zip'
            }
            auth_provider = PlainTextAuthProvider("ZzLDROBvUeUaemzwgNdxQoET", 
                            "sO5Zpe0hbMeqCIkjtofPS49eRbF4nhjLnDh0oY3w1CQIboZGqCMyKQFZGjs4xaclAUwQYLd7,Kmk4DPLGLy+iIlYDODQy6Zw9KUn+EW7qK.uo4J0,iR9MFma-7Y.Z5h2")
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect()
            self.logging.log("Database server has been connected")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

        # Query to fetch the top 100 records
        query = "SELECT * FROM census.adult;"

        self.logging.log("Executing the query")
        try:
            data = session.execute(query)
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

        try:
            csv_file = 'artifacts/data/raw.csv'
            self.logging.log("Writing the results to the CSV file")
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the header (column names)
                writer.writerow(data.column_names)
                # Write each row of data
                for row in data:
                    writer.writerow(row)
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        session.shutdown()
        self.logging.log("CSV file has been written. and data saved to the artifacts/data/raw.csv")

    