import logging
import datetime,os,sys
import pickle
import pandas as pd
import joblib
from app_tracking.logger import App_Logger


class FileOperation:
    def __init__(self):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.filepath = f"artifacts/logs/FileOperation/{current_time}.txt"
        self.logging = App_Logger(self.filepath)

    # Function to save a machine learning model to a file
    def save_model(self,model, file_path):
        try:
            joblib.dump(model, file_path)
            self.logging.log(f"Model saved to file: {file_path}")

        except Exception as e:
            self.logging.log(f"Error saving the model: {str(e)}")
            raise e

    # Function to load a machine learning model from a file
    def load_model(self,file_path):
        try:
            model = joblib.load(file_path)
            self.logging.log(f"Model loaded from file: {file_path}")
            return model
        
        except Exception as e:
            self.logging.log(f"Error loading the model: {str(e)}")
            raise e

    # Function to save data (pandas DataFrame) to a CSV file
    def save_data_to_csv(self,data, file_path):
        try:
            data.to_csv(file_path, index=False)
            self.logging.log(f"Data saved to CSV file: {file_path}")

        except Exception as e:
            self.logging.log(f"Error saving data to CSV file: {str(e)}")
            raise e

    # Function to load data from a CSV file into a pandas DataFrame
    def load_data_from_csv(self,file_path):
        try:
            data = pd.read_csv(file_path)
            self.logging.log(f"Data loaded from CSV file: {file_path}")
            return data
        
        except Exception as e:
            self.logging.log(f"Error loading data from CSV file: {str(e)}")
            raise e

    # Function to save data to a binary file using pickle
    def save_data_to_pickle(self,data, file_path):
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
            self.logging.log(f"Data saved to pickle file: {file_path}")

        except Exception as e:
            self.logging.log(f"Error saving data to pickle file: {str(e)}")
            raise e

    # Function to load data from a binary file using pickle
    def load_data_from_pickle(self,file_path):
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            self.logging.log(f"Data loaded from pickle file: {file_path}")
            return data
        
        except Exception as e:
            self.logging.log(f"Error loading data from pickle file: {str(e)}")
            raise e
        

    def delete_files_in_directory(self,directory_path):
        self.logging.log("Get a list of all files in the directory")
        file_list = os.listdir(directory_path)
        self.logging.log("Iterate over each file and delete it")
        for filename in file_list:
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    self.logging.log(f"Deleted {file_path}")
                else:
                    self.logging.log(f"Skipping {file_path} as it is not a file.")
            except Exception as e:
                self.logging.log(f"Failed to delete {file_path}. Error: {e}")


