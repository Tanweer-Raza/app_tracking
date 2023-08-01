"""
This is the Entry point for Training the Machine Learning Model.
Written By: Rachit More
Version: 1.0
Revisions: None
"""
# Doing the necessary imports
import sys
import warnings
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from Utils.utils import FileOperation
from .src.Stage1_data_ingestion.data_loader import Data_Getter
from .src.Stage2_DataValidation.DataTypeValidation import RawDataValidation,PrePreocessedDataValidation
from .src.Stage3_DataPreprocessing.DataTransformation import DataPreprocessing
from .src.Stage4_model_building.model_building import ModelTraining
from .src.Stage5_model_evaluation.model_evaluation import ModelEvaluation

# Suppress the specific warning
warnings.filterwarnings("ignore" )

def train_model(DataSource):
    try:
        if DataSource == "Local":
            print("Code flow start")
            Data_Getter().ingest_from_csv()
            RawDataValidation()
            DataPreprocessing()
            PrePreocessedDataValidation()
            ModelTraining()
            ModelEvaluation()
        else:
            Data_Getter().ingest_from_database()
            RawDataValidation()
            DataPreprocessing()
            PrePreocessedDataValidation()
            ModelTraining()
            ModelEvaluation()   
        
    except Exception as e:
        raise AppException(e,sys) from e

DataSource = "Local"  
if __name__ == "__main__":
    train_model(DataSource)
    


        