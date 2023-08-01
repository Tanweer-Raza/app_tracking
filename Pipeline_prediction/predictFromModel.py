import datetime,sys
from scipy.sparse import load_npz,save_npz
import pandas as pd
import numpy as np
import contextlib
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from Utils.utils import FileOperation


class Prediction:
    def __init__(self,data):
        try:
            print("Predicting...")
            self.data = data
            self.current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            self.filepath = f"artifacts/prediction/logs/{self.current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileOperation = FileOperation()
            self.path = f"artifacts/prediction/data/{self.current_time}.csv"
            self.preprocessedDatapath = f'artifacts/data/{self.current_time}Sparse_matrix.npz'
            with contextlib.suppress():
                self.DataTransformation()
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

    def DataTransformation(self): 
        try:
            self.logging.log("Prediction Data Transformation has been started")
            column_names = ['Age', 'Workclass', 'Fnlwgt', 'EducationNum',
                                'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex',
                                'Capital_gain', 'Capital_loss', 'HoursPerWeek', 'NativeCountry',
                                ]
            
            self.num_cols = ['Age','Fnlwgt','EducationNum',
                            'Capital_gain','Capital_loss','HoursPerWeek']
            self.cat_cols = ['Workclass','MaritalStatus','Occupation',
                            'Relationship','Race','Sex','NativeCountry']
            
            datadictionary = {} 
            for column,value in zip(column_names,self.data):
                datadictionary[column] = [value]
            
            self.df = pd.DataFrame(datadictionary)
            self.logging.log(f"Prediction data-\n{self.df}")
            self.df.to_csv(self.path)
            self.logging.log(f"Data Saved successfully to {self.path}")

            self.logging.log("Loading preprocessing model..")
            self.preprocessor = self.fileOperation.load_model("artifacts/preprocessor/Preprocessor.pkl")
            self.logging.log("Preprocessing Model has been load successfully")

            self.logging.log("loading train data and train preprocessor Model")
            self.finalData = self.fileOperation.load_data_from_csv("artifacts/data/FinalFeatureData.csv")
            self.preprocessor.fit(self.finalData)
            self.logging.log("Preprocessor Model has been trained with train data successfully")

            self.logging.log("Transforming prediction data")
            self.proccessedData = self.preprocessor.transform(self.df)
            self.logging.log("Prediction Data has been transformed successfully")
            
            self.logging.log("Saving prediction proccessed data..")
            save_npz(self.preprocessedDatapath,self.proccessedData)
            self.logging.log("Prediction proccessed data has been saved successfully")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        
    def prediction(self):
        try:
            self.logging.log("Prediction has been started")

            self.logging.log("Loading prediction proccessed data")
            self.preprocesseddata = load_npz(self.preprocessedDatapath).toarray()
            self.logging.log(f"Prediction processed data has been loaded from {self.preprocessedDatapath} Successfully")

            self.modelpath = "artifacts/models/TrainAndDeploy/BestModel.pkl"
            self.model = self.fileOperation.load_model(self.modelpath)
            self.logging.log(f"Best Model has been loaded from {self.modelpath} successfully")

            self.result = self.model.predict(self.preprocesseddata)
            self.logging.log("Model Predict Successfully")

            if self.result==0:
                self.df["Income"] = ["=<50K"]
                self.fileOperation.save_data_to_csv(self.df,f"artifacts/prediction/result/{self.current_time}.csv")
                return "Adult Census Income Estimate Approximately less than or equal to 50K"
            else:
                self.df["Income"] = [">50K"]
                self.fileOperation.save_data_to_csv(self.df,f"artifacts/prediction/result/{self.current_time}.csv")
                return "Adult Census Income Estimate Approximately more than  50K"
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e


        
        
        



        
        


        








