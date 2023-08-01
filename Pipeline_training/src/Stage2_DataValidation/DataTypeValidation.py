import datetime
import os
import sys
import pandas as pd
import numpy as np
import contextlib
from scipy.sparse import load_npz
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from Utils.utils import FileOperation


class RawDataValidation:
    """
      This class shall be used for data validation
      Written By: Rachit More
      Version: 1.0
      Revisions: None
      """
    def __init__(self,path="artifacts/data/raw.csv"):
        try:
            print("Raw Data Validation has been started")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            self.filepath = f"artifacts/logs/Stage2_RawDatavalidation/{current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileoperation = FileOperation()
            self.path = path
            self.data = self.fileoperation.load_data_from_csv(self.path)
            self.df = pd.DataFrame(self.data)
            self.logging.log(f"{self.df.head}")
            self.logging.log(f"{self.df.columns}")
            self.columns_name = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'EducationNum',
        'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex',
        'Capital_gain', 'Capital_loss', 'HoursPerWeek', 'NativeCountry',
        'Income']
            with contextlib.suppress():
                self.main()
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
    
             
    def isMissingValues(self):
        try:
            if all(self.df.isnull().sum()) == 0 :
                self.logging.log("There is no missing values in dataset")
                return True
            else:
                self.logging.log(f"There are {self.df.isnull().sum()} missing values in dataset ") 
                return False
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        
    def isvalidShape(self):
        try:
            if len(self.df.columns) == 15:
                self.logging.log("There is no missing feature or extra feature in dataset")
                return True
            else:
                self.logging.log(f"There is missing feature or extra feature in dataset {self.df.columns}") 
                return False
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
              
    def isSpecialCharacter(self):
        try:
            special_characters = set('!@#$%^&*?')
            for col in self.df.columns:
                for datapoint in self.df[col]:
                    if any(char.strip() in special_characters for char in str(datapoint)):
                        self.logging.log("There are some special character in dataset")
                        return False
            self.logging.log("There is no special character in dataset")
            return True
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
            
              
    def isDuplicateRows(self):
        try:
            if all(self.df.duplicated()):
                self.logging.log("There is no duplicate values in dataset") 
                return True
            else:
                self.logging.log(f"There is duplicate values in dataset {self.df.duplicated()}")  
                return False
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e   
         
        

    def isValidDataTypes(self):
        try:
            if all(self.df.dtypes != "object"):
                self.logging.log("All feature in Dataset have numeric dtype.")  
                return True
            else:
                self.logging.log(f"All feature in Dataset don't have numeric dtype {self.df.dtypes}.")
                return False
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        
    def isValidColumnName(self):
        try:
            if all(self.df.columns == self.columns_name):
                self.logging.log("Dataset has valid column names.") 
                return True
            else:
                self.logging.log(f"Dataset doesn't have valid column names {self.df.columns}..")
                return False  
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        

    def main(self):
        try:
            self.validating = [self.isMissingValues(),self.isvalidShape(),self.isSpecialCharacter(),
                            self.isDuplicateRows(),self.isValidDataTypes(),self.isValidColumnName(),
                            ]
            
            if all(self.validating):
                self.logging.log("Data validation has been successfully performed, and the data is now fully prepared and ready for scaling or encoding.")
                self.fileoperation.save_data_to_csv(self.df,"artifacts/data/GoodData/GoodData.csv")
                self.fileoperation.save_data_to_csv(self.cat_cols,"artifacts/data/GoodCatData.csv")
                self.fileoperation.save_data_to_csv(self.num_cols,"artifacts/data/GoodNumData.csv")

            else:
                self.logging.log("Data validation has been failed, and the data preprocessing is required.")
                self.fileoperation.save_data_to_csv(self.df,"artifacts/data/BadData.csv")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e


class PrePreocessedDataValidation:
    def __init__(self):
        try:
            print("PrePreocessed data validation has been started")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            self.filepath = f"artifacts/logs/Stage2_PreprocessedDatavalidation/{current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileoperation = FileOperation()

            self.X_train_dense_matrix = load_npz('artifacts/data/X_trainSparse_matrix.npz').toarray()
            self.X_test_dense_matrix= load_npz('artifacts/data/X_testSparse_matrix.npz').toarray()
            self.final_train_dense_matrix = load_npz('artifacts/data/Final_trainSparse_matrix.npz').toarray()
            self.y_train = self.fileoperation.load_data_from_csv("artifacts/data/y_train.csv")
            self.y_test = self.fileoperation.load_data_from_csv("artifacts/data/y_test.csv")
            self.final_target = self.fileoperation.load_data_from_csv("artifacts/data/FinalTargetData.csv")
            
            self.X_train = pd.DataFrame(self.X_train_dense_matrix)
            self.logging.log(f"X_train head :\n{self.X_train.head()}")

            self.X_test = pd.DataFrame(self.X_test_dense_matrix)
            self.logging.log(f"X_test head :\n{self.X_test.head()}")

            self.final_train = pd.DataFrame(self.final_train_dense_matrix)
            self.logging.log(f"X_test head :\n{self.X_test.head()}")

            with contextlib.suppress():
                self.validate(self.X_train,"X_train")
                self.validate(self.X_test,"X_test")
                self.validate(self.y_train,"y_train")
                self.validate(self.y_test,"y_test")
                self.validate(self.final_train,"Final_train")
                self.validate(self.final_target,"final_target")

        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e


    def validate(self,data,dataType):
        try:
            data = data.astype(int)
        except:
            pass

        try:
            if all(data.dtypes != "object"):
                if sum(data.isnull().sum()) !=0:
                    self.logging.log(f"\n{data.dtypes} \nThere are {data.isnull().sum()} null values in data set \nPreprocessed {dataType} Data validation has been failed, and the data preprocessing is required.\n{data}")
                    self.fileoperation.save_data_to_csv(data,f"artifacts/data/BadPreprcessed{dataType}Data.csv")
                else:
                    self.logging.log("Data validation has been successfully performed, and the data is now fully prepared and ready for model training.")
            
            else:
                self.logging.log(f"{data.dtypes}\nPreprocessed {dataType} data validation has been failed, and the data preprocessing is required. \n{data}")
                self.fileoperation.save_data_to_csv(data,f"artifacts/data/BadPreprcessed{dataType}Data.csv")

        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        




    

        
    
    






    