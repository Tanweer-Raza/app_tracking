import datetime,os,sys
import pandas as pd
import numpy as np
import contextlib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from Utils.utils import FileOperation
from scipy.sparse import save_npz



class DataPreprocessing:
     """
     This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

     Written By: Rachit More
     Version: 1.0
     Revisions: None
     """

     def __init__(self):
          try:
               print("Data Preprocessing has been started")
               current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
               self.filepath = f"artifacts/logs/Stage3_DataPreproccesed/{current_time}.txt"
               self.logging = App_Logger(self.filepath)
               self.fileoperation = FileOperation()
               self.path = "artifacts/data/raw.csv"
               self.data = self.fileoperation.load_data_from_csv(self.path)
               self.df = pd.DataFrame(self.data)
               self.column_names = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'EducationNum',
                                   'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex',
                                   'Capital_gain', 'Capital_loss', 'HoursPerWeek', 'NativeCountry',
                                   'Income']
               with contextlib.suppress():
                    self.dataprocessing()
                    self.splitData()
                    self.scaling_and_encoding()
          except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
          
     def dataprocessing(self):
          try:
               self.logging.log("Data Preprocessing Started")
               self.logging.log(f"DataFrame Head -\n{self.df.head()}")
               self.settingColumnsName()
               self.logging.log(f"Updated Columns DataFrame Head -\n{self.df.head()}")
               self.logging.log(f"Featurename has been changed with standard columns name {self.column_names}")

               self.logging.log(f"DataFrame shape -\n{self.df.shape}")
               self.df.drop_duplicates(keep='first',inplace=True)
               self.logging.log("Duplicate values has been droped")
               self.logging.log(f"Updated DataFrame shape -\n{self.df.shape}")
               

               self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
               self.logging.log("Whitespace has been removed")
               self.logging.log(f"Updated DataFrame Head -\n{self.df.head()}")

               self.df.replace('?',np.NaN,inplace=True)
               self.logging.log("Special character ? replaced to np.NaN value")

               self.logging.log("Filling missing values")
               self.df['Workclass'].fillna(self.df['Workclass'].mode(),inplace=True)
               self.logging.log(f"Workclass null values replaced with {self.df['Workclass'].mode()}")

               self.df['Occupation'].fillna(self.df['Occupation'].mode(),inplace=True)
               self.logging.log(f"Occupation null values replaced with {self.df['Occupation'].mode()}")

               self.df['NativeCountry'].fillna(self.df['NativeCountry'].mode(),inplace=True)
               self.logging.log(f"NativeCountry null values replaced with {self.df['NativeCountry'].mode()}")

               self.df['Sex'] = self.df['Sex'].map({'Female': 0, 'Male': 1})

               self.fileoperation.save_data_to_csv(self.df, "artifacts/data/CleanData.csv")
               self.logging.log("Data is clean and saved to the artifacts/data/CleanData.csv file")
               self.logging.log(f"DataFrame Head -\n{self.df.head()}")
                                                  
               self.logging.log("Extracting feature from dataset by dropping target variable and unnessary feature")
               self.X = self.df.drop(labels=['Income','Education'], axis=1)
               self.logging.log("Income and Education feature has been dropped")

               self.fileoperation.save_data_to_csv(self.X,"artifacts/data/FinalFeatureData.csv")
               self.logging.log("Final-feature Data has been saved to artifacts/data/FinalFeatureData.csv for train and deploy process")

               self.logging.log("Extracting target from dataset.")
               self.logging.log("Mapping target value with numerical values")
               self.y = self.df['Income'].map({'<=50K': 0, '>50K': 1})
               self.logging.log("Target values has been mapped")

               self.fileoperation.save_data_to_csv(self.y,"artifacts/data/FinalTargetData.csv")
               self.logging.log("Final-target dataset has been saved to artifacts/data/FinalTargetData.csv")

               self.logging.log("Data Preprocessed Successfully")
          except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e


     def splitData(self):
          try:
               self.logging.log("Splitting the Final-feature and Final-target values")
               self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                           test_size=0.33, random_state=42)
               self.logging.log("Final-feature and Final-target data has been splitted")

               self.fileoperation.save_data_to_csv(self.X_train,"artifacts/data/X_train.csv")
               self.logging.log("X_train data has been saved to artifacts/data/X_train.csv")

               self.fileoperation.save_data_to_csv(self.X_test,"artifacts/data/X_test.csv")
               self.logging.log("X_test data has been saved to artifacts/data/X_test.csv")

               self.fileoperation.save_data_to_csv(self.y_train,"artifacts/data/y_train.csv")
               self.logging.log("y_train data has been saved to artifacts/data/y_train.csv")

               self.fileoperation.save_data_to_csv(self.y_test,"artifacts/data/y_test.csv")
               self.logging.log("y_test data has been saved to artifacts/data/y_test.csv")
          except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
          
          
     def scaling_and_encoding(self):
          try:
               self.logging.log("Data Scaling and Encoding Started")
               # Create transformers for preprocessing steps
               self.logging.log("Extracting numerical and categorical feature columns")
               self.num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns
               self.logging.log(f"Numerical columns are {self.num_cols}")

               self.cat_cols = self.X_train.select_dtypes(include=['object']).columns
               self.logging.log(f"Categorical columns are {self.cat_cols}")
               
               self.logging.log("Creating DataPreprocessing pipeline for numerical and categorical feature")
               Encoder = Pipeline(steps=[
               ('onehot', OneHotEncoder(drop='first'))])

               Scaler = Pipeline(steps=[
               ('scaler', StandardScaler())])

               preprocessor = ColumnTransformer(transformers=[
                              ('cat', Encoder, self.cat_cols),
                              ('num', Scaler, self.num_cols)])  
               self.fileoperation.delete_files_in_directory("artifacts/preprocessor")
               self.fileoperation.save_model(preprocessor, "artifacts/preprocessor/Preprocessor.pkl")
               self.logging.log("Data preprocessing pipeline has been created and saved to artifacts/preprocessor/Preprocessor.pkl")
               
               # Fit and transform the preprocessor on the Final and X_train dataset
               self.logging.log("Fitting and transforming X_train Dataset")
               self.preprocessed_X_train = preprocessor.fit_transform(self.X_train)
               self.train_path = 'artifacts/data/X_trainSparse_matrix.npz'
               save_npz(self.train_path,self.preprocessed_X_train)
               self.logging.log("X_train dataset has been fit and transformed and saved to artifacts/data/X_trainSparse_matrix.npz")

               self.logging.log("Fitting and transforming Final_train Dataset")
               self.preprocessed_Final_train = preprocessor.fit_transform(self.X)
               self.Final_train_path = 'artifacts/data/Final_trainSparse_matrix.npz'
               save_npz(self.Final_train_path,self.preprocessed_Final_train)
               self.logging.log("Final_train dataset has been fit and transformed and saved to artifacts/data/Final_trainSparse_matrix.npz")
               
               self.logging.log("Transforming X_test Dataset")
               self.test_path = 'artifacts/data/X_testSparse_matrix.npz'
               self.preprocessed_X_test = preprocessor.transform(self.X_test)
               save_npz(self.test_path,self.preprocessed_X_test)
               self.logging.log("X_test dataset has been transformed and saved to artifacts/data/X_trainSparse_matrix.npz")
               
               self.logging.log("Data Scaling and Encoding has been done successfully")
          except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
          
     
     def settingColumnsName(self):
          try:
               self.logging.log("Remove spaces around the column names in the DataFrame")
               self.df.columns = self.df.columns.str.strip()
               standardColumnsName = self.column_names
               self.logging.log("Create a dictionary to map the original column names to the standard column names")
               column_mapping = dict(zip(self.df.columns, standardColumnsName))

               self.logging.log("Rename the DataFrame columns using the dictionary")
               self.df.rename(columns=column_mapping, inplace=True)

               self.logging.log("Updated column names:")
          except Exception as e:
              self.logging.log(str(e))
              raise AppException(e,sys) from e


          
                                                  




     


          
          
          

