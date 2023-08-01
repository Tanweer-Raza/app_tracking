import datetime,os,sys
import contextlib
from scipy.sparse import load_npz
from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb 
from sklearn.metrics  import roc_auc_score
from Utils.utils import FileOperation
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException


class ModelTraining:
    def __init__(self):
        try:
            print("Model Training")
            self.current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            self.filepath = f"artifacts/logs/Stage4_ModelTraining/{self.current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileoperation = FileOperation()
            self.X_train = load_npz('artifacts/data/X_trainSparse_matrix.npz').toarray()
            self.X_test = load_npz('artifacts/data/X_testSparse_matrix.npz').toarray()
            self.y_train_path = "artifacts/data/y_train.csv"
            self.y_train = self.fileoperation.load_data_from_csv(self.y_train_path)
            self.y_test_path = "artifacts/data/y_test.csv"
            self.y_test = self.fileoperation.load_data_from_csv(self.y_test_path)
            self.finalTrain = load_npz('artifacts/data/Final_trainSparse_matrix.npz').toarray()
            self.target_path = "artifacts/data/FinalTargetData.csv" 
            self.target = self.fileoperation.load_data_from_csv(self.target_path)

            with contextlib.suppress():
                self.create_and_train_model()
                self.bestmodel()


            with contextlib.suppress():
                self.bestModelForDeploy()
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        
    
        
    def create_and_train_model(self):
        try:
            print("Model Training Started")
            self.logging.log("Model Training has been Started")

            self.logging.log("creating and fitting GradientBoostingClassifier model")
            self.model1 = GradientBoostingClassifier(learning_rate = 0.2, 
                                                    n_estimators = 200).fit(self.X_train,self.y_train)
            self.logging.log(f"GradientBoostingClassifier model {self.model1} has been trained")
            
            # self.fileoperation.save_model(self.model1, f"artifacts/models/{modeltype}/GradientBoostingClassifier.pkl")
            # self.logging.log(f"GradientBoostingClassifier model has been saved to artifacts/models/{modeltype}/GradientBoostingClassifier.pkl")

            self.logging.log(f"creating and fitting XGBClassifier model")
            self.model2 = XGBClassifier(learning_rate = 0.1,
                                        n_estimators = 200).fit(self.X_train,self.y_train)
            self.logging.log(f"XGBClassifier model {self.model2} has been trained")

            # self.fileoperation.save_model(self.model2, f"artifacts/models/{modeltype}/XGBClassifier.pkl")
            # self.logging.log(f"XGBClassifier model has been saved to artifacts/models/{modeltype}/XGBClassifier.pkl")


            self.logging.log("creating and fitting LGBMClassifier model")
            self.model3 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31,
                                        learning_rate=0.05, n_estimators=100, 
                                        objective='binary', random_state=42).fit(self.X_train,self.y_train)
            self.logging.log(f"LGBMClassifier model {self.model3} has been trained")

            # self.fileoperation.save_model(self.model3, f"artifacts/models/{modeltype}/LGBMClassifier.pkl")
            # self.logging.log(f"LGBMClassifier model has been saved to artifacts/models/{modeltype}/LGBMClassifier.pkl")

            
            self.logging.log("creating and fitting HistGradientBoostingClassifier model")
            self.model4 = HistGradientBoostingClassifier(learning_rate=0.2, 
                                                        max_depth=10).fit(self.X_train,self.y_train)
            self.logging.log(f"HistGradientBoostingClassifier model {self.model4} has been trained")

            # self.fileoperation.save_model(self.model4, f"artifacts/models/{modeltype}/HistGradientBoostingClassifier.pkl")
            # self.logging.log(f"HistGradientBoostingClassifier model has been saved to artifacts/models/{modeltype}/HistGradientBoostingClassifier.pkl")

            self.logging.log("Model Train has been Done Successfully")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        

    def auc_roc_score(self,model):
        try:
            self.logging.log("Calculating AUC and ROC Score")
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            print(self.y_test.shape, y_pred_proba.shape)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            self.logging.log(f"AUC and ROC Score has been calculated for {model} - {auc_roc}")
            return auc_roc
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

    def bestmodel(self):
        try:
            self.logging.log("Extracting best model")

            self.logging.log("Loading models ..")
            # model1 = self.fileoperation.load_model(f"artifacts/models/{modeltype}/GradientBoostingClassifier.pkl")
            # model2 = self.fileoperation.load_model(f"artifacts/models/{modeltype}/XGBClassifier.pkl")
            # model3 = self.fileoperation.load_model(f"artifacts/models/{modeltype}/LGBMClassifier.pkl")
            # model4 = self.fileoperation.load_model(f"artifacts/models/{modeltype}/HistGradientBoostingClassifier.pkl")
            self.logging.log("Models has been loaded")

            self.logging.log("Creating dictionary of model")
            self.models = {type(self.model1).__name__: self.model1,
                        type(self.model2).__name__: self.model2,
                        type(self.model3).__name__: self.model3,
                        type(self.model4).__name__: self.model4}

            self.best_model = None
            best_auc_roc =  0
            for model_name, model in self.models.items():
                auc_roc = self.auc_roc_score(model)
                if auc_roc > best_auc_roc:
                    self.best_model = model
                    best_auc_roc = auc_roc 
            self.logging.log(f"Best Model - {self.best_model} and auc and roc - {best_auc_roc}")
            self.fileoperation.delete_files_in_directory("artifacts/models/TrainAndTest")
            self.fileoperation.save_model(self.best_model,"artifacts/models/TrainAndTest/BestModel.pkl")
            self.logging.log("Best Model found successfully")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e


    def bestModelForDeploy(self):
        try:
            self.logging.log("Loading models ..")
            # model1 = self.fileoperation.load_model(f"artifacts/models/TrainAndDeploy/GradientBoostingClassifier.pkl")
            # model2 = self.fileoperation.load_model(f"artifacts/models/TrainAndDeploy/XGBClassifier.pkl")
            # model3 = self.fileoperation.load_model(f"artifacts/models/TrainAndDeploy/LGBMClassifier.pkl")
            # model4 = self.fileoperation.load_model(f"artifacts/models/TrainAndDeploy/HistGradientBoostingClassifier.pkl")

            model1 = GradientBoostingClassifier(learning_rate = 0.2, 
                                                    n_estimators = 200)
            
            model2 = XGBClassifier(learning_rate = 0.1,
                                        n_estimators = 200)
            
            model3 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31,
                                        learning_rate=0.05, n_estimators=100, 
                                        objective='binary', random_state=42)
            model4 = HistGradientBoostingClassifier(learning_rate=0.2, 
                                                        max_depth=10) 

            models = {type(model1).__name__ : model1,
                        type(model2).__name__: model2,
                        type(model3).__name__: model3,
                        type(model4).__name__: model4}
            
            for model_name,model in models.items():
                if type(self.best_model).__name__ == model_name:
                    model.fit(self.finalTrain,self.target)
                    
                    self.fileoperation.delete_files_in_directory("artifacts/models/TrainAndDeploy")
                    self.fileoperation.save_model(model,f"artifacts/models/TrainAndDeploy/BestModel.pkl")
                    self.logging.log("Best Model found for train and deploy successfully")
                    break
                else:
                    self.logging.log(f"Best Model Name {type(self.best_model).__name__} and Deploy Model Name{model_name}")
                    self.logging.log("no best Model found for train and deploy successfully")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e




        





