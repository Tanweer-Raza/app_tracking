import sys
from Pipeline_training.trainingModel import train_model
from Pipeline_prediction.predictFromModel import Prediction
from app_tracking.exception import AppException

test1 = [25,"Private",12345, 10, "Never-married",
          "Tech-Support", "Never-married", "Indian Face",
            "Male",0,0,60, "India"]

if __name__=="__main__":
    try:
      train_model("Cloud")
      Obj = Prediction(test1)
      Obj.prediction()
    except:
       try:
          train_model("Cloud")
          Obj = Prediction(test1)
          Obj.prediction()
        except Exception as e:
          raise AppException(e,sys) as e

