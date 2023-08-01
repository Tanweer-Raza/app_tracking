import datetime
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from Pipeline_prediction.predictFromModel import Prediction
from Pipeline_training.trainingModel import train_model
from app_tracking.exception import AppException
from app_tracking.logger import App_Logger

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
path = f"artifacts/logs/main/{current_time}.txt"
logging = App_Logger(path)

logging.log("Creating application")

application = Flask(__name__)
app = application
CORS(app)
logging.log("Application has been Created")


@app.route("/", methods=['POST','GET'])
@cross_origin()
def home():
    logging.log("Home Page")
    return render_template('index.html')

@app.route("/option", methods=['POST','GET'])
@cross_origin()
def option():
    logging.log("Data Source Option Page")
    return render_template('option.html')

@app.route('/train', methods=['POST','GET'])
@cross_origin()
def train():
    logging.log("Training Page")
    if request.method == 'POST':
        source = request.form.get('Local') or request.form.get('Cloud')
        logging.log(f"{source}")
        if source:
            try:
                train_model(source)
                message = "Model Trained Successfully"
                logging.log(message)
                return render_template("train.html", message=message)      
            except Exception as e:
                message = "Oops! Something Went Wrong \n"
                return render_template("train.html", message=message)
        else: 
            raise Exception("Option not selected")
    else:          
        message = "Train Page didn't get 'POST' request"
        logging.log(message)
        return render_template("train.html", message=message)

@app.route("/test", methods=['POST','GET'])
@cross_origin()
def test():
    return render_template("test.html")

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['Age'])
            workclass = request.form['Workclass']
            fnlwgt = float(request.form['FNLWGT'])
            education = request.form['Education']
            marital_status = request.form['Marital Status']
            occupation = request.form['Occupation']
            relationship = request.form['Relationship']
            race = request.form['Race']
            sex = float(request.form['Sex'])
            capital_gain = float(request.form['Capital Gain'])
            capital_loss = float(request.form['Capital Loss'])
            hours_per_week = float(request.form['Hours/Week'])
            country = request.form['Country']

            data = [age, workclass, fnlwgt, education, marital_status,
                    occupation, relationship, race, sex, capital_gain,
                    capital_loss, hours_per_week, country]

            predict_object = Prediction(data=data)
            result = predict_object.prediction()
            logging.log(result)
            return render_template('result.html', result=result)
        except Exception as e:
            logging.log(f"Oops ! \n{str(e)}")
            return render_template('result.html', result="Oops! Something Went Wrong")
            raise AppException(e,sys) from e
    else: 
        logging.log("Predict Page didn't get 'POST' request")  
        return render_template('result.html', result="Predict Page didn't get 'POST' request")
    

@app.route("/contact", methods=['POST','GET'])
@cross_origin()
def contact():
    logging.log("Contact Page")
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
