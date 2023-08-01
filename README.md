# Adult Census Income Prediction Project

## Overview

This project focuses on predicting an individual's income based on various demographic and socioeconomic features from the U.S. Census Bureau's dataset. The dataset contains information such as age, education, occupation, marital status, and more, and we aim to build a machine learning model that can accurately predict whether a person's income exceeds $50,000 per year.

## Dataset

The dataset used in this project is the "Adult Census Income" dataset. It contains the following features:

1. **Age**: The age of the individual. <br>
2. **Workclass**: The type of employment, such as private, self-employed, government, etc. <br>
3. **FNLWGT**: The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian noninstitutional population of the US. <br>
4. **Education**: The highest level of education achieved. <br>
5. **Education-Num**: A numerical representation of education, often representing the number of years of education completed. <br>
6. **Marital Status**: Whether the individual is married, divorced, widowed, etc. <br>
7. **Occupation**: The type of occupation the individual is engaged in. <br>
8. **Relationship**: The role in the family (e.g., husband, wife, etc.). <br>
9. **Race**: The racial group the individual belongs to. <br>
10. **Sex**: The gender of the individual. <br>
11. **Capital Gain**: The capital gains made by the individual in the past year. <br>
12. **Capital Loss**: The capital losses incurred by the individual in the past year. <br>
13. **Hours per week**: The number of hours the individual works per week. <br>
14. **Native-country**: The age of the individual. <br>
14. **Income**: The target variable, indicating whether the individual's income exceeds $50,000 (1) or not (0). <br>

## Project Structure

The project is organized into the following directories:
1. **app_tracking**: Contains logging scripts and exception handling scripts. <br>
1. **artifacts**: Contains the all data files, models and logs file. <br>
2. **notebooks**: Jupyter notebooks for data exploration, data preprocessing, model training, and evaluation. <br>
3. **PipelinePrediction**: Contain python scripts for prediction. <br>
4. **Pipeline_training**: Contains source code files, including scripts for data ingestion,data validation,data preprocessing, model training and model evaluation. <br>
5. **Utils**: Contains utits files. <br>
6. **create_project_structure.sh**: A file create general project structure.<br>
7. **application**: A file where the code flows. <br>
8. **requirements.txt**: A file specifying the required Python packages and their versions for running this project. <br>
9. **setup.py**: A file used for packaging and distribution purposes. <br>
10. **test.py**: A file used for test the train and predict pipeline <br>

### Classification Models Used:

 **GradientBoostingClassifier** <br>
 **XGBClassifier**<br>
 **LGBMClassifier** <br>
 **HistGradientBoostingClassifier** <br>

 ### Preprocessing Models Used:
  **StandardScaler** <br>
  **OneHotEncoder** <br>

 ### Mlops Tools Used:
  **Git** - Version Control <br>
  **Docker** - Containerization <br>
  **MLflow** -  Frameworks <br>
  **AWS** - Cloud Platforms <br>

 
## Installation

To run this project locally, you need create Python 3.9 Install the required packages by running the following command:

'conda create -p venv python=3.9 -y'

'source activate ./venv'

'pip install -r requirements.txt'

'python application.py'


## How to Use

1. Clone this repository to your local machine.

2. Navigate to the project directory:


3. Explore the Jupyter notebooks in the "notebooks" directory to understand the data, preprocess it, and build and evaluate the machine learning model.

4. Run the model training script in the "src" directory to train the model on the entire dataset:


5. The trained model will be saved in the "artifacts/models" directory.

## Conclusion

This project demonstrates the application of machine learning techniques to predict income levels based on demographic and socioeconomic features. The model can be further improved with hyperparameter tuning and more advanced algorithms. Feel free to explore, contribute, and extend this project to enhance its performance and usability.

## Credits

Dataset Source: UCI Machine Learning Repository - Adult Data Set - https://archive.ics.uci.edu/dataset/2/adult

## License

This project is licensed under the [MIT License](LICENSE).
# neww
