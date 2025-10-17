This project uses Machine Learning (SVM Classifier) to predict whether a person is diabetic or not based on diagnostic measurements from the PIMA Indian Diabetes Dataset.
It demonstrates data preprocessing, standardization, model training, and evaluation using Python and Scikit-learn.


This project uses Support Vector Machines (SVM) to classify individuals as Diabetic (1) or Non-Diabetic (0).

#  Dataset

Dataset Name: PIMA Indian Diabetes Dataset
Source: Kaggle - Pima Indians Diabetes Database


## steps Involved 
1️Importing Dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

2️Loading the Dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

3️ Data Exploration

.head() – Displays first 5 rows

.shape – Shows dataset size

.describe() – Statistical summary

.value_counts() – Distribution of diabetic vs. non-diabetic

4️ Splitting Features and Labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

5️  Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

6️ Train-Test Split and Model Training
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

7️ Model Evaluation
X_train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

 Output

Training Accuracy: ~78–80%

Testing Accuracy: ~75–78%
(Accuracy may vary slightly depending on random seed)

Data standardization is cruinear kernel works effectively for this dataset.

Model generalization can be improved with hyperparameter tuning or other ML models.
