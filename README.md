# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Placement_Data.csv dataset.
2. Drop the unwanted columns (sl_no, salary) and convert the categorical columns into numbers using LabelEncoder.
3. Split the data into features (X) and target (placement status), then into training and testing sets, and train a Logistic Regression model.
4. Predict the placement status for the test data and evaluate the model using accuracy, confusion matrix and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Udhaya Nandhini M
RegisterNumber:  212225240177
*/
```
```
import io
import pandas as pd
from google.colab import files
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

uploaded = files.upload()
data = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))
print(data.head())

data1 = data.drop(["sl_no", "salary"], axis=1, errors="ignore")
print(data1.isnull().sum())

le = LabelEncoder()
for col in data1.select_dtypes(include="object").columns:
    data1[col] = le.fit_transform(data1[col])
print(data1.head())

X = data1.drop("status", axis=1)
y = data1["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted:", y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![alt text](<Screenshot 2026-11-03 092115.png>)
![alt text](<Screenshot 2026-11-03 092127.png>)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
