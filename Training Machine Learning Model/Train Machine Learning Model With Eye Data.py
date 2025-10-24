import os

import sqlite3

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "eyeDatabase.db")
data_path = os.path.join(base_dir, "eyeCloseData.txt")
model_path = os.path.join(base_dir, "modelEye.pkl")

#Connect to database or create it if it doesn't exist
conn = sqlite3.connect(db_path)

#Create table if table does not exist
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS eyeData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eyegap REAL NOT NULL,
    eyeclose INTEGER NOT NULL
)
''')

#Delete all rows from table
#cursor.execute("DELETE FROM eyeData")

#Read data in from file to insert into table
data = []
with open(data_path, "r") as file:
    for line in file:
        gap, close = line.strip().split(",")
        data.append((float(gap), int(close)))

#Insert all data into table if table is empty
cursor.execute("SELECT COUNT(*) FROM eyeData")
count = cursor.fetchone()[0]
if count == 0:
    cursor.executemany('''
        INSERT INTO eyeData (eyegap, eyeclose)
        values (?, ?)
        ''', [(gap, int(close)) for gap, close in data])

#Insert all data from database into Pandas dataframe
df = pd.read_sql_query("SELECT * FROM eyeData", conn)

conn.commit() #Commit changes to database
conn.close() #Closes the connection to database

X = df[['eyegap']] #2D array-like features
y = df['eyeclose'] #Target variable

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.1, random_state = 42
)

#Create random forest model with training data
modelEye = RandomForestClassifier(n_estimators = 100, random_state = 42)
modelEye.fit(X_train, y_train)
print("Random forest model for eye data")

#5-fold cross validation, train on 4 parts and test on 1
scores = cross_val_score(modelEye, X, y, cv = 5, scoring = 'accuracy')

#Print cross validation backtesting results
print("Cross-validation accuracies: ", scores)
print("Mean accuracy: ", scores.mean())

# Predict on test set and evaluate accuracy
y_pred = modelEye.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Predict for new samples
new_eyegap = [[0.010]]
prediction1 = modelEye.predict(new_eyegap)
print("Eyegap: 0.010")
print("Predicted eyeclose: ", prediction1[0])
print("Expected prediction: 1")

new_eyegap = [[0.020]]
prediction2 = modelEye.predict(new_eyegap)
print("Eyegap: 0.020")
print("Predicted eyeclose: ", prediction2[0])
print("Expected prediction: 0")

#Save model
joblib.dump(modelEye, model_path)