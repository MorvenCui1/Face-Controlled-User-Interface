import sqlite3

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#Connect to database or create it if it doesn't exist
conn = sqlite3.connect('mouthDatabase.db')

#Create table if table does not exist
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS mouthData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mouthgap REAL NOT NULL,
    mouthclose INTEGER NOT NULL
)
''')

#Delete all rows from table
cursor.execute("DELETE FROM mouthData")

#Read data in from file to insert into table
data = []
with open("mouthCloseData.txt", "r") as file:
    for line in file:
        gap, close = line.strip().split(",")
        data.append((float(gap), int(close)))

#Insert all data into table
cursor.execute("SELECT COUNT(*) FROM mouthData")
count = cursor.fetchone()[0]
if count == 0:
    cursor.executemany('''
        INSERT INTO mouthData (mouthgap, mouthclose)
        values (?, ?)
        ''', [(gap, int(close)) for gap, close in data])

#Insert all data from database into Pandas dataframe
df = pd.read_sql_query("SELECT * FROM mouthData", conn)

conn.commit() #Commit changes to database
conn.close() #Closes the connection to database

X = df[['mouthgap']] #2D array-like features
y = df['mouthclose'] #Target variable

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.1, random_state = 42
)

#Create random forest model with training data
modelMouth = RandomForestClassifier(n_estimators = 100, random_state = 42)
modelMouth.fit(X_train, y_train)
print("Random forest model for mouth data")

#5-fold cross validation, train on 4 parts and test on 1
scores = cross_val_score(modelMouth, X, y, cv = 5, scoring = 'accuracy')

#Print cross validation backtesting results
print("Cross-validation accuracies: ", scores)
print("Mean accuracy: ", scores.mean())

# Predict on test set and evaluate accuracy
y_pred = modelMouth.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Predict for new samples
new_mouthgap = [[-24]]
prediction1 = modelMouth.predict(new_mouthgap)
print("Mouthgap: -24")
print("Predicted mouthclose: ", prediction1[0])
print("Expected prediction: 1")

new_mouthgap = [[-5]]
prediction2 = modelMouth.predict(new_mouthgap)
print("Mouthgap: -5")
print("Predicted mouthclose: ", prediction2[0])
print("Expected prediction: 0")