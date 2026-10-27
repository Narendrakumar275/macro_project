print("AI-Based Crop Recommendation System")
print("Model Training Started")
print("Hello, Farmer")
import pandas as pd
data = pd.read_csv("Crop_recommendation.csv")
print(data.head())
X =data[
    ['N', 'P', 'K', 'temperature',
      'humidity', 'ph', 'rainfall']
]

#select output

y =data['label']
print("Input Data:")
print(X.head())
print("\nOutput Data:")
print(y.head())
print("\nInput shape:", X.shape)
print("Output shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print("\nTraining Data shape:", X_train.shape)
print("Testing Data shape:", X_test.shape)

# create a random forest model

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators =100,
                               random_state = 42)
model.fit(X_train, y_train)
print("\nModel training completed")

#model accuracy check

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:",round(accuracy*100,2),"%")

#train model save

import joblib
joblib.dump(model,"crop_model.pkl")
print("\nModel saved successfully")