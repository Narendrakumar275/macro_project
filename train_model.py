
print("AI-Based Crop Recommendation System")
print("Model Training Started")
print("Hello, Farmer")

import pandas as pd
import joblib
#GridSearch optimization for the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# load dataset
data = pd.read_csv("Crop_recommendation.csv")

print("\nFirst 5 rows:")
print(data.head())

# Check missing values
print("\nMissing values in each column:")
print(data.isnull().sum())
print("Total missing values:", data.isnull().sum().sum())

# Check duplicate rows
print("\nTotal rows:", len(data))
print("Duplicate rows:", data.duplicated().sum())

# remove missing values and duplicate rows
data = data.dropna()
data = data.drop_duplicates()

# Select input features
X = data[
    ['N', 'P', 'K', 'temperature',
     'humidity', 'ph', 'rainfall']
]

# Select target
y = data['label']

print("\nInput shape:", X.shape)
print("Output shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# random forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Random Forest evaluation
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")

print("\nRandom Forest training completed")
print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")
print("Random Forest F1-score:",
      round(rf_f1, 4))
# Grid Search Hyperparameter Optimization

print("\nGrid Search Started")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

optimized_pred = best_model.predict(X_test)

optimized_accuracy = accuracy_score(y_test, optimized_pred)
optimized_f1 = f1_score(
    y_test, optimized_pred, average="weighted"
)

print("\nOptimized Random Forest Results")
print("Accuracy:", round(optimized_accuracy * 100, 2), "%")
print("F1-score:", round(optimized_f1, 4))

# decision tree model
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

# Decision Tree evaluation
dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average="weighted")

print("\nDecision Tree training completed")
print("Decision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")
print("Decision Tree F1-score:",
      round(dt_f1, 4))

# Save random forest model as the main model
joblib.dump(best_model, "crop_model.pkl")

print("\nOptimized Random Forest model saved successfully")

print("\nRandom Forest model saved successfully")
