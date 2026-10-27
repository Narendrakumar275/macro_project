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
