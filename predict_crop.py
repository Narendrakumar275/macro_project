import joblib
import pandas as pd

# Load the trained model
model = joblib.load("crop_model.pkl")

# Enter soil and weather values
soil_data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202
}])

# Predict the suitable crop
prediction = model.predict(soil_data)

print("Recommended Crop:", prediction[0])