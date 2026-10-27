import joblib
import pandas as pd

# Load the trained model
model = joblib.load("crop_model.pkl")
# Take soil and weather values from the user

N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter soil pH: "))
rainfall = float(input("Enter Rainfall (mm): "))
# Enter soil and weather values
soil_data = pd.DataFrame([{
    "N": N,
    "P": P,
    "K": K,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall
}])

# Predict the suitable crop
prediction = model.predict(soil_data)

print("Recommended Crop:", prediction[0])
# Fertilizer recommendation function

def recommend_fertilizer(crop):
    fertilizer_guide = {
        "rice": "Check soil-test results and local rice fertilizer recommendations.",
        "wheat": "Check soil-test results and local wheat fertilizer recommendations.",
        "maize": "Check soil-test results and local maize fertilizer recommendations."
    }

    return fertilizer_guide.get(
        crop.lower(),
        "Please consult local agricultural recommendations for this crop."
    )


# Display fertilizer guidance
fertilizer = recommend_fertilizer(prediction[0])

print("Recommended Crop:", prediction[0])
print("Fertilizer Guidance:", fertilizer)