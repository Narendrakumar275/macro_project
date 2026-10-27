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