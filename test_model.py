
import joblib
import pandas as pd

print("CropAI Unit Testing Started")

# Test 1: Load model
model = joblib.load("crop_model.pkl")
print("TEST 1 PASSED: Model loaded")

# Test 2: Prepare sample input
sample_data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])

assert sample_data.shape == (1, 7)
print("TEST 2 PASSED: Input data is correct")

# Test 3: Predict crop
prediction = model.predict(sample_data)

assert len(prediction) == 1
assert isinstance(prediction[0], str)
assert prediction[0] != ""

print("TEST 3 PASSED: Prediction generated")
print("Predicted Crop:", prediction[0])

print("\nAll unit tests passed!")
