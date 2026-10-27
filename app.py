from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("crop_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values submitted through the website form
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Prepare input in the format expected by the model
        soil_data = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }])

        # Predict the crop
        prediction = model.predict(soil_data)[0]

        return render_template(
            "index.html",
            prediction=prediction
        )

    except (ValueError, KeyError):
        return "Please enter valid numeric values for all fields.", 400


if __name__ == "__main__":
    app.run(debug=True)