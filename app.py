from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "heart-disease-model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]

def calculate_risk(probability):
    if probability < 0.30:
        return "Low Risk", "low"
    if probability < 0.60:
        return "Moderate Risk", "moderate"
    return "High Risk", "high"

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Age": int(request.form["Age"]),
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingBP": int(request.form["RestingBP"]),
            "Cholesterol": int(request.form["Cholesterol"]),
            "FastingBS": int(request.form["FastingBS"]),
            "RestingECG": request.form["RestingECG"],
            "MaxHR": int(request.form["MaxHR"]),
            "ExerciseAngina": request.form["ExerciseAngina"],
            "Oldpeak": float(request.form["Oldpeak"]),
            "ST_Slope": request.form["ST_Slope"]
        }

        input_df = pd.DataFrame([data], columns=FEATURES)
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = int(model.predict(input_df)[0])

        risk_label, risk_class = calculate_risk(probability)

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability * 100, 1),
            risk_label=risk_label,
            risk_class=risk_class,
            submitted=data
        )

    except Exception as error:
        return render_template(
            "index.html",
            error=f"Unable to process the information: {error}"
        )

if __name__ == "__main__":
    app.run(debug=True)
