from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("model.pkl")
le_approved = joblib.load("encoder_approved.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect numeric inputs only
            age = int(request.form["age"])
            income = float(request.form["income"])
            credit_score = int(request.form["credit_score"])

            # Prepare features: [age, income, credit_score]
            features = np.array([[age, income, credit_score]])

            # Predict
            prediction_encoded = model.predict(features)[0]
            prediction = le_approved.inverse_transform([prediction_encoded])[0]

            return render_template("form.html", prediction=prediction)

        except Exception as e:
            return render_template("form.html", error=str(e))

    return render_template("form.html")

# ✅ Health check endpoint
@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
