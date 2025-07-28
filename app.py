from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load("model.pkl")
le_gender = joblib.load("encoder_gender.pkl")
le_married = joblib.load("encoder_married.pkl")
le_approved = joblib.load("encoder_approved.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            income = float(request.form["income"])
            loan_amount = float(request.form["loan_amount"])
            loan_term = int(request.form["loan_term"])
            credit_score = int(request.form["credit_score"])
            gender = request.form["gender"]
            married = request.form["married"]

            gender_enc = le_gender.transform([gender])[0]
            married_enc = le_married.transform([married])[0]

            features = np.array([[age, income, loan_amount, loan_term, credit_score, gender_enc, married_enc]])
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
