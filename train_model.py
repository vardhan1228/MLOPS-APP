from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Sample data: [age, income, loan_amount, loan_term, credit_score, gender, married]
X = [
    [22, 40000, 15000, 12, 600, "male", "yes"],     # denied (age < 25 and credit score < 650)
    [30, 50000, 20000, 24, 700, "female", "no"],     # approved
    [45, 80000, 30000, 36, 750, "male", "yes"],      # approved
    [23, 35000, 10000, 6, 620, "female", "no"],      # denied
    [27, 60000, 25000, 18, 670, "male", "yes"],      # approved
    [24, 50000, 15000, 12, 640, "female", "no"],     # denied
    [29, 90000, 30000, 30, 700, "male", "yes"],      # approved
]

# Generate labels based on rule
y = []
for sample in X:
    age, _, _, _, credit_score, _, _ = sample
    if age < 25 or credit_score < 650:
        y.append("denied")
    else:
        y.append("approved")

# Encode categorical features
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_approved = LabelEncoder()

genders = [row[5] for row in X]
married_statuses = [row[6] for row in X]

gender_encoded = le_gender.fit_transform(genders)
married_encoded = le_married.fit_transform(married_statuses)
approved_encoded = le_approved.fit_transform(y)

# Replace gender and married in X with encoded values
for i in range(len(X)):
    X[i][5] = gender_encoded[i]
    X[i][6] = married_encoded[i]

X = np.array(X, dtype=float)

# Train model
model = RandomForestClassifier()
model.fit(X, approved_encoded)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_gender, "encoder_gender.pkl")
joblib.dump(le_married, "encoder_married.pkl")
joblib.dump(le_approved, "encoder_approved.pkl")

print("✅ Model and encoders saved.")
