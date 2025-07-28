from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# Extended dataset with low-income denied examples
raw_X = [
    [22, 40000, 15000, 12, 600, "male", "no"],     # denied
    [30, 50000, 20000, 24, 700, "female", "no"],   # approved
    [45, 80000, 30000, 36, 750, "male", "yes"],    # approved
    [23, 35000, 10000, 6, 620, "female", "no"],    # denied
    [27, 60000, 25000, 18, 670, "male", "yes"],    # approved
    [24, 50000, 15000, 12, 640, "female", "no"],   # denied
    [29, 90000, 30000, 30, 700, "male", "yes"],    # approved
    # Added low-income cases
    [28, 5000, 1000, 6, 700, "male", "no"],        # denied
    [26, 8000, 2000, 12, 690, "female", "yes"],    # denied
    [40, 7000, 3000, 24, 710, "female", "no"],     # denied
]

# Generate labels based on rule
y = []
for sample in raw_X:
    age, income, _, _, credit_score, _, _ = sample
    if age < 25 or credit_score < 600 or income < 10000:
        y.append("denied")
    else:
        y.append("approved")

# Encode categorical features
genders = [row[5] for row in raw_X]
married_statuses = [row[6] for row in raw_X]

le_gender = LabelEncoder()
le_married = LabelEncoder()

gender_encoded = le_gender.fit_transform(genders)
married_encoded = le_married.fit_transform(married_statuses)

# Build feature matrix
X = []
for i in range(len(raw_X)):
    row = raw_X[i]
    X.append([
        row[0],             # age
        row[1],             # income
        row[2],             # loan_amount
        row[3],             # loan_term
        row[4],             # credit_score
        gender_encoded[i],
        married_encoded[i],
    ])

X = np.array(X, dtype=float)

# Encode target
le_approved = LabelEncoder()
y_encoded = le_approved.fit_transform(y)

# Train the model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save model and encoders
joblib.dump(model, "model_full.pkl")
joblib.dump(le_gender, "encoder_gender.pkl")
joblib.dump(le_married, "encoder_married.pkl")
joblib.dump(le_approved, "encoder_approved.pkl")

print("✅ Model retrained with low-income examples. Saved.")
