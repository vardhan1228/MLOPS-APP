from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

X = [
    [25, 50000, 20000, 12, 700, "male", "yes"],
    [35, 70000, 30000, 24, 650, "female", "no"],
    [45, 90000, 50000, 36, 800, "male", "yes"],
    [30, 40000, 15000, 6, 600, "female", "no"],
    [50, 100000, 40000, 48, 750, "male", "yes"]
]

y = ["approved", "denied", "approved", "denied", "approved"]

le_gender = LabelEncoder()
le_married = LabelEncoder()
le_approved = LabelEncoder()

for i in range(len(X)):
    X[i][5] = le_gender.fit_transform([row[5] for row in X])[i]
    X[i][6] = le_married.fit_transform([row[6] for row in X])[i]

y_encoded = le_approved.fit_transform(y)
X = np.array(X, dtype=float)

model = RandomForestClassifier()
model.fit(X, y_encoded)

joblib.dump(model, "model.pkl")
joblib.dump(le_gender, "encoder_gender.pkl")
joblib.dump(le_married, "encoder_married.pkl")
joblib.dump(le_approved, "encoder_approved.pkl")
print("✅ Model and encoders saved.")
