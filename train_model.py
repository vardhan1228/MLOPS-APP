from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Original data: [age, income, loan_amount, loan_term, credit_score, gender, married]
raw_X = [
    [22, 40000, 15000, 12, 600, "male", "no"],     # denied
    [30, 50000, 20000, 24, 700, "female", "no"],   # approved
    [45, 80000, 30000, 36, 750, "male", "yes"],    # approved
    [23, 35000, 10000, 6, 620, "female", "no"],    # denied
    [27, 60000, 25000, 18, 670, "male", "yes"],    # approved
    [24, 50000, 15000, 12, 640, "female", "no"],   # denied
    [29, 90000, 30000, 30, 700, "male", "yes"],    # approved
]

# Generate target labels based on your rule
y = []
for sample in raw_X:
    age, income, _, _, credit_score, _, _ = sample
    if age < 25 or credit_score < 600 or income < 10000:
        y.append("denied")
    else:
        y.append("approved")

# Extract only the features: age, income, credit_score
X = [[row[0], row[1], row[4]] for row in raw_X]

# Encode the output labels only (approved/denied)
from sklearn.preprocessing import LabelEncoder
le_approved = LabelEncoder()
y_encoded = le_approved.fit_transform(y)

# Convert X to numpy array
X = np.array(X, dtype=float)

# Train the model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save model and output label encoder
joblib.dump(model, "model.pkl")
joblib.dump(le_approved, "encoder_approved.pkl")

print("✅ Model trained with age, income, credit_score only and saved.")
