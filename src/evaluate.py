import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

print("Evaluating model...")

# Load model
model = joblib.load("models/model.pkl")

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Final Accuracy:", accuracy)

# Save results
os.makedirs("results", exist_ok=True)

with open("results/metrics.txt", "w") as f:
    f.write(f"Final Accuracy: {accuracy}")

print("Results saved in results/metrics.txt")