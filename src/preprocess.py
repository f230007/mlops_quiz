import pandas as pd
import os
from sklearn.model_selection import train_test_split
print("Loading dataset...")

# Load CSV
data = pd.read_csv("data/raw/data.csv")

# Handle missing values (for automation safety)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create processed directory
os.makedirs("data/processed", exist_ok=True)

# Save processed files
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing completed.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])