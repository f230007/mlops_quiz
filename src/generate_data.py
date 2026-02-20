import os
import pandas as pd
from sklearn.datasets import load_iris

print("Loading Iris dataset...")

# Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Create raw data folder if not exists
os.makedirs("data/raw", exist_ok=True)

# Save CSV
df.to_csv("data/raw/data.csv", index=False)

print("Iris dataset saved to data/raw/data.csv")
print("Dataset shape:", df.shape)