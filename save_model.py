# save_model.py
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("mlproject1.csv")

# Keep only required columns
X = df[
    [
        "number of bathrooms",
        "number of bedrooms",
        "living area",
        "number of views",
        "grade of the house",
        "Area of the house(excluding basement)",
        "Area of the basement"
    ]
]

y = df["Price"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = GradientBoostingRegressor()
model.fit(x_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved successfully!")
