import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# Load the model
model = joblib.load("model.pkl")

# Title
st.title("🏡 House Price Prediction App")

# User inputs
bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, step=0.5)
bedrooms = st.number_input("Number of Bedrooms", min_value=1)
living_area = st.number_input("Living Area (in sqft)", min_value=100)
views = st.number_input("Number of Views", min_value=0)
grade = st.number_input("Grade of the House", min_value=1)
area_excl_basement = st.number_input("Area (excluding basement)", min_value=100)
basement_area = st.number_input("Basement Area (in sqft)", min_value=0)

# Prediction button
if st.button("Predict Price"):
    # Prepare input for model
    input_features = np.array([[bathrooms, bedrooms, living_area, views,
                                grade, area_excl_basement, basement_area]])
    
    # Predict
    prediction = model.predict(input_features)
    
    # Output
    st.success(f"🏷️ Predicted House Price: ₹{prediction[0]:,.2f}")

