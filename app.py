import streamlit as st
import pandas as pd
import joblib
import json

# Load the saved components
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('logistic_regression_model.pkl')
label_enc_y = joblib.load('label_encoder_y.pkl')

# Load the feature order from JSON
with open('feature_order.json', 'r') as f:
    feature_order = json.load(f)

# Streamlit app title
st.title("Employee Attrition Prediction")

# Create input fields for each feature
st.header("Input Employee Features")
input_data = {}

# Dynamically create input fields based on feature_order
for feature in feature_order:
    if feature in label_encoders:  # Categorical feature
        options = list(label_encoders[feature].classes_)
        input_data[feature] = st.selectbox(f"{feature}", options)
    else:  # Numerical feature
        input_data[feature] = st.number_input(f"{feature}", value=0)

# Button to make prediction
if st.button("Predict Attrition"):
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
    
    # Ensure columns are in the correct order
    input_df = input_df[feature_order]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of attrition (class 1)
    
    # Interpret the prediction
    attrition = label_enc_y.inverse_transform([prediction])[0]
    
    # Display the result
    st.subheader("Prediction Result")
    st.write(f"**Attrition**: {attrition}")
    st.write(f"**Probability of Attrition**: {probability:.2f}")