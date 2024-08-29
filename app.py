
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib  # For loading the scaler

# Load the trained model and scaler
model = load_model('/content/mandates_prediction_model.h5')
scaler = joblib.load('/content/standard_scaler.pkl')

# Streamlit app
st.title("Election Final Mandates Prediction")

# Input fields for each feature
totalVoters = st.number_input("Enter total voters", value=0)
votersPercentage = st.number_input("Enter voters percentage", value=0.0)
Votes = st.number_input("Enter votes", value=0)
Hondt = st.number_input("Enter Hondt", value=0)
Percentage = st.number_input("Enter percentage", value=0.0)

# Organize the inputs into a DataFrame
input_data = np.array([[totalVoters, votersPercentage, Votes, Hondt, Percentage]])
input_scaled = scaler.transform(input_data)  # Apply the same scaling used during training

# Predict the final mandates
prediction = model.predict(input_scaled)

# Display the result
st.write(f"Predicted Final Mandates: {prediction[0][0]:.2f}")
