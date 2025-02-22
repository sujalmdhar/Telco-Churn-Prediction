import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model, scaler, and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('telco_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    .footer a {
        color: #3498db;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app title
st.title("📞 Telco Customer Churn Prediction")
st.markdown("Predict customer churn for a telecom company based on customer details.")

# Input fields for user to enter data
st.sidebar.header("📋 Customer Details")

# Use columns to organize input fields
col1, col2 = st.sidebar.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

# Additional service details
st.sidebar.markdown("### 🔧 Service Details")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Predict button
if st.sidebar.button("🚀 Predict Churn"):
    # Create a dictionary with the input data
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method
    }

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)

    # Ensure all expected features are present
    expected_features = scaler.feature_names_in_
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing features with default value 0

    # Reorder columns to match the scaler's expected input
    input_df = input_df[expected_features]

    # Scale the input data
    input_data_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)

    # Display result
    st.subheader("📊 Prediction Result")
    if prediction_label[0] == "Yes":
        st.error("🚨 This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

    # Show prediction probabilities
    prediction_proba = model.predict_proba(input_data_scaled)
    st.write(f"📈 Probability of Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"📉 Probability of Not Churning: {prediction_proba[0][0]:.2f}")

    # Visualize probabilities
    st.progress(prediction_proba[0][1])

# Add some additional information or visualizations
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("This app predicts customer churn for a telecom company based on customer details.")

# Advanced Footer
st.markdown(
    """
    <div class="footer">
        <p>Created by <a href="https://github.com/sujalmdhar" target="_blank">Sujal Manandhar</a> | Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
