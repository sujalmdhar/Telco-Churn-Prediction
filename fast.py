import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model, scaler, and label encoder
model = joblib.load('telco_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit app title
st.title("Telco Customer Churn Prediction")

# Sidebar for user inputs
st.sidebar.header("Customer Details")

# Numerical features
tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.sidebar.slider("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Categorical features
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Predict button
if st.sidebar.button("Predict"):
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
    st.subheader("Prediction Result")
    if prediction_label[0] == "Yes":
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")

    # Show prediction probabilities
    prediction_proba = model.predict_proba(input_data_scaled)
    st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data_scaled)
    st.subheader("SHAP Explanation")
    st.write("The SHAP values below show how each feature contributed to the prediction.")
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

    # Interactive Plotly Visualization
    st.subheader("Interactive Feature Analysis")
    feature_importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", ascending=False)

    fig = px.bar(feature_importance_df, x="SHAP Value", y="Feature", orientation="h", title="Feature Importance (SHAP Values)")
    st.plotly_chart(fig)

# Add some additional information or visualizations
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app predicts customer churn for a telecom company based on customer details.")
