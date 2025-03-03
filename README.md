# Telco Customer Churn Prediction

Telco Customer Churn Prediction Web App Interface - https://telco-churn-prediction-0929.streamlit.app/

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation Instructions](#installation-instructions)
- [Files](#files)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Reproduction Instructions](#reproduction-instructions)
- [Results and Interpretation](#results-and-interpretation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer churn, which occurs when a customer discontinues a service, is a significant problem in the telecom industry. This project applies machine learning techniques to predict customer churn using various customer demographics and subscription data. The main aim is to provide a model that can predict churn so that businesses can proactively address customer retention. 

## Dataset
The dataset used in this project is the Telco Customer Churn dataset, available on Kaggle. It contains the following columns:
- **customerID**: Unique identifier for each customer
- **gender**: Gender of the customer
- **SeniorCitizen**: Whether the customer is a senior citizen (1, 0)
- **Partner**: Whether the customer has a partner (Yes, No)
- **Dependents**: Whether the customer has dependents (Yes, No)
- **tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether the customer has phone service (Yes, No)
- **MultipleLines**: Whether the customer has multiple lines (Yes, No)
- **InternetService**: Type of internet service (DSL, Fiber optic, No)
- **OnlineSecurity**: Whether the customer has online security (Yes, No)
- **TechSupport**: Whether the customer has tech support (Yes, No)
- **Contract**: Type of contract (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether the customer uses paperless billing (Yes, No)
- **PaymentMethod**: Payment method used (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- **MonthlyCharges**: Monthly charges to the customer
- **TotalCharges**: Total charges to the customer
- **Churn**: Whether the customer churned (Yes, No)

## Installation Instructions

### a. Clone the repository:
git clone [https://github.com/sujalmdhar/Telco-Churn-Prediction]

### b. Install dependencies:
Install required packages to run this code.

### c. Requirements
- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- lime
- shap
- joblib

## Files
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: CSV file.
- `Telco_Churn_Prediction.ipynb`: Jupyter notebook with the analysis and model development.
- `telco_churn_prediction.py`: Python script for the same analysis.
- `telco_churn_model.pkl`: The trained Random Forest model.
- `scaler.pkl`: File contains the mean and standard deviation of each feature from the training data, used for scaling.
- `label_encoder`: File contains the mapping between original string labels and numerical labels, used for encoding the target variable.
- `lime_explanation`: Customer index 5,10, & 15.
- `telco_report.pdf`: A detailed report of the analysis, methodology, and results.

## Model Development

### 1. Data Preprocessing:
- Missing values were handled by imputing the median of "TotalCharges" for missing values.
- Categorical features were one-hot encoded, and the target variable "Churn" was label encoded.
- Numerical features were scaled using StandardScaler.

### 2. Model Selection: Multiple models were trained, including:
- Logistic Regression
- Ridge Classifier
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
    
### 3. Hyperparameter Tuning: 
Hyperparameters for the Random Forest model were optimized using GridSearchCV with cross-validation.

### 4.Evaluation Metrics:
Accuracy, precision, recall, F1-score, and AUC-ROC were calculated to evaluate model performance.

## Model Evaluation
The models were evaluated using cross-validation (5-fold stratified), and the final model was selected based on the best performance. Random Forest emerged as the best model, achieving the highest accuracy and AUC-ROC score.

## Model Interpretation and Explainability
- Local Interpretable Model-agnostic Explanations (LIME): Local Interpretable Model-agnostic Explanations was used to understand individual predictions made by the Random Forest model.
- Permutation Feature Importance: Permutation feature importance was used to assess the global importance of features.
- Feature Importance Analysis: A DataFrame is created to store the importance of each feature based on permutation importance. The features are sorted in descending order of their importance values to highlight the most significant predictors.
- Partial Dependence Plots (PDP): Partial Dependence Plots were generated to visualize the marginal effect of specific features (tenure, MonthlyCharges, TotalCharges) on the predicted probability of churn. PDPs show how the model's predictions change as the value of a feature varies, holding other features constant.
- Individual Conditional Expectation (ICE): Individual Conditional Expectation (ICE) plots were created to show how the predictions for individual instances change as a feature varies.
  
## Reproduction Instructions
- Clone this repository and install the required dependencies.
- Visit Kaggle Telco Customer Churn Dataset and download the dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) or download it from my repository.
- Run the Jupyter notebook or the Python script to reproduce the analysis.

## Results and Interpretation
- The final model used for churn prediction was Random Forest, which showed the best performance in terms of accuracy and AUC-ROC.
- The feature importance analysis indicated that "MonthlyCharges," "tenure," and "Contract" were the most important factors influencing customer churn.
- LIME and PDP were used to interpret the individual predictions, providing insights into the key features driving churn.
- Customer Segmentation through clustering revealed three distinct customer groups with different churn risks, helping businesses tailor retention strategies.
- SMOTE effectively addressed class imbalance, leading to improved performance in predicting churn.

## Contributing
Feel free to open issues, submit pull requests, or contribute in any other way. If you want to add features or improve the project, please ensure that you follow the guidelines.

## License
This project is licensed under the MIT License.
