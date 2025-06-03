**Project: ML Heart Failure Detection**

Heart failure risk prediction using Python and Streamlit.

This project aims to provide an interactive web application that predicts the risk of heart failure in patients based on clinical data. It helps healthcare professionals or users make informed decisions by analyzing risk factors using a trained machine learning model.

🔧 Technical Stack
Data Processing & Machine Learning: Python (NumPy, Pandas, Scikit-learn)

Web Application: Streamlit

Model Serialization: Pickle

Visualization: Streamlit widgets and output tables

Dataset: Heart Failure Clinical Records Dataset

📈 Workflow
1. Data Ingestion
Clinical records were loaded from heart_failure_clinical_records_dataset.csv. The dataset includes demographic and health-related features for 299 patients.

2. Data Preparation
Data was cleaned and preprocessed using Pandas. Feature types were corrected, nulls checked, and categorical variables encoded for machine learning.

3. Exploratory Data Analysis (EDA)
EDA was conducted (not included in app.py, but assumed in the training pipeline) to understand trends in mortality based on features such as:

Ejection fraction

Serum creatinine

Age

High blood pressure

Anaemia, etc.

4. Model Training
A Random Forest Classifier was trained using Scikit-learn. Key steps:

Train-test split

Model tuning

Performance metrics generation

The model, along with its performance metrics (accuracy, classification report, and confusion matrix), was saved using pickle in RF_Model_Data.pkl.

5. Streamlit App Development
app.py is the main script for the web interface. It allows users to:

Input 12 clinical parameters via sliders or dropdowns

View real-time predictions (high/low risk)

See prediction probabilities

Explore model accuracy, classification report, and confusion matrix in an expandable section

🔬 Prediction Model
Algorithm: Random Forest Classifier

Accuracy: Provided in the app (based on test data)

Model Insights: Includes classification report and confusion matrix

Prediction Categories:

0: Low Risk (Survived)

1: High Risk (Deceased)

📊 App Output
User Inputs
12 clinical features:

Age

Anaemia

Creatinine Phosphokinase

Diabetes

Ejection Fraction

High Blood Pressure

Platelets

Serum Creatinine

Serum Sodium

Sex

Smoking

Follow-up Time

Model Output
Risk Prediction: "At risk" or "Not likely to suffer heart failure"

Probability Table: Breakdown of prediction confidence

Model Metrics: Accuracy, classification report, confusion matrix
