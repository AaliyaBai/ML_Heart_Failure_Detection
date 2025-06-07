# 💓 ML Heart Failure Detection
## Overview
### **Heart failure risk prediction using Python and Streamlit.**

This project provides an interactive web application to predict the risk of heart failure in patients based on clinical data. It is designed to help healthcare professionals and users make informed decisions about patient care.

---

## 🎯 Project Summary
### Aim:
The goal of this project is to develop a machine learning model that accurately predicts the risk of heart failure in patients based on clinical parameters. This tool is designed to assist medical professionals in early detection and intervention.

### Approach:
Using a dataset of 299 patient records, we performed exploratory data analysis to understand key risk indicators. A Random Forest Classifier was trained after preprocessing the data. The model is deployed via a Streamlit web app that enables users to input patient data and receive a real-time risk assessment.

 **Key Observations:**

- Mortality is significantly higher among patients with:

- Low ejection fraction

- High serum creatinine

- Older age

- Anaemia and high blood pressure (mild correlation)

- These insights helped shape feature selection and model design.

**Final Inference:**

The Random Forest model demonstrates solid predictive performance and offers interpretability through feature importance and performance metrics. The Streamlit app enhances usability, allowing fast, informed decision-making for both clinicians and researchers.

## 🚀 Technical Stack
Data Processing & Machine Learning:
Python, NumPy, Pandas, Scikit-learn

- Web Application:
Streamlit

- Model Serialization:
Pickle

- Visualization:
Streamlit widgets & output tables

- Dataset:
Heart Failure Clinical Records Dataset

### 📊 Exploratory Data Analysis (EDA)
The EDA was performed in the Jupyter Notebook: ML-HeartFailure_EDA.ipynb

**Key insights:**

- Age: Older patients tend to have higher mortality rates.

- Ejection Fraction: Lower ejection fraction is strongly associated with increased risk of death.

- Serum Creatinine: Elevated serum creatinine levels indicate kidney dysfunction, correlating with higher risk.

- High Blood Pressure: Patients with high blood pressure show a slightly higher risk.

- Anaemia: Presence of anaemia is more common in deceased patients.

- Correlation Matrix: Shows relationships between clinical features; ejection fraction and serum creatinine are key drivers of prediction.

- Visuals Used: Histograms, KDE plots, bar plots, pair plots, and heatmaps for feature comparison and correlation.

## 📈 Workflow
 Data Ingestion: Load clinical records from heart_failure_clinical_records_dataset.csv (299 patients).

 Data Preparation: Clean and preprocess data (correct feature types, check nulls, encode categorical variables).

## Exploratory Data Analysis (EDA):
Analyze trends in mortality based on features like:

- Ejection fraction

- Serum creatinine

- Age

- High blood pressure

- Anaemia

...and more

## Model Training:

Train a Random Forest Classifier

Split data into train and test sets

Generate performance metrics (accuracy, classification report, confusion matrix)

Save the model as RF_Model_Data.pkl

## Streamlit App Development:
The main script (app.py) allows users to:

Input 12 clinical parameters via sliders/dropdowns

View real-time predictions (“high” or “low” risk)

See prediction probabilities

Explore model accuracy, classification report, and confusion matrix

## 🔬 Prediction Model
Algorithm: Random Forest Classifier

Accuracy: Displayed in the app (based on test data)

Model Insights: Classification report & confusion matrix

Prediction Categories:

- 0: Low Risk (Survived)

- 1: High Risk (Deceased)
