# 💓 ML Heart Failure Detection

## Overview
**Heart failure risk prediction using Python and Streamlit.**

This project provides an interactive web application to predict the risk of heart failure in patients based on clinical data. It is designed to help healthcare professionals and users make informed decisions about patient care.

---

## 🚀 Technical Stack

- **Data Processing & Machine Learning:**  
  Python, NumPy, Pandas, Scikit-learn

- **Web Application:**  
  Streamlit

- **Model Serialization:**  
  Pickle

- **Visualization:**  
  Streamlit widgets & output tables

- **Dataset:**  
  Heart Failure Clinical Records Dataset

---

## 📈 Workflow

1. **Data Ingestion:**  
   Load clinical records from `heart_failure_clinical_records_dataset.csv` (299 patients).

2. **Data Preparation:**  
   Clean and preprocess data (correct feature types, check nulls, encode categorical variables).

3. **Exploratory Data Analysis (EDA):**  
   Analyze trends in mortality based on features like:
   - Ejection fraction
   - Serum creatinine
   - Age
   - High blood pressure
   - Anaemia
   - ...and more

4. **Model Training:**  
   - Train a Random Forest Classifier
   - Split data into train and test sets
   - Tune hyperparameters
   - Generate performance metrics (accuracy, classification report, confusion matrix)
   - Save the model as `RF_Model_Data.pkl`

5. **Streamlit App Development:**  
   The main script (`app.py`) allows users to:
   - Input 12 clinical parameters via sliders/dropdowns
   - View real-time predictions (“high” or “low” risk)
   - See prediction probabilities
   - Explore model accuracy, classification report, and confusion matrix

---

## 🔬 Prediction Model

- **Algorithm:** Random Forest Classifier
- **Accuracy:** Displayed in the app (based on test data)
- **Model Insights:** Classification report & confusion matrix

**Prediction Categories:**
- `0`: Low

