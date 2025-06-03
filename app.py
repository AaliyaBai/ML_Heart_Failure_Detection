import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

model, model_accuracy, class_report, conf_matrix = pickle.load(open('RF_Model_Data.pkl', 'rb'))

st.title("🫀Heart Failure Risk Prediction")

# Input form
st.sidebar.header("**PATIENT CLINICAL MEASUREMENTS**")
st.sidebar.write("Fill the details below to predict the risk of heart failure.")
st.sidebar.markdown("---")

#setting clinical measurements and structure in strreamlit
age = st.sidebar.slider("Age", 18, 100, 50, 1)
anaemia = st.sidebar.selectbox("Anaemia (Low Red Blood Cells)?", ["No", "Yes"])
creatinine_phosphokinase = st.sidebar.number_input("Creatinine Phosphokinase (mcg/L)", value=200)
diabetes = st.sidebar.selectbox("Diabetes?", ["No", "Yes"])
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 10, 80, 35)
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure?", ["No", "Yes"])
platelets = st.sidebar.number_input("Platelets (kiloplatelets/mL)", min_value=0, max_value=1000000)
serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.0, 10.0)
serum_sodium = st.sidebar.number_input("Serum Sodium (mEq/L)", value = 200)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
smoking = st.sidebar.selectbox("Smoking?", ["No", "Yes"])
time = st.sidebar.slider("Follow-up Time (days)", 1, 300, 120)

#mapping the string values
sex_map  = {"Female":0, "Male":0}
anaemia_map = {"No":0,"Yes":1}
diabetes_map = {"No":0,"Yes":1}
high_blood_pressure_map = {"No":0,"Yes":1}
smoking_map = {"No":0,"Yes":1}

#function for "Predict Heart Failure" button
if st.sidebar.button("Predict Heart Failure"):
    #create input array
    user_input = np.array([[age, anaemia_map[anaemia], creatinine_phosphokinase, diabetes_map[diabetes], ejection_fraction, 
                            high_blood_pressure_map[high_blood_pressure], platelets, serum_creatinine, serum_sodium, 
                            sex_map[sex], smoking_map[smoking], time]])
    
    prediction = model.predict(user_input) # predicting if the patient his likely to have heart failure or not

    prediction_probability = model.predict_proba(user_input) # checking model prediction probability
    
   
    st.header("**Prediction**")
    if prediction[0] == 1:
        st.error("⚠️ The patient is at **risk of heart failure**.")
    else:
        st.success("✅ The patient is **not likely** to suffer heart failure.")

    st.header("**Prediction Probability**")
    probability_df = pd.DataFrame({
                    'Risk Category': ['Low Risk(Survived)', 'High Risk(Deceased)'],
                    'Probability': [f"{prediction_probability[0,0] * 100:.2f}%", f"{prediction_probability[0,1] * 100:.2f}%"] 
                                    })
    st.table(probability_df)

    
st.markdown("---")
expander = st.expander("Show Model Performance on Test Set")
with expander:
    st.write(f"**Model Accuracy:** {model_accuracy:.4f}")
    st.text("Classification Report:")
    st.text(class_report) 
    st.text("Confusion Matrix:")
    st.text(str(conf_matrix)) 
    st.text(f"prediction accuracy for a patient is likely to get heart failure: {model_accuracy:.4f}")