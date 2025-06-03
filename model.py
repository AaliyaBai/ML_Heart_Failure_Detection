import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle, os

def load_data():
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    return df

def train_model(df): 
    df_copy = df.copy()
    X = df_copy.drop('DEATH_EVENT', axis=1)
    y = df_copy['DEATH_EVENT']
    
    #handling missing values replacing with mean of each column
    imputer = SimpleImputer(strategy='mean')
    X_imputer = imputer.fit_transform(X)

    sc = StandardScaler()
    X_Scaled = sc.fit_transform(X_imputer)

    X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, test_size=0.2, random_state=42)

    #training model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    #model accuracy

    y_pred_test = model.predict(X_test)
    accuracy = model.score(X_test, y_test) # checking model prediction accuracy
    class_report = classification_report(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    return model, accuracy, class_report, conf_matrix

dataset = load_data()
trained_data = train_model(dataset)

#save model to a file
with open('RF_Model_Data.pkl', 'wb') as f:
    pickle.dump(trained_data, f)