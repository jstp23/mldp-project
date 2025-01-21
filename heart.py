# Web application for prediction
import streamlit as st
import pandas as pd
import numpy as np
import joblib


model_path = 'heart_model.pkl'
best_model = joblib.load(model_path)

st.write("""
         # Heart Disease Prediction App
         This app predicts the **Heart Disease**!
         """)

st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 29)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 94)
    chol = st.sidebar.slider('Cholesterol', 126, 564, 126)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 71)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 0.0)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Flourosopy', 0, 3, 0)
    cp = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    thal = st.sidebar.selectbox('Thal', ['Normal', 'Fixed Defect', 'Reversable Defect'])
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

    sex = 1 if sex == 'Male' else 0
    exang = 1 if exang == 'Yes' else 0
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_map[cp]
    thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversable Defect': 2}
    thal = thal_map[thal]
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope = slope_map[slope]

    # Create input features array
    data = {
        'age': age,
        'sex': sex, 
        'trestbps': trestbps, 
        'chol': chol, 
        'thalach': thalach, 
        'exang': exang, 
        'oldpeak': oldpeak,
        'ca': ca,
        'cp_0': 1 if cp == 0 else 0,
        'cp_1': 1 if cp == 1 else 0,
        'cp_2': 1 if cp == 2 else 0,
        'cp_3': 1 if cp == 3 else 0,    
        'thal_0': 1 if thal == 0 else 0,
        'thal_1': 1 if thal == 1 else 0,
        'thal_2': 1 if thal == 2 else 0,
        'thal_3': 1 if thal == 3 else 0,
        'slope_0': 1 if slope == 0 else 0,
        'slope_1': 1 if slope == 1 else 0,
        'slope_2': 1 if slope == 2 else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Handle user input
input_df = user_input_features()

# Display user input features
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
prediction = best_model.predict(input_df)
prediction_proba = best_model.predict_proba(input_df)

# Display results
st.subheader('Prediction')
st.write('This person has heart disease.' if prediction[0] == 1 else 'This person does not have heart disease.')

st.subheader('Prediction Probability')
st.write(prediction_proba)