# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:30:55 2021

@author: saniy
"""
import streamlit as st


import pandas as pd
import numpy as np
import pickle

st.write("""
         # Heart Rate Prediction App
""")


def user_input_features():
        age = st.sidebar.text_input("Enter the age",29,2)
        sex = st.sidebar.selectbox('Sex',('1','0'))
        cp = st.sidebar.selectbox('Chest Pain type',('0','1','2','3'))
        trestbps = st.sidebar.text_input("Resting blood pressure",100)
        chol = st.sidebar.slider('Serum cholestoral in mg/dl', 100,600,240)
        fbs = st.sidebar.selectbox('Fasting blood sugar > 120 mg/dl',('0','1'))
        restecg = st.sidebar.selectbox('Resting electrocardiographic results',('0','1','2'))                           
        thalach = st.sidebar.text_input("Mazimum heart rate achieved",72)
        exang = st.sidebar.selectbox('Exercise induced angina', ('0','1'))
        oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest',0,8)
        slope = st.sidebar.selectbox('The slope of the peak exercise ST segment',('0','1','2'))
        ca = st.sidebar.selectbox('Number of major vessels (0-3) colored by flourosopy', ('0','1','2','3'))
        thal = st.sidebar.selectbox('Thalassemia: 0 = normal; 1 = fixed defect; 2 = reversable defect', ('0','1','2'))
        
        data = {'age': age,
                'sex':sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach' : thalach,
                'exang' : exang,
                'oldpeak' : oldpeak,
                'slope' : slope,
                'ca' : ca,
                'thal' : thal}
        
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

@st.cache
def load_data():
    
    heart_raw = pd.read_csv('heart.csv')
    heart = heart_raw.drop(columns=['target'])
    return heart

df = pd.concat([input_df,load_data()],axis=0)
df = df[:1]
st.write('In the below dataframe the row has the values you input from the sidebar')
st.write(df)

load_clf = pickle.load(open('heart_clf.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
attack = np.array(['0','1'])
st.write(attack[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
    
