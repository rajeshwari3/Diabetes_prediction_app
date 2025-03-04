# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:32:35 2025

@author: RAJESHWARI
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('trained_model.sav','rb'))


#creating function for prediction

def diabetic_prediction(input_data):   #it is a list so we need to change it into numpy array

    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array because we are giving only 1 row as input

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


    prediction=loaded_model.predict(input_data_reshaped)         #replaced classifier with loaded_model
    print("Prediction: " ,prediction)                 #prediction is a list so it we represent it with [0]
    if(prediction[0]==0):
      return "The person is not diabetic"

    else:
      return "The person is diabetic"
    
def main():
    #Giving a title
    st.title("Diabetes Prediction Web App")
    
    #Inputs

    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BP value')
    SkinThickness=st.text_input('Skin Thickness')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('DP Function Value')
    Age=st.text_input('Age of the Person')
    
    #code for prediction
    
    diagnosis=''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetic_prediction([Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)

if __name__=='__main__':
    main()    
