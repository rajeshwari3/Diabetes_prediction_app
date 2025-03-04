# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

#loading the saved model

loaded_model=pickle.load(open('D:/Model_deploy/trained_model.sav','rb'))

input_data=(10,168,74,0,0,38,0.537,34)   #it is a list so we need to change it into numpy array

input_data_as_numpy_array=np.asarray(input_data)

#reshape the array because we are giving only 1 row as input

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


prediction=loaded_model.predict(input_data_reshaped)         #replaced classifier with loaded_model
print("Prediction: " ,prediction)                 #prediction is a list so it we represent it with [0]
if(prediction[0]==0):
  print("The person is not diabetic")

else:
  print("The person is diabetic")
