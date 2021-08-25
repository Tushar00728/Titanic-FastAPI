# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 02:34:19 2021

@author: D007
"""

import uvicorn
from fastapi import FastAPI

from Tiydant import Titanic
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    
    return {'message': 'Titanic Survival predictor'}

@app.get('/{name}')
def get_name(name:str):
    return {'Welcome to this app': f'{name}'}

@app.post('/predict') #Expose the prediction functionality, make a prediction from the JSON data
def predict_titanic(data:Titanic):
    data = data.dict()
    Age = data['Age']
    SibSp = data['SibSp']
    Parch = data['Parch']
    Fare = data['Fare']
    embarked_C = data['embarked_C']
    embarked_Q = data['embarked_Q']
    embarked_S = data['embarked_S']
    sex_female= data['sex_female']
    sex_male = data['sex_male']
    pclass_1 = data['pclass_1']
    pclass_2 = data['pclass_2']
    pclass_3 = data['pclass_3']
     
    prediction = classifier.predict([[Age, SibSp,Parch, Fare, embarked_C, embarked_Q,
            embarked_S, sex_female, sex_male, pclass_1,pclass_2, pclass_3]])

    if(prediction[0] == 1 ):
        prediction = "Survived"
    else:
        predicton = "Not Survived"
    
    return {
        'prediction': prediction
        }

if __name__=='__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)
    
        
