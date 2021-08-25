# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 02:15:12 2021

@author: D007
"""

from pydantic import BaseModel

class Titanic(BaseModel):
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    embarked_C: int
    embarked_Q: int
    embarked_S: int 
    sex_female: int 
    sex_male: int 
    pclass_1: int
    pclass_2: int 
    pclass_3: int
    
    
    
    
    
    
    
    
    
    