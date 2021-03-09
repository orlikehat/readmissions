# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:02:56 2021

@author: orlyk
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_scaler(x_train,x_test):
    filtering_params=pd.read_csv(r"C:\Users\orlyk\readmissions\project\git_code\readmissions\models\functions\filtering_params.csv")
    filtering_params=dict(zip(list(filtering_params.condition), list(filtering_params.value)))
   
    if filtering_params['scale']=="standard":
        print("run standard scaler")
        scaler = StandardScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        
    else:
        print("no scaling")
        
        
    return x_train, x_test
        
        
    