# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:11:45 2021

@author: orlyk
"""

import pandas as pd
from sklearn.impute import SimpleImputer


def run_impute(x_train):
    filtering_params=pd.read_csv(r"C:\Users\orlyk\readmissions\project\git_code\readmissions\models\functions\filtering_params.csv")
    filtering_params=dict(zip(list(filtering_params.condition), list(filtering_params.value)))
    
    if filtering_params['impute']=="simple":
        print("simple imputation")
        x_train=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(x_train), columns = x_train.columns)
        
    else:
        print("no imputation")
        
    return x_train
