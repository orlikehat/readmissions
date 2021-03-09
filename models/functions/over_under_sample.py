# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:20:25 2021

@author: orlyk
"""
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def run_over_under_sample(x_train,y_train):
    filtering_params=pd.read_csv(r"C:\Users\orlyk\readmissions\project\git_code\readmissions\models\functions\filtering_params.csv")
    filtering_params=dict(zip(list(filtering_params.condition), list(filtering_params.value)))
    
    if filtering_params['under_sample']=="1":
        print("run under-sample; original size:")
        print(Counter(y_train))
        if x_train.isnull().values.any():
            print("impute before down-sampling")
            
        else:
            undersample = RandomUnderSampler(sampling_strategy=0.3)
            x_train, y_train = undersample.fit_resample(x_train, y_train)
            print("under-sampled size:")
            print(Counter(y_train))


    if filtering_params['over_sample']=="1":
        print("run SMOTE; original size:")
        counter = Counter(y_train)
        print(counter)
        if x_train.isnull().values.any():
            print("impute before SMOTE")
            
        else:
            oversample = SMOTE(sampling_strategy=0.4)
            x_train, y_train = oversample.fit_resample(x_train, y_train)
            print("over-sampled size:")
            counter = Counter(y_train)
            print(counter)
            
            
    return x_train, y_train
        
        
       