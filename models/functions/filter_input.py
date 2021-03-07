# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:50:44 2021

@author: orlyk
"""


import pandas as pd




def filter_input(df):
    filtering_params=pd.read_csv(r"C:\Users\orlyk\readmissions\project\code\models\functions\filtering_params.csv")
    filtering_params=dict(zip(list(filtering_params.condition), list(filtering_params.value)))

    
    print("original_shape: " + str(df.shape))
    
    #only internal
    if filtering_params['IS_INTERNAL_ONLY']=='True':
        df=df[(df["dept_cat_disch_internal"]==1) | (df["dept_cat_disch_internal_ICU"]==1)]# |(df["dept_cat_disch_geriatrics"]==1)]
        print("internal_shape: " + str(df.shape))
    
     #year
    df=df[(df.year>=float(filtering_params["TH_year"]))]
    
    #threshold patients
    df=df.dropna(thresh=float(filtering_params["TH_patients"])*(df.shape[1]), axis=0)
    
    if filtering_params["ONLY_IS_CHEM"]=='True':
        df=df.dropna(subset=['LABS_BUN_result_last'])
    if filtering_params["ONLY_IS_COUNT"]=='True':
        df=df.dropna(subset=['LABS_RDW_result_first'])
    
    #threshold features
    df=df.dropna(thresh=float(filtering_params["TH_features"])*(df.shape[0]), axis=1)
    
    #threshold age
    df=df[df["age"]<float(filtering_params["TH_age"])]
    
    #threshold LOS
    if filtering_params["IS_LOS_24hr"]=='True':
        df=df[df["LOS"]>0]
      
    print("filtered_shape: " + str(df.shape))
    
    return df    