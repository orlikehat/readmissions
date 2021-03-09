# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:16:20 2020

@author: orlyk
"""
import pandas as pd
import numpy as np

def f_preprocess_CCI():
    df_CCI =pd.read_pickle(r"C:/Users/orlyk/readmissions/project/preprocessed/CCI/df_CCI_pop.pkl")
    df_CCI.replace({None: "0"},inplace=True)
    
    
    for column  in df_CCI.iloc[:,1:]:
        df_CCI[column]=pd.to_numeric(df_CCI[column])
        
        
    for col in df_CCI.iloc[:,3:]:
        df_CCI[col]=np.where(df_CCI[col]>0,1,0)
        
    df_CCI_bg=df_CCI[["SPMyocardialinfarction",
                      "CongestiveHeartFailure",
                      "Peripheralvasculardisea",
                      "SPCerebrovasculardisea",
                      "Connectivetissuedisease",
                      "Dementia",
                      "Chronicpulmonarydisease",
                      "Ulcerdisease",
                      "liverdisease",
                      "Diabetes",
                      "Hemiplegia",
                      "renaldisease",
                      "Tumor",
                      "Leukemia",
                      "Lymphoma"]]
    df_CCI_bg=df_CCI_bg.add_prefix('bg_cci_')    

    df_CCI_main=df_CCI[["CaseNum","CharlsDiseases","CharlsScore"]]
    
    df_CCI=pd.merge(df_CCI_bg,df_CCI_main,left_index=True, right_index=True)
        
    df_CCI.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\CCI\CCI_for_model\CCI_for_model.pkl")    
    df_CCI.to_csv(r"C:\Users\orlyk\readmissions\project\preprocessed\CCI\CCI_for_model\CCI_for_model.csv")    

     
    
    return df_CCI