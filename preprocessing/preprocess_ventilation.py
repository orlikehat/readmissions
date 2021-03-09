# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:22:17 2020

@author: orlyk
"""

import pandas as pd
import numpy as np
def f_preprocess_ventilation():
    df_ICU=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\ventilation\ICU\df_ventilation_pop.pkl")
    df_vent=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\ventilation\mechanical_chameleon\mech_vent_pop.pkl")
    df_oxygen=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\ventilation\oxygen_chameleon\oxygen_support_chameleon_pop.pkl")
    df_pop=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\population\df_readmin_with_labels_base_slim.pkl")
    l_casenum=df_pop["CaseNum"].drop_duplicates()
    
    #ICU
    df_ICU_vent=df_ICU[df_ICU["class"]=="mech_ventilation"]
    df_ICU_oxygen=df_ICU[df_ICU["class"]=="support"]
    
    df_ICU_vent["is_ventilation"]=1
    df_ICU_oxygen["is_oxygen_support"]=1
    
    df_ICU_vent=df_ICU_vent[["CaseNum","is_ventilation"]]
    df_ICU_oxygen=df_ICU_oxygen[["CaseNum","is_oxygen_support"]]
    
    #chameleon mechanical ventilation
    df_vent["is_ventilation"]=1
    df_vent=df_vent[["CaseNum","is_ventilation"]]
    
    #chameleon oxygen support
    df_oxygen["is_oxygen_support"]=1
    df_oxygen=df_oxygen[["CaseNum","is_oxygen_support"]]
    
    #mechanical ventilation all: 
    df_mechanical_vent=pd.concat([df_ICU_vent,df_vent])
    #oxygen support all"
    df_oxygen_support=pd.concat([df_ICU_oxygen,df_oxygen])
    
    df_mechanical_vent.drop_duplicates(inplace=True)
    df_oxygen_support.drop_duplicates(inplace=True)
    
    
    
    #merge all
    df=pd.merge(l_casenum, df_mechanical_vent, how="left",on="CaseNum")
    df=pd.merge(df,df_oxygen_support,how="left",on="CaseNum")
    
    df["is_ventilation"]=np.where(df["is_ventilation"]==1,1,0)
    df["is_oxygen_support"]=np.where(df["is_oxygen_support"]==1,1,0)

    
    
    
    df.to_pickle("C:/Users/orlyk/readmissions/project/preprocessed/ventilation/ventilation_for_model/ventilation_for_model.pkl")
    return df