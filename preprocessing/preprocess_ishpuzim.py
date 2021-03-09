# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:07:40 2021

@author: orlyk
"""
import pandas as pd
def f_preprocess_ishpuzim():

    df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\ishpuzim_indicators\df_ishpuzim_pop.pkl")
    
    #quarters admitted/discharged
    df['EnterQuarterDesc'] = df['EnterQuarterDesc'].str[:2]
    df['ExitQuarterDesc'] = df['ExitQuarterDesc'].str[:2]
    
    #dummies
    df=pd.get_dummies(data=df, columns=['ExitEndOfWeekFLG','KupaCode','EnterQuarterDesc',
                                        'ExitQuarterDesc'])
    
    #drop unessecary
    df=df.drop(columns=['ExitEndOfWeekFLG_W  ','First_Sodium_Value','PatNum'])
    
    df.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\ishpuzim_indicators\ishpuzim_for_model\ishpuzim_preprocessed.pkl")
    
    return df