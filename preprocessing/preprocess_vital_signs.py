# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:42:10 2020

@author: orlyk
"""

import pandas as pd
import numpy as np


#todo: revise cont_to_discrete according to labs - blood count preprocessing

def f_preprocess_VS():

    
    df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\vital_signs\df_vital_signs_pop.pkl")
    df_namer=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\vital_signs\df_vital_signs_pop_namer.pkl")
    output_path="C:/Users/orlyk/readmissions/project/preprocessed/vital_signs/VS_for_model/"
    
    #combine chameleon and namer
    df=pd.concat([df,df_namer])
    df["code"]=df["CaseNum"].astype(str)+"_"+df["eng_param_name"].astype(str)+"_"+df["date_time"].astype(str)
    df=df.drop_duplicates(subset="code")
    
    
    
    
    def cont_to_discrete(df,name,calcs,casenums):    
        
            df_gb=df.groupby(["CaseNum"])
            df_fin=casenums
           
            if "max" in calcs:
              
                df_max=df_gb.max()
                df_max.reset_index(inplace=True)
                df_max=df_max[["CaseNum","Result"]]
                df_max.rename(columns={"Result": "result_max_"+name}, inplace=True)
                df_fin=pd.merge(df_fin,df_max,how="left",on="CaseNum")
                
            if "min" in calcs:   
        
                df_min=df_gb.min()
                df_min.reset_index(inplace=True)
                df_min=df_min[["CaseNum","Result"]]
                df_min.rename(columns={"Result": "result_min_"+name}, inplace=True)
                df_fin=pd.merge(df_fin,df_min,how="left",on="CaseNum")
                
                
            if "first" in calcs:
                df.sort_values(by='date_time',ascending=True, inplace=True)
                df_first=df.drop_duplicates(subset=['CaseNum'], keep='first', inplace=False)
                df_first=df_first[["CaseNum","Result","date_time"]]
                df_first.rename(columns={"Result": "result_first_"+name}, inplace=True)
                df_first.rename(columns={"date_time": "date_time_first_"+name}, inplace=True)
                df_fin=pd.merge(df_fin,df_first,how="left",on="CaseNum")
        
                
            if "last" in calcs:  
                df.sort_values(by='date_time',ascending=True, inplace=True)
                df_last=df.drop_duplicates(subset=['CaseNum'], keep='last', inplace=False)
                df_last=df_last[["CaseNum","Result","date_time"]]
                df_last.rename(columns={"Result": "result_last_"+name}, inplace=True)
                df_last.rename(columns={"date_time": "date_time_last_"+name}, inplace=True)
                df_fin=pd.merge(df_fin,df_last,how="left",on="CaseNum")
                
            return df_fin
            
    
    calcs=["max","min","first","last"]  
    l_casenum=df["CaseNum"].drop_duplicates()
    
    
    df_hr=cont_to_discrete(df[df["eng_param_name"]=="hr"],"HR",calcs,l_casenum)
    df_sbp=cont_to_discrete(df[df["eng_param_name"]=="sbp"],"sys_BP",calcs,l_casenum)
    df_dpb=cont_to_discrete(df[df["eng_param_name"]=="dbp"],"dias_BP",calcs,l_casenum)
    df_temp=cont_to_discrete(df[df["eng_param_name"]=="temp"],"TMP",calcs,l_casenum)
    df_sat=cont_to_discrete(df[df["eng_param_name"]=="spo2"],"SATUR",calcs,l_casenum)
    
    df_VS=pd.merge(df_hr,df_sbp, on="CaseNum", how="outer")
    df_VS=pd.merge(df_VS,df_dpb, on="CaseNum", how="outer")
    df_VS=pd.merge(df_VS,df_temp, on="CaseNum", how="outer")
    df_VS=pd.merge(df_VS,df_sat, on="CaseNum", how="outer")
    
    
    df_VS["date_time_first_sys_BP"]=pd.to_datetime(df_VS["date_time_first_sys_BP"])
    
    df_VS['year'] = df_VS['date_time_first_sys_BP'].dt.year
    df_VS["source"]="chameleon"
    
    df_VS_short=df_VS[['CaseNum',
     'result_max_HR',
     'result_min_HR',
     'result_first_HR',
     'result_last_HR',
     'result_max_sys_BP',
     'result_min_sys_BP',
     'result_first_sys_BP',
     'result_last_sys_BP',
     'result_max_dias_BP',
     'result_min_dias_BP',
     'result_first_dias_BP',
     'result_last_dias_BP',
     'result_max_TMP',
     'result_min_TMP',
     'result_first_TMP',
     'result_last_TMP',
     'result_max_SATUR',
     'result_min_SATUR',
     'result_first_SATUR',
     'result_last_SATUR']]
    
    
    
    df_VS.to_pickle(output_path+"vs_processed_full.pkl")
    df_VS_short.to_pickle(output_path+"vs_processed_short.pkl")
    
    return df_VS_short
    
    
    