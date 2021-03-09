# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:45:03 2020

@author: orlyk
"""

import pandas as pd
import numpy as np
import os
import re


def f_preprocess_blood_chem():
    path = "C:/Users/orlyk/readmissions/project/preprocessed/labs/blood_chem/blood_chem_preprocessed"
    files = os.listdir( path )
    df_pop=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\population\df_readmin_with_labels_base_slim.pkl")
    l_casenum=df_pop["CaseNum"].drop_duplicates()
    df_blood_chem=l_casenum
    
    global calcs
    calcs=["max","min","first","last","last_norm_to_first","min_norm_to_first","max_norm_to_first","max_norm_to_min"]

    def cont_to_discrete(df,calcs,casenums):    
                
                df_fin=casenums
                
                if "max" in calcs:
                    df["result"]= df.filter(regex='result')
                    df_max=df.sort_values('result',ascending=False).drop_duplicates(['CaseNum'])
                    
                    new_names = [(i,i+'_max') for i in df_max.iloc[:,1:].columns.values]
                    df_max.rename(columns = dict(new_names), inplace=True)
                    df_max=df_max.drop(columns=["result_max"])
                    df_fin=pd.merge(df_fin,df_max,how="left",on="CaseNum")
                
                if "min" in calcs:   
                    df["result"]= df.filter(regex='result')
                    df_min=df.sort_values('result',ascending=True).drop_duplicates(['CaseNum'])
                    
                    new_names = [(i,i+'_min') for i in df_min.iloc[:,1:].columns.values]
                    df_min.rename(columns = dict(new_names), inplace=True)
                    df_min=df_min.drop(columns=["result_min"])
                    df_fin=pd.merge(df_fin,df_min,how="left",on="CaseNum")
                
                
                if "first" in calcs:
                    df["date_time"]= df.filter(regex='date_time')
                    df.sort_values(by='date_time',ascending=True, inplace=True)
                    df_first=df.drop_duplicates(subset=['CaseNum'], keep='first', inplace=False)
                    new_names = [(i,i+'_first') for i in df_first.iloc[:,1:].columns.values]
                    df_first.rename(columns = dict(new_names), inplace=True)
                    df_first=df_first.drop(columns=['date_time_first','result_first'])
                    df_fin=pd.merge(df_fin,df_first,how="left",on="CaseNum")
                
                
                if "last" in calcs: 
                    df["date_time"]= df.filter(regex='date_time')
                    df.sort_values(by='date_time',ascending=True, inplace=True)
                    df_last=df.drop_duplicates(subset=['CaseNum'], keep='last', inplace=False)
                    new_names = [(i,i+'_last') for i in df_last.iloc[:,1:].columns.values]
                    df_last.rename(columns = dict(new_names), inplace=True)
                    df_last=df_last.drop(columns=['date_time_last','result_last'])
                    
                    df_fin=pd.merge(df_fin,df_last,how="left",on="CaseNum")
                    
                    
                if (("last" in calcs) and ("first" in calcs)and("last_norm_to_first" in calcs)):
                    df_fin["temp_first"]= df_fin.filter(regex=r'(result_first)')
                    df_fin["temp_last"]= df_fin.filter(regex=r'(result_last)')
                    df_fin["last_result_normed"]=df_fin["temp_last"]/df_fin["temp_first"]
                    
                    name=list(df_fin.columns)[1] 
                    name=(re.split('_+', name))[0]
                    name=name+"_last_result_normed"
                    df_fin.rename(columns={"last_result_normed":name},inplace=True)
                    df_fin=df_fin.drop(columns=["temp_first","temp_last"])
                    
                       
                if (("max" in calcs) and ("first" in calcs)and("max_norm_to_first" in calcs)):
                    df_fin["temp_first"]= df_fin.filter(regex=r'(result_first)')
                    df_fin["temp_max"]= df_fin.filter(regex=r'(result_max)')
                    df_fin["max_result_normed"]=df_fin["temp_max"]/df_fin["temp_first"]
                    
                    name=list(df_fin.columns)[1] 
                    name=(re.split('_+', name))[0]
                    name=name+"_max_result_normed"
                    df_fin.rename(columns={"max_result_normed":name},inplace=True)
                    df_fin=df_fin.drop(columns=["temp_first","temp_max"])
                    

                if (("min" in calcs) and ("first" in calcs)and("min_norm_to_first" in calcs)):
                    df_fin["temp_first"]= df_fin.filter(regex=r'(result_first)')
                    df_fin["temp_min"]= df_fin.filter(regex=r'(result_min)')
                    df_fin["min_result_normed"]=df_fin["temp_min"]/df_fin["temp_first"]
                    
                    name=list(df_fin.columns)[1] 
                    name=(re.split('_+', name))[0]
                    name=name+"_min_result_normed"
                    df_fin.rename(columns={"min_result_normed":name},inplace=True)
                    df_fin=df_fin.drop(columns=["temp_first","temp_min"])
#                
#                if (("min" in calcs) and ("first" in calcs)and("max" in calcs) and ("last" in calcs)):
#                    df_fin["temp_first"]= df_fin.filter(regex=r'(result_first)')
#                    df_fin["temp_last"]= df_fin.filter(regex=r'(result_last)')
#                    df_fin["temp_min"]= df_fin.filter(regex=r'(result_min)')
#                    df_fin["temp_max"]= df_fin.filter(regex=r'(result_max)')
#                    df_fin["single_test_result"]=(df_fin["temp_first"]+df_fin["temp_last"]+df_fin["temp_min"]+df_fin["temp_max"])/4==df_fin["temp_max"]
#                    df_fin["single_test_result"]=df_fin["single_test_result"].astype(int)
#                    
#                    name=list(df_fin.columns)[1] 
#                    name=(re.split('_+', name))[0]
#                    name=name+"_single_test_result"
#                    df_fin.rename(columns={"single_test_result":name},inplace=True)
#                    df_fin=df_fin.drop(columns=["temp_first","temp_min","temp_last","temp_max"])
    
            
                return df_fin
    for file in files:
        df=pd.read_pickle(path+"/"+file)
        
        #eliminate zeros to avoid inf when normalizing to first
        l=list(df.columns)
        result_column=l[1]
        df[result_column]=np.where(df[result_column]==0,0.001,df[result_column])
        
        df=cont_to_discrete(df,calcs,l_casenum)
        df_blood_chem=pd.merge(df_blood_chem,df,how="outer",on="CaseNum")
        
         
    ##add engineered features:##################
    #df_blood_df_blood_chem_ef=df_blood_count.dropna()
    #add NLR
    #df_blood_count_ef["NLR_last"]=df_blood_count_ef["lymphocytes_abs_result_last"]/df_blood_count_ef["neutro_abs_result_last"]
    #df_blood_count_ef["NLR_first"]=df_blood_count_ef["lymphocytes_abs_result_first"]/df_blood_count_ef["neutro_abs_result_first"]
    #df_blood_count_ef["NLR_max"]=df_blood_count_ef["lymphocytes_abs_result_max"]/df_blood_count_ef["neutro_abs_result_max"]
    #df_blood_count_ef["NLR_min"]=df_blood_count_ef["lymphocytes_abs_result_min"]/df_blood_count_ef["neutro_abs_result_min"]
    
     ##########################################   
    
    
    
    
    
    
    df_blood_chem_results_only= df_blood_chem.filter(regex=r'(result|CaseNum)')
    
    df_blood_chem.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\labs\blood_chem\lab_blood_chem_for_model\blood_chem_results_with_dates.pkl")
    df_blood_chem_results_only.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\labs\blood_chem\lab_blood_chem_for_model\blood_chem_results_only.pkl")
    
    return df_blood_chem_results_only
    
    
        