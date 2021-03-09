# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:31:38 2020

@author: orlyk
"""

import pandas as pd
import numpy as np


def f_preprocess_diagnoses():
    
    df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\population\df_readmin_with_labels_base.pkl")
    output_path="C:/Users/orlyk/readmissions/project/preprocessed/diagnoses/diagnoses_for_model/"
    
    df_diag=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\dwh\df_dwh_prd_prd_fact_diagnosis.pkl")
    def pivot_diagnoses(df):
        df["N"]=1
                
        diag_table = pd.pivot_table(df, values=['N'], index="CaseNum",columns=['Block'], aggfunc=np.sum, fill_value=0)    
        #diag_table=diag_table.drop_duplicates()
        diag_table = diag_table.reset_index()
        diag_table.columns = list(map("".join, diag_table.columns))
        return diag_table
    
    
    df=df[df["BASE_FLG"]==1]
    df=df[["CaseNum","PatNum","Age","year"]]
    df=pd.merge(df,df_diag,on="CaseNum",how="left")
    df=df[["CaseNum","PatNum","Age","year","DiagCode_ICD9","Diag_Type_Code","Diag_Free_Text"]]
    
    df_md=pd.read_excel(r"C:\Users\orlyk\readmissions\project\code\support_files\Diagnosis_codes_MDClone_short.xlsx")
    
    df["Diag_Free_Text"]=df["Diag_Free_Text"].str.upper()
    df_md["Diagnosis"]=df_md["Diagnosis"].str.upper()
    
    #part to diagnoses discharge/admission  because of lack of memory when merging
    df_disch=df[df["Diag_Type_Code"]==4]
    df_disch=pd.merge(df_disch,df_md,left_on="Diag_Free_Text",right_on="Diagnosis",how="left")
    df_disch_table=pivot_diagnoses(df_disch)
    for col in df_disch_table.columns:
        if col !='CaseNum':
            df_disch_table[col]=np.where(df_disch_table[col]>0,1,0)
    
    
    
    df_adm=df[df["Diag_Type_Code"]==2]
    df_adm=pd.merge(df_adm,df_md,left_on="Diag_Free_Text",right_on="Diagnosis",how="left")
    df_adm_table=pivot_diagnoses(df_adm)
    for col in df_adm_table.columns:
        if col !='CaseNum':
            df_adm_table[col]=np.where(df_adm_table[col]>0,1,0)
    
    df_disch_table=df_disch_table.drop(columns='NGeneral symptoms and signs')
    df_disch_table=df_disch_table.drop(columns='NUNSPECIFIED')
    df_disch_table=df_disch_table.drop(columns='NPersons with potential health hazards related to family and personal history and certain conditions')
    
    df_adm_table=df_adm_table.drop(columns='NGeneral symptoms and signs')
    df_adm_table=df_adm_table.drop(columns='NUNSPECIFIED')
    df_adm_table=df_adm_table.drop(columns='NPersons with potential health hazards related to family and personal history and certain conditions')
    
    
    
    df_disch_table.to_pickle(output_path+"disch_table.pkl")
    df_adm_table.to_pickle(output_path+"adm_table.pkl")
    
    return df_adm_table,df_disch_table
    
    
    
    
    #df_disch["N"]=1
    #    
    #        
    #diag_table = pd.pivot_table(df_disch, values=['N'], index="CaseNum",columns=['Block'], aggfunc=np.sum, fill_value=0)    
    ##diag_table=diag_table.drop_duplicates()
    #diag_table = diag_table.reset_index()
    #diag_table.columns = list(map("".join, diag_table.columns)
    #)
    #
    #
    #
    ##df["DiagCode_ICD9"]=df["DiagCode_ICD9"].astype('str')
    ##
    ##df["DiagCode_ICD9"] =df["DiagCode_ICD9"].str.lstrip('0') 
    ##
    ##df_ICD["icd9code"]=df_ICD["icd9code"].astype('str')
    #df["DiagCode_ICD9"] =df["DiagCode_ICD9"].str.lstrip('0') 
    #
    #df["DiagCode_ICD9"]=pd.to_numeric(df["DiagCode_ICD9"],errors="ignore")
    #df_ICD["icd9code"]=pd.to_numeric(df_ICD["icd9code"],errors="ignore")
    #
    #df_m=pd.merge(df,df_ICD,left_on="DiagCode_ICD9",right_on="icd9code",how="left")
    #
    #
    #
    #df_disch=df[df["Diag_Type_Code"]==4]
    #
    
    
    
    
    
    
    #df_icd_cat=pd.read_excel(r"C:\Users\orlyk\readmissions\project\code\support_files\ICD_categorization.xlsx")
    #df = pd.merge(df,df_icd_cat, how='left', left_on='icd9_final',right_on="icd9code")
    #df=df[["CaseNum","Diag_Type_Code","Depth1","Depth2"]]
    #
    #
    #df_discharge=df[df["Diag_Type_Code"]==1]
    #
    #df_discharge["N"]=1
    #    
    #       
    #diag_table = pd.pivot_table(df_discharge, values=['N'], index="CaseNum",columns=['Depth2'], aggfunc=np.sum, fill_value=0)    
    #diag_table=diag_table.drop_duplicates()
    #diag_table = diag_table.reset_index()
    #diag_table.columns = list(map("".join, diag_table.columns))
    #
    ##df=pd.merge(df,diag_table, how="left",on="CaseNum")
    #
    
