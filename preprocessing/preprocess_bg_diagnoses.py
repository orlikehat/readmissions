# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:57:00 2021

@author: orlyk
"""

import pandas as pd
import numpy as np



#df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\diagnoses\df_diagnoses_full.pkl")
df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\diagnoses\temp.pkl")

df=df[df["Diag_Type_Code"]==1]

#df=pd.read_pickle(r'C:\Users\orlyk\readmissions\project\preprocessed\population\df_readmin_with_labels_base.pkl')
#df=df[["CaseNum","PatNum","EnterDate"]]
pat_l=list(df["PatNum"].unique())
df["date_time"]=pd.to_datetime(df["date_time"])
df_fin=pd.DataFrame()
counter=0
for pat in pat_l:
    counter=counter+1
    df_pat=df[df["PatNum"]==pat]
    df_pat=df_pat.reset_index()
    df_pat=df_pat.sort_values(by=['date_time'])
    temp_df=pd.DataFrame()
   
    for i in range(len(df_pat)):
       temp_df=df_pat[0:i+1]
       temp_df["casenum_updated"]=df_pat["CaseNum"][i]
       df_fin=df_fin.append(temp_df)
      # df_fin=df_fin.reset_index()
      # df_fin=df_fin.drop_duplicates(subset=['CaseNum','Diag_Free_text'], keep='last')

#
#df_pat=b.reset_index()
#df_pat=df_pat.sort_values(by=['date_time'])
#for i in range(len(df_pat)):
#   temp_df=df_pat[0:i]
#   df_fin=df_fin.append(temp_df)  
#   df_fin=df_fin.drop_duplicates(subset=['CaseNum','Diag_Free_Text'], keep='last')
#  
#a=df_pat[0:2]    
#    
#
#
#
#def pivot_diagnoses(df):
#    df["N"]=1
#            
#    diag_table = pd.pivot_table(df, values=['N'], index="CaseNum",columns=['Block'], aggfunc=np.sum, fill_value=0)    
#    #diag_table=diag_table.drop_duplicates()
#    diag_table = diag_table.reset_index()
#    diag_table.columns = list(map("".join, diag_table.columns))
#    return diag_table
#
#
#df=df[df["BASE_FLG"]==1]
#df=df[["CaseNum","PatNum","Age","year"]]
#df=pd.merge(df,df_diag,on="CaseNum",how="left")
#df=df[["CaseNum","PatNum","Age","year","DiagCode_ICD9","Diag_Type_Code","Diag_Free_Text"]]
#
#df_md=pd.read_excel(r"C:\Users\orlyk\readmissions\project\code\support_files\Diagnosis_codes_MDClone_short.xlsx")
#
#df["Diag_Free_Text"]=df["Diag_Free_Text"].str.upper()
#df_md["Diagnosis"]=df_md["Diagnosis"].str.upper()
#
##part to diagnoses discharge/admission  because of lack of memory when merging
#df_disch=df[df["Diag_Type_Code"]==4]
#df_disch=pd.merge(df_disch,df_md,left_on="Diag_Free_Text",right_on="Diagnosis",how="left")
#df_disch_table=pivot_diagnoses(df_disch)
#for col in df_disch_table.columns:
#    if col !='CaseNum':
#        df_disch_table[col]=np.where(df_disch_table[col]>0,1,0)
#
#
#
#df_adm=df[df["Diag_Type_Code"]==2]
#df_adm=pd.merge(df_adm,df_md,left_on="Diag_Free_Text",right_on="Diagnosis",how="left")
#df_adm_table=pivot_diagnoses(df_adm)
#for col in df_adm_table.columns:
#    if col !='CaseNum':
#        df_adm_table[col]=np.where(df_adm_table[col]>0,1,0)
#
#df_disch_table=df_disch_table.drop(columns='NGeneral symptoms and signs')
#df_disch_table=df_disch_table.drop(columns='NUNSPECIFIED')
#df_disch_table=df_disch_table.drop(columns='NPersons with potential health hazards related to family and personal history and certain conditions')
#
#df_adm_table=df_adm_table.drop(columns='NGeneral symptoms and signs')
#df_adm_table=df_adm_table.drop(columns='NUNSPECIFIED')
#df_adm_table=df_adm_table.drop(columns='NPersons with potential health hazards related to family and personal history and certain conditions')
#
#
#
#df_disch_table.to_pickle(output_path+"disch_table.pkl")
#df_adm_table.to_pickle(output_path+"adm_table.pkl")
