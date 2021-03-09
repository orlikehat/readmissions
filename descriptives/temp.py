# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:42:00 2020

@author: orlyk
"""
import pandas as pd
df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\diagnoses\diagnoses_for_model\adm_table.pkl")


#df["N_cat"]=df.sum(axis=0)


y = df.values.sum(axis=0)   # requires < 4 GB of memory
y = pd.Series(y, index=df.columns)
y.to_frame()

y=y.drop(index='CaseNum')
y=y.drop(index='NGeneral symptoms and signs')
y=y.drop(index='NUNSPECIFIED')
y=y.drop(index='NPersons with potential health hazards related to family and personal history and certain conditions')
NPersons with potential health hazards related to family and personal history and certain conditions
y=y.sort_values(ascending=False)

y_top=y.iloc[1:20]

def diag_top_20(df,name):
    
    sum_table = df.values.sum(axis=0)   
    sum_table = pd.Series(sum_table, index=df.columns)
    sum_table.to_frame()

    sum_table=sum_table.drop(index='CaseNum')
    sum_table=sum_table.drop(index='NGeneral symptoms and signs')
    sum_table=sum_table.drop(index='NUNSPECIFIED')

    
    sum_table=sum_table.sort_values(ascending=False)
    sum_table_top=sum_table.iloc[1:20]
        
    sum_table_top.to_csv(output_path+name+"_top20.csv")    
    
output_path="C:/Users/orlyk/readmissions/project/descriptives/diagnoses/"
   
df_sum_all_adm=diag_top_20(df,"admission_all")


y=y.astype(int)