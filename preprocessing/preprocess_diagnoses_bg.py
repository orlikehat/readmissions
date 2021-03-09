# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:57:06 2021

@author: orlyk
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:31:38 2020

@author: orlyk
"""

import pandas as pd
import numpy as np


def f_preprocess_diagnoses_bg():

    df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\population\df_readmin_with_labels_base.pkl")
    output_path="C:/Users/orlyk/readmissions/project/preprocessed/diagnoses/diagnoses_for_model/"
    
    df_diag=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\diagnoses\df_diagnoses_bg_pop.pkl")
    def pivot_diagnoses(df,level):
            df["N"]=1
                    
            diag_table = pd.pivot_table(df, values=['N'], index="CaseNum",columns=[level], aggfunc=np.sum, fill_value=0)    
            #diag_table=diag_table.drop_duplicates()
            diag_table = diag_table.reset_index()
            diag_table.columns = list(map("".join, diag_table.columns))
            return diag_table
        
        
    df=df[df["BASE_FLG"]==1]
    df=df[["CaseNum","PatNum","Age","year"]]
    df=pd.merge(df,df_diag,on="CaseNum",how="left")
    df=df[['CaseNum', 'PatNum_x','DiagCode_ICD9','Diag_Free_Text','Category', 'Block', 'Chapter']]
    
    df_bg_block261=pivot_diagnoses(df,'Block')
    for col in df_bg_block261.columns:
        if col !='CaseNum':
            df_bg_block261[col]=np.where(df_bg_block261[col]>0,1,0)
    df_bg_block261['Total_bg_block']= df_bg_block261.sum(axis=1)
            
           
    df_bg_chapter19=pivot_diagnoses(df,'Chapter')
    for col in df_bg_chapter19.columns:
        if col !='CaseNum':
            df_bg_chapter19[col]=np.where(df_bg_chapter19[col]>0,1,0)
    df_bg_chapter19['Total_bg_chapter']= df_bg_chapter19.sum(axis=1)
    
    df_bg_category1395=pivot_diagnoses(df,'Category')
    for col in df_bg_category1395.columns:
        if col !='CaseNum':
            df_bg_category1395[col]=np.where(df_bg_category1395[col]>0,1,0)
    df_bg_category1395['Total_bg_category']= df_bg_category1395.sum(axis=1)
    
    #df_bg_no_category=df.dropna(subset=["Diag_Free_Text"])
    #df_bg_no_category=df_bg_no_category[['CaseNum','Diag_Free_Text']].drop_duplicates()
    #df_bg_no_category=pivot_diagnoses(df_bg_no_category,'Diag_Free_Text')
    #for col in df_bg_no_category.columns:
    #    if col !='CaseNum':
    #        df_bg_no_category[col]=np.where(df_bg_no_category[col]>0,1,0)
    #df_bg_no_category['Total_bg_no_category']= df_bg_no_category.sum(axis=1)
    
    
    
    
    df_bg_block261.to_pickle(output_path+"bg_table_block261.pkl")
    df_bg_block261=df_bg_block261.add_prefix('BLOCK_')  
    df_bg_chapter19.to_pickle(output_path+"bg_table_chapter19.pkl")
    df_bg_chapter19=df_bg_chapter19.add_prefix('CHAPTER_')  
    df_bg_category1395.to_pickle(output_path+"bg_table_category1395.pkl")
    df_bg_category1395=df_bg_category1395.add_prefix('CATEGORY_')
    
    df_fin=pd.merge(df_bg_block261,df_bg_chapter19,how="left",right_on="CHAPTER_CaseNum",left_on="BLOCK_CaseNum")
    df_fin=pd.merge(df_fin,df_bg_category1395,how="left",right_on="CATEGORY_CaseNum",left_on="BLOCK_CaseNum")
    
    df_fin=df_fin.drop(columns=['BLOCK_CaseNum', 'CHAPTER_CaseNum'])
    df_fin = df_fin.rename(columns={"CATEGORY_CaseNum": "CaseNum"})
    
    return  df_fin#,df_bg_chapter19,df_bg_category1395
    


