# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:35:27 2021

@author: orlyk
"""
import pandas as pd
import numpy as np
def f_preprocess_microbiology():

    df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\microbiology\microbiology_pop.pkl")
    
    
    #df=df[['CaseNum',
    # 'category_anaerobe',
    # 'category_candida',
    # 'category_negative',
    # 'category_positive',
    # 'contaminant',
    # ]]
    
    
    #positive culture flag + positive count
    df_is_positive=df[["CaseNum","organism_code"]].groupby(by="CaseNum").count()
    df_is_positive=df_is_positive.rename(columns={"organism_code": "num_of_pos_cult"})
    df_is_positive["positive_flag"]=np.where(df_is_positive["num_of_pos_cult"]>0,1,0)
    #df_is_positive["num_of_cultures"]=df_is_positive["organism_code"]
    
    ##num of microorganisms
    df_microorg=df.groupby(by="CaseNum")
    df_microorg=df_microorg["organism_code"].nunique()
    df_microorg=df_microorg.rename("num_of_organisms")
    
    
    # num of microorg in each microbiology category
    df_cat=df[["CaseNum","organism_code",'category_anaerobe', 'category_candida',
           'category_negative', 'category_positive', 'contaminant']].drop_duplicates()
    df_cat=df_cat.groupby(by="CaseNum").sum()
    
    
    #merge all
    df_fin=pd.merge(df_is_positive,df_microorg,left_index=True, right_index=True)
    df_fin=pd.merge(df_fin,df_cat,left_index=True, right_index=True)
    
    df_fin.drop(columns=['organism_code'],inplace=True)
    df_fin=df_fin.reset_index()
    
    
    df_fin.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\microbiology\microbiology_for_model\microbiology_preprocessed.pkl")
    
    return df_fin