# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:22:06 2020

@author: orlyk
"""
import pandas as pd
import numpy as np
#df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\population\population_for_model\df_basic_data.pkl")
#df["LABEL_HOSP"]=np.where(df["LABEL_HOSP"]==1,"readmission","no_readmission")
#df["LABEL_JUST_ER"]=np.where(df["LABEL_JUST_ER"]==1,"readmission","no_readmission")
#
#df_1=df[df["LABEL_HOSP"]=="readmission"]
#df_0=df[df["LABEL_HOSP"]=="no_readmission"]
#
#df_1_ER=df[df["LABEL_JUST_ER"]=="readmission"]
#df_0_ER=df[df["LABEL_JUST_ER"]=="no_readmission"]
#


def describe_population(df,name):
    #N
    N=df["age"].count()
    
    #age
    age_mean=df["age"].mean()
    age_SD=df["age"].std()
    
    #sex
    females=df["sex"].sum()/df["sex"].count()*100
    
    #LOS
    LOS=df["LOS"].median()
    LOS_q25=df["LOS"].quantile(q=0.25)
    LOS_q75=df["LOS"].quantile(q=0.75)
    LOS_IQR=LOS_q75-LOS_q25
    
     
      
    d = {'vars':["N","age_mean","age_SD","sex_perc_females","LOS","LOS_IQR"],
         'values_'+name: [N,age_mean,age_SD,females,LOS,LOS_IQR]} 
                    
     
    df_table = pd.DataFrame(data=d)
    
    return df_table
    
    
    
    
    