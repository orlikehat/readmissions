# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np




df=pd.read_pickle(r'C:\Users\orlyk\readmissions\project\preprocessed\population\df_readmin_with_labels_base.pkl')
df=df[["CaseNum","PatNum","EnterDate"]]
pat_l=list(df["PatNum"].unique())
df["EnterDate"]=pd.to_datetime(df["EnterDate"])
df_fin=pd.DataFrame()
counter=0
for pat in pat_l:
    counter=counter+1
    df_pat=df[df["PatNum"]==pat]
    df_pat=df_pat.reset_index()
    temp_df=pd.DataFrame()
   
    for i in range(len(df_pat)):
        temp_df[i]=(df_pat["EnterDate"]-df_pat["EnterDate"][i]).dt.days
    
    temp_df=np.where(temp_df>=0,0,np.where(temp_df>-366,1,0))
    df_pat["hosp_past_year"]=temp_df.sum(axis=0)
    df_fin=df_fin.append(df_pat)
    print(counter)
    
df_fin.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\adm_previous_year\adm_previous_year.pkl")    
    
        
        
        
# visits_pandas_df.groupby('PatientId').apply(lambda x: (((x['visit_start_date'].shift(-1)-x['visit_end_date']).dt.days.shift(1).le(30)) ).astype(int)).values

# x['Admission Date'].\
# shift(-1)

# df = df.sort_values(['Patient','Admission Date'])
# pd.to_datetime(x)