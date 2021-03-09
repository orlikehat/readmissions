# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:16:18 2021

@author: orlyk
"""

import pandas as pd
import numpy as np


def f_preprocess_surgery():

    df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\surgery\df_surgery_pop.pkl")
    df_grouped=df[["ServSqNum","CaseNum"]].groupby(by="CaseNum").count()
    #df_grouped["is_surgery"]=1
    
    #urgent admission - yes/no
    df_adm=df[["IsAdmissionsUrgentFLG","CaseNum"]].groupby(by="CaseNum").sum()
    df_adm["IsAdmissionsUrgentFLG"]=np.where(df_adm["IsAdmissionsUrgentFLG"]>0,1,0)
    
    #no of urgent procedures
    df_urgent_no=df[["UrgencyFLG","CaseNum"]].groupby(by="CaseNum").sum()
    #df_adm["UrgencyFLG"]=np.where(df_adm["IsAdmissionsUrgentFLG"]>0,1,0)
    
    #no of sessia
    df_sessia_no=df[["SesiaFLG","CaseNum"]].groupby(by="CaseNum").sum()
    #df_adm["UrgencyFLG"]=np.where(df_adm["IsAdmissionsUrgentFLG"]>0,1,0)
    
    #was thwer an elective procedure - yes/no
    df["elective"]=np.where(df["UrgencyType"]==1,1,0)
    df_elective_flag=df[["elective","CaseNum"]].groupby(by="CaseNum").sum()
    df_elective_flag["elective"]=np.where(df_elective_flag["elective"]>0,1,0)
    
    df_fin=pd.merge(df_grouped,df_adm,left_index=True, right_index=True)
    df_fin=pd.merge(df_fin,df_urgent_no,left_index=True, right_index=True)
    df_fin=pd.merge(df_fin,df_sessia_no,left_index=True, right_index=True)
    df_fin=pd.merge(df_fin,df_elective_flag,left_index=True, right_index=True)
    
    #df_fin=df_fin.drop(columns=['ServSqNum'])
    df_fin=df_fin.rename(columns={"UrgencyFLG": "urgent_num", "SesiaFLG": "sesia_num","ServSqNum":"num_of_surgery"})
    df_fin=df_fin.reset_index()
    
    df_fin.to_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\surgery\surgery_for_model\suregry_preprocessed.pkl")

    return df_fin






#
#
#
#
#urgency_type_d = {1: 'elective', 2: 'urgent', 3: 'sessia'}
#df['urgency_type_cat'] = df['UrgencyType'].map(urgency_type_d)
#
#surgery_type_d= {1: 'morning', 2: 'toranut', 3: 'sessia'}
#df['surgery_type_cat'] = df['SurgeryTypes'].map(surgery_type_d)
#
#df=df.sort_values(by='SurgeryStartDate', ascending=True)
##df=df.drop_duplicates()
#
#df=pd.merge(l_casenum,df,on="CaseNum",how="left")
#
#IsAdmissionsUrgentFLG