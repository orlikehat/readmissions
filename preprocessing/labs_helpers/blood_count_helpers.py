# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:43:52 2020

@author: orlyk
"""

import pandas as pd
import seaborn as sns
import os
import matplotlib 
import numpy as np 
import matplotlib.pyplot as plt 

path = "C:/Users/orlyk/readmissions/project/preprocessed/labs/blood_count/"
    
df_baso_abs=pd.read_pickle(path+"basophils_abs_pop.pkl")    

df_baso_abs.basophils_abs_result = df_baso_abs['basophils_abs_result'].replace({'.....': '-',
                                              ":::::":"-"})
df_baso_abs = df_baso_abs[df_baso_abs['basophils_abs_result'] != '-']
df_baso_asb_group=df_baso_abs.groupby(by="basophils_abs_result")

df_baso_asb_group_hist=df_baso_asb_group.nunique()


df_baso_perc=pd.read_pickle(path+"basophils_perc_pop.pkl")    

df_baso_perc.basophils_perc_result = df_baso_perc['basophils_perc_result'].replace({'.....': '-',":::::":"-"})
df_baso_perc = df_baso_perc[df_baso_perc['basophils_perc_result'] != '-']


df=pd.read_pickle(path+"platelet_volume_pop.pkl")   
df["platelet_volume_result"] = df["platelet_volume_result"].replace({'-----': '-',":::::":"-",".....":"-"})
df = df[df["platelet_volume_result"] != '-']


df=pd.read_pickle(path+"MCHC_pop.pkl")   
df["MCHC_result"] = df["MCHC_result"].replace({'-----': '-',":::::":"-",".....":"-","+++++":"-"})
df = df[df["MCHC_result"] != '-']



df=pd.read_pickle(path+"hgb_pop.pkl")   
df["MCHC_result"] = df["MCHC_result"].replace({'-----': '-',":::::":"-",".....":"-","+++++":"-"})
df = df[df["MCHC_result"] != '-']



hgb_pop
MCHC_pop