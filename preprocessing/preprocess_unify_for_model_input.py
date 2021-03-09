# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:17:51 2020

@author: orlyk
"""

import pandas as pd
import numpy as np

from preprocess_basic_data import f_preprocess_basic_data
from preprocess_diagnoses import f_preprocess_diagnoses
from preprocess_vital_signs import f_preprocess_VS
from preprocess_blood_count import f_preprocess_blood_count
from preprocess_blood_chem import f_preprocess_blood_chem
from preprocess_ventilation import f_preprocess_ventilation
from preprocess_CCI import f_preprocess_CCI
from preprocess_blood_coag import f_preprocess_blood_coag
from preprocess_microbiology import f_preprocess_microbiology
from preprocess_surgery import f_preprocess_surgery
from preprocess_ishpuzim import f_preprocess_ishpuzim
from preprocess_diagnoses_bg import f_preprocess_diagnoses_bg

IS_RUN_POPULATION=False 
IS_RUN_DIAGNOSES=False
IS_RUN_VS=False
IS_RUN_BLOOD_COUNT=False   
IS_RUN_BLOOD_CHEM=False
IS_RUN_BLOOD_COAG=False
IS_RUN_VENTILATION=False
IS_RUN_CCI=False
IS_RUN_MICROBIOLOGY=False
IS_RUN_SURGERY=False
IS_RUN_HOSP_PREV_YEAR=False#super slow so no option of running here
IS_RUN_ISHPUZIM=False
IS_RUN_DIAGNOSES_BG=False
#todo add other hierarchies in diagnoses bg
IS_OUTLIER_REMOVAL=True
IS_FEATURE_REMOVAL=True
IS_DUMMIES=True
IS_DIAG_BLOCK_ONLY=False


data_folder_path = "C:/Users/orlyk/readmissions/project/preprocessed/"
output_path="C:/Users/orlyk/readmissions/project/preprocessed/model_input/"


#population:
if IS_RUN_POPULATION:
    df=f_preprocess_basic_data()
else: 
    df=pd.read_pickle("C:/Users/orlyk/readmissions/project/preprocessed/population/population_for_model/df_basic_data_short.pkl")

#diagnoses
if IS_RUN_DIAGNOSES:
    df_diag_adm,df_diag_disch=f_preprocess_diagnoses()
else:
    df_diag_adm = pd.read_pickle (data_folder_path+ "diagnoses/diagnoses_for_model/adm_table.pkl")
    df_diag_disch=pd.read_pickle (data_folder_path+ "diagnoses/diagnoses_for_model/disch_table.pkl")

df_diag_adm=df_diag_adm.add_prefix('DIAG_') 
df_diag_adm=df_diag_adm.add_suffix('_adm') 
 
df_diag_disch=df_diag_disch.add_prefix('DIAG_')    
df_diag_disch=df_diag_disch.add_suffix('_dis') 


#vital signs
if IS_RUN_VS:
    df_VS=f_preprocess_VS()
else:
    df_VS=pd.read_pickle (data_folder_path+ "vital_signs/VS_for_model/vs_processed_short.pkl")    

df_VS=df_VS.add_prefix('VS_')    
#blood_count
if IS_RUN_BLOOD_COUNT:
    df_blood_count=f_preprocess_blood_count()
else:
    df_blood_count=pd.read_pickle (data_folder_path+ "labs/blood_count/lab_blood_count_for_model/blood_count_results_only.pkl")

df_blood_count=df_blood_count.add_prefix('LABS_')    
    
if IS_RUN_BLOOD_CHEM:
    df_blood_chem=f_preprocess_blood_chem()
else:
    df_blood_chem=pd.read_pickle (data_folder_path+ "labs/blood_chem/lab_blood_chem_for_model/blood_chem_results_only.pkl")
df_blood_chem=df_blood_chem.add_prefix('LABS_')    

    
if IS_RUN_VENTILATION:
    df_ventilation=f_preprocess_ventilation()
else: 
    df_ventilation=pd.read_pickle(data_folder_path+ "ventilation/ventilation_for_model/ventilation_for_model.pkl")
df_ventilation=df_ventilation.add_prefix('VENT_')    



if IS_RUN_CCI:
    df_CCI=f_preprocess_CCI()
else:
    df_CCI=pd.read_pickle(data_folder_path+ "CCI/CCI_for_model/CCI_for_model.pkl")
df_CCI=df_CCI.add_prefix('CCI_')    


if IS_RUN_BLOOD_COAG:
    df_blood_coag=f_preprocess_blood_coag()
else:
    df_blood_coag=pd.read_pickle(data_folder_path+ "labs/blood_coagulation/blood_coag_for_model/blood_coag_results_only.pkl")
df_blood_coag=df_blood_coag.add_prefix('LABS_')    


if IS_RUN_MICROBIOLOGY:
    df_microbiology=f_preprocess_microbiology()
else:    
    df_microbiology=pd.read_pickle(data_folder_path+ "microbiology/microbiology_for_model/microbiology_preprocessed.pkl")
df_microbiology=df_microbiology.add_prefix('MICRO_BIO_')    

if IS_RUN_SURGERY:
    df_suregry=f_preprocess_surgery()
else:    
    df_suregry=pd.read_pickle(data_folder_path+ "surgery/surgery_for_model/suregry_preprocessed.pkl")
df_suregry=df_suregry.add_prefix('SURGERY_')    
    
if IS_RUN_HOSP_PREV_YEAR:
    print("run HOSP PREV seperately")
else:
    df_prev=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\adm_previous_year\adm_previous_year_v.pkl")
df_prev=df_prev.add_prefix('PREV_')    

if IS_RUN_ISHPUZIM:
    df_ishpuzim=f_preprocess_ishpuzim()
else:
    df_ishpuzim=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\ishpuzim_indicators\ishpuzim_for_model\ishpuzim_preprocessed.pkl")
df_ishpuzim=df_ishpuzim.add_prefix('ISHP_')  


if IS_RUN_DIAGNOSES_BG:
    df_diagnoses_bg=f_preprocess_diagnoses_bg()
else:
    df_diagnoses_bg=pd.read_pickle(r"C:/Users/orlyk/readmissions/project/preprocessed/diagnoses/diagnoses_for_model/bg_table_block261.pkl")
df_diagnoses_bg=df_diagnoses_bg.add_prefix('DIAG_BG_')  





  

df_merged=pd.merge(df,df_VS,how="left",right_on="VS_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_diag_adm,how="left",right_on="DIAG_CaseNum_adm",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_diag_disch,how="left",right_on="DIAG_CaseNum_dis",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_blood_count,how="left",right_on="LABS_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_blood_chem,how="left",right_on="LABS_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_ventilation,how="left",right_on="VENT_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_CCI,how="left",right_on="CCI_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_blood_coag,how="left",right_on="LABS_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_microbiology,how="left",right_on="MICRO_BIO_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_suregry,how="left",right_on="SURGERY_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_prev,how="left",right_on="PREV_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_ishpuzim,how="left",right_on="ISHP_CaseNum",left_on="CaseNum")
df_merged=pd.merge(df_merged,df_diagnoses_bg,how="left",right_on="DIAG_BG_CaseNum",left_on="CaseNum")


#df_merged.drop(columns=['VS_CaseNum', 'LABS_CaseNum_x','LABS_CaseNum_y','LABS_CaseNum','DIAG_CaseNum_adm','DIAG_CaseNum_dis',
#                        "VENT_CaseNum","CCI_CaseNum","MICRO_BIO_CaseNum","SURGERY_CaseNum"], inplace=True)

df_merged.drop(columns=['VS_CaseNum', 'LABS_CaseNum_x','LABS_CaseNum_y','LABS_CaseNum','DIAG_CaseNum_adm',"DIAG_CaseNum_dis",
                        "VENT_CaseNum","CCI_CaseNum","MICRO_BIO_CaseNum","SURGERY_CaseNum","PREV_CaseNum","ISHP_CaseNum","DIAG_BG_CaseNum"], inplace=True)




if IS_DUMMIES:
    df_merged=pd.get_dummies(data=df_merged, columns=['dept_cat_adm', 'dept_cat_disch','entry_type','discharge_type'])
    #df_merged["MICRO_BIO_category_anaerobe"]=np.where(df_merged["MICRO_BIO_category_anaerobe"]>0,1,0)
    #df_merged["MICRO_BIO_category_candida"]= np.where(df_merged["MICRO_BIO_category_candida"]>0,1,0)
    #df_merged["MICRO_BIO_category_negative"]=np.where(df_merged["MICRO_BIO_category_negative"]>0,1,0)
    #df_merged["MICRO_BIO_category_positive"]=np.where(df_merged["MICRO_BIO_category_positive"]>0,1,0)
    #df_merged["MICRO_BIO_contaminant"]=np.where(df_merged["MICRO_BIO_contaminant"]>0,1,0)
    #df_merged["MICRO_BIO_is_positive_culture"]=np.where(df_merged["MICRO_BIO_is_positive_culture"]>0,1,0)

#fill missing with zeros in surgey and microbiology

surgery_cols = [col for col in df_merged.columns if 'SURGERY' in col]
df_merged[surgery_cols]=df_merged[surgery_cols].fillna(0)

microbio_cols = [col for col in df_merged.columns if 'MICRO_BIO' in col]
df_merged[microbio_cols]=df_merged[microbio_cols].fillna(0)


    
    
#feature engineering:

##age interactions
#df_num_max= df_merged.filter(regex=r'(max)')
#for col in df_num_max:
#    df_num_max[col]=df_num_max[col]*df_merged['age']
#    
# 
#df_num_max=df_num_max.add_suffix('_AGE_interaction')
#   
#df_merged=pd.merge(df_merged,df_num_max,left_on=None, right_on=None, left_index=True, right_index=True)        
#
#df_num_last= df_merged.filter(regex=r'(last)')
#for col in df_num_last:
#    df_num_last[col]=df_num_last[col]*df_merged['age']
#    
# 
#df_num_last=df_num_last.add_suffix('_AGE_interaction')
#   
#df_merged=pd.merge(df_merged,df_num_last,left_on=None, right_on=None, left_index=True, right_index=True)        
#
#normalize to first
#df_num_last= df_merged.filter(regex=r'(last)')
#for col in df_num_last:
#    df_num_last[col]=df_num_max[col]*df_merged['age']
#    
# 
#df_num_max=df_num_max.add_suffix('_AGE_interaction')
#   
#df_merged=pd.merge(df_merged,df_num_max,left_on=None, right_on=None, left_index=True, right_index=True)        
#






#rename feat
#df_merged=df_merged.rename(columns={"DIAG_NBody mass index (BMI)_adm": "Diag_BMI_adm",
#                   "DIAG_NDentofacial anomalies [including malocclusion] and other disorders of jaw_adm": "DIAG_NDentofacial_anomalies_and_other_disorders_of_jaw_adm",
#                   "DIAG_NHuman immunodeficiency virus [HIV] disease_adm":"DIAG_HIV_disease_adm",
#                   "DIAG_NMood [affective] disorders_adm":"DIAG_NMood_disorders_adm",
#                   "DIAG_BG_NBody mass index (BMI)": "Diag_BMI_bg",
#                   "DIAG_BG_NDentofacial anomalies [including malocclusion] and other disorders of jaw": "DIAG_NDentofacial_anomalies_and_other_disorders_of_jaw_bg",
#                   "DIAG_BG_NHuman immunodeficiency virus [HIV] disease":"DIAG_HIV_disease_bg",
#                   "DIAG_BG_NMood [affective] disorders":"DIAG_NMood_disorders_bg",
#                   "CATEGORY_NAcquired pure red cell aplasia [erythroblastopenia]":"CATEGORY_NAcquired pure red cell aplasia erythroblastopenia,
#                   "CATEGORY_NAcute nasopharyngitis [common cold]":"CATEGORY_NAcute nasopharyngitis [common cold]",              
#                   "CATEGORY_NAcute obstructive laryngitis [croup] and epiglottitis":"CATEGORY_NAcute obstructive laryngitis and epiglottitis",
#                   "CATEGORY_NAnogenital herpesviral [herpes simplex] infections":"CATEGORY_NAnogenital herpesviral herpes simplex infections",
#                   "CATEGORY_NAsymptomatic human immunodeficiency virus [HIV] infection status":"CATEGORY_NAsymptomatic human immunodeficiency virus HIV infection status",
#                   "CATEGORY_NBody mass index [BMI]":"CATEGORY_NBody mass index BMI",
#                   "CATEGORY_NChlamydial lymphogranuloma (venereum)":"CATEGORY_NChlamydial lymphogranuloma venereum",
#                   "CATEGORY_NChronic kidney disease (CKD)":"CATEGORY_NChronic kidney disease CKD",
#                   "CATEGORY_NComplications following (induced) termination of pregnancy":"CATEGORY_NComplications following induced termination of pregnancy",
#                   "CATEGORY_NContact with and (suspected) exposure to communicable diseases":"CATEGORY_NContact with and suspected exposure to communicable diseases",
#                   
#                   
#    })


df_merged.columns = df_merged.columns.str.replace('[', ' ').str.replace(']', ' ').str.replace(')', ' ').str.replace('(', ' ')

    
    # "DIAG_NBody mass index (BMI)_dis":"Diag_BMI_dis",
                  # "DIAG_NDentofacial anomalies [including malocclusion] and other disorders of jaw_dis":"DIAG_NDentofacial_anomalies_and_other_disorders_of_jaw_dis",
                  # "DIAG_NHuman immunodeficiency virus [HIV] disease_dis":"DIAG_HIV_disease_dis",
                  # "DIAG_NMood [affective] disorders_dis":"DIAG_NMood_disorders_adm"})


#df_HS=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\HS\df_HS_pop.pkl")
#l_labels=df_pop[["CaseNum","age","LABEL_HOSP"]]
#df_HS=pd.merge(l_labels,df_HS,on="CaseNum",how="left")
                  
                  
if IS_DIAG_BLOCK_ONLY:
    diag_category_cols = [col for col in df_merged.columns if 'DIAG_BG_CATEGORY' in col]
    df_merged=df_merged.drop(columns=[diag_category_cols])
    diag_chapter_cols = [col for col in df_merged.columns if 'DIAG_BG_CHAPTER' in col]
    df_merged=df_merged.drop(columns=[diag_chapter_cols])
    
    
    
    
    

df_merged.to_pickle(output_path+"temp_with_dummies.pkl")

df_merged_head=df_merged.head(5000)
df_merged_head.to_csv(output_path+"head_with_dummies.csv")
#df_HS.to_pickle(output_path+"HS_only.pkl")