# -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from functions.helper_functions import *
#from feature_selection import feature_selection_func
#from feature_selection import feature_selection_l1
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
import shap
#from feature_selection_SHAP import feature_selection_shap
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from functions.filter_input import filter_input
from functions.feature_selection_methods import * 
from functions.impute_input import *
from functions.over_under_sample import run_over_under_sample
from functions.scale_input import run_scaler
from functions.classification_metrics import *
from functions.helpers.get_catagorical import *
from datetime import datetime

from functions.prediction_algo  import *

is_only_HS=False


os.chdir(r"O:\OrlI\readmissions\preprocessed\model_input")
file = 'medical_with_dummies.pkl'
df= pd.read_pickle (file)

df['EnterDate'] = pd.to_datetime(df['EnterDate'])
#df_test=df[df["EnterDate"]>'2019-07-01 23:59:59' & df["EnterDate"]>'2019-07-01 23:59:59']

filtering_params_df=pd.read_csv(r"O:\OrlI\readmissions\code\prediction_models\functions\filtering_params.csv")
filtering_params=dict(zip(list(filtering_params_df.condition), list(filtering_params_df.value)))

global output_path
output_path="O:/OrlI/readmissions/model_results/"
#create folder output folder
directory=datetime.today().strftime('%d-%m-%y - %H%M')
global path
path = os.path.join(output_path, directory) 
os.mkdir(path)

#create sub-folders 
os.mkdir(path + '/metrics')
os.mkdir(path + '/metrics' + "/curve_figures")
os.mkdir(path + '/probs')
os.mkdir(path + '/SHAP')
os.mkdir(path + '/SHAP' + "/SHAP_figures")
os.mkdir(path + '/features')
os.mkdir(path + '/coefs' )
os.mkdir(path + '/coefs'+"/coefs_figures" )



if filtering_params['IS_VISIT']=='first':
    df=df.sort_values(by="EnterDate")
    df=df.drop_duplicates(subset=['PatNum'])
elif filtering_params['IS_VISIT']=='last': 
    df=df.sort_values(by="EnterDate")
    df=df.drop_duplicates(subset=['PatNum'],keep="last")
    
    
    




#####add temp
#df["wid_children"]=df["ChildrenNum"]*df["family_stat_widow"]
#df["single_children"]=df["ChildrenNum"]*df["family_stat_single"]
#

    
df=filter_input(df)
df = df.fillna(value=np.nan)

#if IS_FULL_MODEL=="0":
#    df=df[df["EnterDate"]<'2019-07-01 23:59:59']
#else:
#    run_full_model(df,date1,date2)

df_train=df[df["EnterDate"]<'2019-07-01 23:59:59']
df_test=df[df["EnterDate"]>'2019-07-01 23:59:59'] 
df_test=df_test[df_test["EnterDate"]<'2020-01-15 23:59:59'] 
df_test_original=df_test




df_train=df_train.drop(columns=['CaseNum','ExitDate','LABEL_JUST_ER','enter_month','discharge_month','log_LOS'])
df_test=df_test.drop(columns=['CaseNum','ExitDate','LABEL_JUST_ER','enter_month','discharge_month','log_LOS'])
#features=list(df_test.columns)



y_train=df_train["LABEL_HOSP"]
x_train=df_train.drop(columns=['LABEL_HOSP'])


y_test=df_test["LABEL_HOSP"]
x_test=df_test.drop(columns=['LABEL_HOSP'])


#atnums=x["PatNum"]
x_train=x_train.drop(columns=['PatNum','EnterDate'])
x_test=x_test.drop(columns=['PatNum','EnterDate'])








features=list(x_test.columns)


x_train,x_test=run_impute(x_train,x_test)

x_train, x_test=run_scaler(x_train,x_test)
       
        
   #train_model 
train_model, model,y_pred,probs_test,probs_train=run_prediction_model(x_train, y_train,x_test,y_test)

#calssification metrics
sensitivity_recall,specificity,ppv,f1,auc_test,auc_train,accuracy,pr_auc=report_metrics(y_train,y_test, y_pred,probs_train,probs_test,1)                    

ax = sns.boxplot(x=y_test, y=probs_test)#,whis=[5, 95],showfliers=False)
#plt.title("Test (N=71)", fontdict=None, loc='center')
plt.ylabel ("probabilities") 
plt.ylim(0.2, 0.9)
plt.show()


# shap importance #######################
        
if filtering_params['algo']=='xgb':
         
    X_importance=pd.DataFrame(x_test,columns=features)
    explainer = shap.TreeExplainer(train_model)
    shap_values = explainer.shap_values(X_importance)
    
    shap.summary_plot(shap_values, X_importance)
    plt.savefig(path + '/SHAP/SHAP_figures/SHAP_swarm.png') 
    plt.figure()
    shap.summary_plot(shap_values, X_importance, plot_type='bar')
    plt.savefig(path + '/SHAP/SHAP_figures/SHAP_bar.png') 
    plt.figure()
    
    
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
     
    importance_df.to_csv(path+"/SHAP/SHAP_importance_list.csv",index=False)
         
    


if filtering_params['algo']=='LR_l1':
        print("Non Zero weights:", np.count_nonzero(train_model.coef_))
        coeffs=train_model.coef_
        coeffs=np.transpose(coeffs)


        dff = pd.DataFrame(data=coeffs)

        print(dff.shape)
#features=pd.DataFrame(X_train)
        dff = pd.DataFrame(data=coeffs,index=features)
        dff=dff.rename(columns={0: "coeffs"})
        #dff["features"]=pd.Series(features,name="feature")
        
        dff=dff.reindex(dff["coeffs"].abs().sort_values(ascending=False).index)
        

        dff.to_csv(path+"/coefs/coefs_list"+".csv")#,index=False)
       # importance_contcat.to_csv(path+"/coefs/coefs_all_iter.csv")#,index=False)


        dff_for_plot=dff.head(20)

        dff_for_plot.plot(kind='barh', figsize=(9, 7))
        #dff_for_plot.title('LR model - coefficients')
        #dff_for_plot.axvline(x=0, color='.5')
        #dff_for_plot.subplots_adjust(left=.3)
        plt.savefig(path + '/coefs/coefs_figures/coefs_bar'+".png") 
        plt.figure()

    
    


















df_test_short=df_test_original[['CaseNum','EnterDate','dept_cat_disch_general_ICU',
                                'dept_cat_disch_geriatrics',
                                'dept_cat_disch_internal',
                                'dept_cat_disch_internal_ICU',
                                'dept_cat_disch_internal_special',
                                'dept_cat_disch_orthopedics',
                                'dept_cat_disch_surgical_ICU',
                                'dept_cat_disch_surgical_general',
                                'dept_cat_disch_surgical_special',
                                'dept_cat_disch_women','LABEL_HOSP'          
                                                                ]]
df_test_short=df_test_short.reset_index()
probs_test=pd.DataFrame(probs_test)
probs_test=probs_test.rename(columns={0: "prob_test"})
y_pred=pd.DataFrame(y_pred)
y_pred=y_pred.rename(columns={0: "y_pred"})


probs_test_df=pd.merge(df_test_short,probs_test ,  left_index=True, right_index=True)
probs_test_df=pd.merge(probs_test_df,y_pred ,  left_index=True, right_index=True)


df_internal=probs_test_df[(probs_test_df["dept_cat_disch_internal"]==1) |(probs_test_df["dept_cat_disch_internal_ICU"]==1)] 
df_surgical=probs_test_df[(probs_test_df["dept_cat_disch_surgical_special"]==1) |(probs_test_df["dept_cat_disch_surgical_special"]==1) |
            (probs_test_df["dept_cat_disch_orthopedics"]==1) | 
            (probs_test_df["dept_cat_disch_surgical_ICU"]==1)] 

df_internal_special=probs_test_df[(probs_test_df["dept_cat_disch_internal_special"]==1) | (probs_test_df["dept_cat_disch_general_ICU"]==1) ]        
df_medical=probs_test_df[(probs_test_df["dept_cat_disch_internal_special"]==1) | (probs_test_df["dept_cat_disch_general_ICU"]==1)| (probs_test_df["dept_cat_disch_internal"]==1) |(probs_test_df["dept_cat_disch_internal_ICU"]==1) ]  


frames=[df_internal,df_surgical,df_internal_special]
sensitivity_recall,specificity,ppv,f1,auc_test,auc_train,accuracy,pr_auc=report_metrics(df_internal["LABEL_HOSP"],df_internal["LABEL_HOSP"],df_internal["y_pred"], df_internal["prob_test"],df_internal["prob_test"],1)                    
sensitivity_recall,specificity,ppv,f1,auc_test,auc_train,accuracy,pr_auc=report_metrics(df_internal_special["LABEL_HOSP"],df_internal_special["LABEL_HOSP"],df_internal_special["y_pred"], df_internal_special["prob_test"],df_internal_special["prob_test"],1)                    
sensitivity_recall,specificity,ppv,f1,auc_test,auc_train,accuracy,pr_auc=report_metrics(df_surgical["LABEL_HOSP"],df_surgical["LABEL_HOSP"],df_surgical["y_pred"], df_surgical["prob_test"],df_surgical["prob_test"],1)                    
sensitivity_recall,specificity,ppv,f1,auc_test,auc_train,accuracy,pr_auc=report_metrics(df_medical["LABEL_HOSP"],df_medical["LABEL_HOSP"],df_medical["y_pred"], df_medical["prob_test"],df_medical["prob_test"],1)                    
    


probs_test_df.to_csv(path+"/probs/test_"+str(i)+".csv",index=False)

                                               
#features=list(x.columns)
















