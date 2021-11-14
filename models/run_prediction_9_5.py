# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:02:48 2021

@author: orlyk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:43:25 2021

@author: orlyk
"""

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
file = 'internal_with_dummies.pkl'
df= pd.read_pickle (file)

df['EnterDate'] = pd.to_datetime(df['EnterDate'])
#df_test=df[df["EnterDate"]>'2019-07-01 23:59:59' & df["EnterDate"]>'2019-07-01 23:59:59']






#print(df.shape)
#
##age 70
##age 85
##cci 2
##cci 6

##################temp!!!!!!!!!!!!!
# df=df.head(500)



filtering_params_df=pd.read_csv(r"O:\OrlI\readmissions\code\prediction_models\functions\filtering_params.csv")
global filtering_params
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







#####add temp
#df["wid_children"]=df["ChildrenNum"]*df["family_stat_widow"]
#df["single_children"]=df["ChildrenNum"]*df["family_stat_single"]
#


df=filter_input(df)
df = df.fillna(value=np.nan)


############  temp



##################################################################
df.to_pickle(path+'/input_df.pkl' )

df=df.drop(columns=['CaseNum',  'EnterDate', 'ExitDate','LABEL_JUST_ER','enter_month','discharge_month','log_LOS'])

from sklearn.utils import shuffle
df = shuffle(df)

y=df["LABEL_HOSP"]
x=df.drop(columns=['LABEL_HOSP'])


patnums=x["PatNum"]
x=x.drop(columns=['PatNum'])
#features=list(x.columns)


if is_only_HS:
    x=x.drop(columns=[ 'age', 'gender', 'LOS','HB_Under12', 'Sodium_under135', 'ActiveCancer', 'Procedure_Flg',
       'LOS_over5', 'nonElective', 'Previos_admissions_cnt'])





gkf = list(GroupKFold(n_splits=int(filtering_params['cv']) ).split(x,y,patnums))
indx=[x[1] for x in gkf]

#initiate
probs_test_df=pd.DataFrame()
probs_train_df=pd.DataFrame()
roc_auc_s =pd.Series()
roc_auc_train_s =pd.Series()
sensitivity_s=pd.Series()
specificity_s=pd.Series()
ppv_s=pd.Series()
f1_s=pd.Series()
pr_auc_s=pd.Series()

shap_values_mat=np.empty([0,x.shape[1]])
X_importance_mat=pd.DataFrame()
importance_contcat=pd.DataFrame()
features_combined=pd.DataFrame()




 
if int(filtering_params['cv'])>0: 
    for i in range(len(indx)):  
        x_test=x.iloc[indx[i]]
        x_test_ind=list(x_test.index)
        x_train=x.loc[~x.index.isin(x_test_ind)]
        
        y_test=y.iloc[indx[i]]
        y_test_ind=list(y_test.index)
        y_train=y.loc[~y.index.isin(y_test_ind)]
        
        #feature selection
        
        if filtering_params['Feature_selection'] =='chi' :
            #catCols = [col for col in x_train.columns if x_train[col].nunique==2]
            cat_cols=get_categorical(x_train)
            cont_cols=list(x_train.drop(cat_cols, axis = 1).columns)

            fs_list_cat=run_feature_selection(x_train[cat_cols],y_train,"chi",20)
            fs_list_cont=run_feature_selection(x_train[cont_cols],y_train,"shap",30)
            
            fs_list=fs_list_cat+fs_list_cont
            
        else: 
           # if filtering_params['Feature_selection'] =='l1' :              
            fs_list=run_feature_selection(x_train,y_train,filtering_params['Feature_selection'],100)
        
        
        #make these items are still in if they were ommitted in fs 
#        if filtering_params['Feature_selection'] !='0' :
#            must_haves=[ 'HS_HB_Under12', 'HS_Sodium_under135', 'HS_ActiveCancer', 'HS_Procedure_Flg',
#           'HS_LOS_over5', 'HS_nonElective']
#            
#            for feat in must_haves:
#                if feat not in fs_list:
#                    fs_list.append(feat)
            
        
        
        x_train=x_train[fs_list]  
        x_test=x_test[fs_list]     
        features=features=list(x_test.columns)
        #feature_array= pd.Series(features)
        feature_array= pd.DataFrame(features)

        feature_array.to_csv(path+"/features/feature_list"+str(i)+".csv",index=False)
        #features_combined=pd.concat([features_combined,feature_array],axis=0)
        #features_combined=features_combined.rename(columns={0: "feature"})
        #features_combined["one"]=1
            
        #imputation
        x_train,x_test=run_impute(x_train,x_test)
        
        #over and/or under sampling
        x_train,y_train=run_over_under_sample(x_train, y_train)
    
        # scale
        x_train, x_test=run_scaler(x_train,x_test)
       
        
       #train_model 
        train_model, model,y_pred,probs_test,probs_train=run_prediction_model(x_train, y_train,x_test,y_test)
        
              
        #calssification metrics
        sensitivity_recall,specificity,ppv,f1,auc_test,auc_train,accuracy,pr_auc=report_metrics(y_train,y_test, y_pred,probs_train,probs_test,i)                    
                                                                                         
        
        specificity_s,specificity_mean=summarize_metrics(specificity,specificity_s)
        sensitivity_s,sensitivity_mean=summarize_metrics(sensitivity_recall,sensitivity_s)
        ppv_s,ppv_mean=summarize_metrics(ppv,ppv_s)
        f1_s,f1_mean=summarize_metrics(f1,f1_s)
        roc_auc_s,roc_auc_mean=summarize_metrics(auc_test,roc_auc_s)
        roc_auc_train_s,roc_auc_train_mean=summarize_metrics(auc_train,roc_auc_train_s)
        pr_auc_s,pr_auc_mean=summarize_metrics(pr_auc,pr_auc_s)
        
        case_=pd.Series(list(x_test.index))
        case_.to_csv(path + "/casenums"+str(i)+".csv") 
    
        # shap importance #######################
        if filtering_params['algo']=='xgb':
         
            X_importance=pd.DataFrame(x_test,columns=features)
            X_importance_mat=pd.concat([X_importance_mat,X_importance])
            explainer = shap.TreeExplainer(train_model)
            shap_values = explainer.shap_values(X_importance)
            
            shap.summary_plot(shap_values, X_importance)
            plt.savefig(path + '/SHAP/SHAP_figures/SHAP_swarm_'+str(i)+".png") 
            plt.figure()
            shap.summary_plot(shap_values, X_importance, plot_type='bar')
            plt.savefig(path + '/SHAP/SHAP_figures/SHAP_bar_'+str(i)+".png") 
            plt.figure()
            
        
            shap_sum = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
            importance_df.columns = ['column_name', 'shap_importance']
            importance_contcat=pd.concat([importance_contcat,importance_df],axis=1)
 
            
            #importance_df = importance_df.sort_values('shap_importance', ascending=False)
             
            importance_df.to_csv(path+"/SHAP/SHAP_importance_list"+str(i)+".csv",index=False)
            importance_contcat.to_csv(path+"/SHAP/shap_all_iter.csv",index=False)
    
    
    
        #plot auc curves #################### OPTIONAL
        
        ######################
        
        
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
            
            importance_contcat=pd.concat([importance_contcat,dff],axis=1)

            dff.to_csv(path+"/coefs/coefs_list"+str(i)+".csv")#,index=False)
            importance_contcat.to_csv(path+"/coefs/coefs_all_iter.csv")#,index=False)


            dff_for_plot=dff.head(20)
            
            
           # dff_for_plot.plot(kind='barh', figsize=(9, 7))
          
            #plt.savefig(path + '/coefs/coefs_figures/coefs_bar_'+str(i)+".png") 
            #plt.figure()
            
        elif filtering_params['algo']=='RF':
            fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

            # Display
            fi.head(60)
            importance20=fi.head(20)
            plt.figure(figsize=(8,10))
            
            ax = sns.barplot(x="importance", y="feature", data=importance20,color="steelblue")

            plt.figure( )
        
        
        
 
        
        #prob plot####################################
        ax = sns.boxplot(x=y_test, y=probs_test)#,whis=[5, 95],showfliers=False)
        #plt.title("Test (N=71)", fontdict=None, loc='center')
        plt.ylabel ("probabilities") 
        plt.ylim(0, 1)
        plt.show()
    
    
        
        probs_test=pd.DataFrame(probs_test)
        probs_test=probs_test.rename(columns={0: "prob_test"})
        y_test_tmp=y_test.reset_index().drop(["index"], axis=1)
        probs_test=pd.merge(y_test_tmp,probs_test,left_index=True, right_index=True)
    
        probs_test.to_csv(path+"/probs/test_"+str(i)+".csv",index=False)
    
    
        probs_test_df=pd.concat([probs_test_df, probs_test],axis=0)
        
        probs_train=pd.DataFrame(probs_train)
        probs_train=probs_train.rename(columns={0: "prob_train"})
        y_train_tmp=y_train.reset_index().drop(["index"], axis=1)
        probs_train=pd.merge(y_train_tmp,probs_train,left_index=True, right_index=True)
        
        probs_train.to_csv(path+"/probs/train_"+str(i)+".csv",index=False)

    
        probs_train_df=pd.concat([probs_train_df, probs_train],axis=0)
        
           

        
        
        #probs_test_df=probs_test_df.rename(columns={0: i})
        #probs_test_mean=probs_test_df.mean(axis=1)
        
    #    probs_train=pd.Series(probs_train)
    #    probs_train_df=pd.concat([probs_train_df, probs_train],axis=1)
    #    probs_train_df=probs_train_df.rename(columns={0: i})
    #    probs_train_mean=probs_train_df.mean(axis=1)
    
    
    
    
    #create final metrics df
    metrics_df=create_metrics_df(roc_auc_train_s,roc_auc_s,sensitivity_s,
                                 specificity_s,ppv_s,f1_s,pr_auc_s,
                                 roc_auc_train_mean,roc_auc_mean,sensitivity_mean,
                                 specificity_mean,ppv_mean,f1_mean,pr_auc_mean)    
        
     
    #create final roc curves 
    plt.figure(figsize=(6,6))
    logit_roc_auc1 = roc_auc_score (probs_test_df.iloc[:,0], probs_test_df.iloc[:,1])
    fpr1, tpr1, thresholds1 = roc_curve (probs_test_df.iloc[:,0], probs_test_df.iloc[:,1])
    plt.plot (fpr1, tpr1, label='AUC   = %0.2f' % logit_roc_auc1)
       
    logit_roc_auc2 = roc_auc_score (probs_train_df.iloc[:,0], probs_train_df.iloc[:,1])
    fpr1, tpr1, thresholds1 = roc_curve (probs_train_df.iloc[:,0], probs_train_df.iloc[:,1])
    plt.plot (fpr1, tpr1, label='AUC   = %0.2f' % logit_roc_auc2)
       
    
    plt.plot ([0, 1], [0, 1], 'r--')
    plt.xlim ([0.0, 1.0])
    plt.ylim ([0.0, 1.05])
    plt.xlabel ('False Positive Rate')
    plt.ylabel ('True Positive Rate')
    #plt.title ('Receiver operating characteristic')
    plt.legend (loc="lower right")
    plt.savefig(path + '/metrics/curve_figures'+"/roc_auc_all.png") 
    plt.figure()
    
    #write to file
    filtering_params_df.to_excel(path+"/filtering_params.xlsx",index=False)
    
    metrics_df.to_excel(path+"/metrics/classification_metrics.xlsx",index=False)
    probs_test_df.to_excel(path+"/probs/probs_test_all.xlsx",index=False)
    
    
    
