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
from helper_functions import *
from feature_selection import feature_selection_func
from feature_selection import feature_selection_l1
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
import shap
from feature_selection_SHAP import feature_selection_shap
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from functions.filter_input import filter_input



os.chdir(r"C:\Users\orlyk\readmissions\project\preprocessed\model_input")
file = 'temp_with_dummies.pkl'
df= pd.read_pickle (file)


path_xlsx="O:/research/syncope_giris/analysis/preprocess/prediction_models"
path_figures="O:/research/syncope_giris/analysis/plots/"

df=df.drop(columns=['CaseNum',  'EnterDate', 'ExitDate','LABEL_JUST_ER','enter_month','discharge_month','log_LOS'])



df=filter_input(df)

######################################################


y=df["LABEL_HOSP"]
x=df.drop(columns=['LABEL_HOSP'])


patnums=x["PatNum"]
x=x.drop(columns=['PatNum'])
features=list(x.columns)


gkf = list(GroupKFold(n_splits=10 ).split(x,y,patnums))
indx=[x[1] for x in gkf]
roc_value_array = []
xg_probs_all=pd.DataFrame()

for i in range(len(indx)):  
    x_test=x.iloc[indx[i]]
    x_test_ind=list(x_test.index)
    x_train=x.loc[~x.index.isin(x_test_ind)]
    
    y_test=y.iloc[indx[i]]
    y_test_ind=list(y_test.index)
    y_train=y.loc[~y.index.isin(y_test_ind)]
    
   
    if IS_FEATURE_SELECTION:
        x_fs=x_train
        x_fs = x_fs.fillna(x_fs.mode().iloc[0])
        y_fs=y_train
        
        all_fs=pd.concat([x_train,y_train],axis=1)
        all_fs=all_fs.dropna()
        #all_fs = all_fs.applymap(str)
        x_fs=all_fs.drop(columns=['LABEL_HOSP'])
        y_fs=all_fs["LABEL_HOSP"]     
        
        
        num_feats=round(x_fs.shape[1]/5)
        
        #fs_list,df_fs_all=feature_selection_l1(x_fs,y_fs,num_feats)
        fs_list=feature_selection_l1(x_fs,y_fs,num_feats)

        x_train=x_train[fs_list]
        x_test=x_test[fs_list]
    
    if IS_FEATURE_SELECTION_SHAP:
        
        fs_shap_features=feature_selection_shap(x_train,y_train)
        
        x_train=x_train[fs_shap_features]
        x_test=x_test[fs_shap_features]
        
        
        
          
    
    
        
    if IS_FEATURE_SELECTION_diagnoses_only:
        diag_cols = [col for col in x_train.columns if 'DIAG' in col]
        non_diag_cols= [col for col in x_train.columns if 'DIAG' not in col]
        
        x_train_diag=x_train[diag_cols]
        x_test_diag=x_test[diag_cols]
                
        x_train_non_diag=x_train[non_diag_cols]
        x_test_non_diag=x_test[non_diag_cols]
        
        x_train_diag=x_train_diag.dropna()
        y_train_diag=y_train[x_train_diag.index]
        
                       
        num_feats=round(x_train_diag.shape[1]/15)
        
        fs_list=feature_selection_l1(x_train_diag,y_train_diag,num_feats)
        fs_shap_features=feature_selection_shap(x_train_non_diag,y_train)

    
        x_train_diag=x_train_diag[fs_list]
        x_test_diag=x_test_diag[fs_list]
        
        x_train_non_diag=x_train_non_diag[fs_shap_features]
        x_test_non_diag=x_test_non_diag[fs_shap_features]
        
        
        
        
        
        x_train=pd.concat([x_train_diag,x_train_non_diag],axis=1)
        x_test=pd.concat([x_test_diag,x_test_non_diag],axis=1)

        
    features=features=list(x_test.columns)


    if IS_OVER_UNDER_SAMPLING:  
        #impute:
        x_train=pd.DataFrame(SimpleImputer().fit_transform(x_train), columns = x_train.columns)
#        else:
#            imputer = KNNImputer(n_neighbors=8)
#            x=imputer.fit_transform(x) 
#            x=pd.DataFrame(data=x,columns=c_names)   
#            
        print(Counter(y_train))
        # define undersample strategy
        undersample = RandomUnderSampler(sampling_strategy=0.2)
        # fit and apply the transform
        x_train, y_train = undersample.fit_resample(x_train, y_train)
        # summarize class distribution
        print(Counter(y_train))
        counter = Counter(y_train)
        print(counter)
        # transform the dataset
        oversample = SMOTE(sampling_strategy=0.4)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        # summarize the new class distribution
        counter = Counter(y_train)
        print(counter)

        
    
    #feature selection
    
    

    if IS_SCALE:
        scaler = StandardScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
    
    else:
        x_train = x_train.as_matrix()
        x_test = x_test.as_matrix()
  
    
  
        
            
        
        
     #xgboost
    print("run XGB model") 
    model = xgboost.XGBClassifier(learning_rate= 0.1,max_depth= 6, alpha= 10,min_child_weight=6,scale_pos_weight=6)
    model = xgboost.XGBClassifier(subsample= 0.7, seed= 1001, scale_pos_weight= 6,
                                  min_child_weight= 5, max_depth= 9, learning_rate= 0.01, gamma= 0.5, colsample_bytree= 0.6)
    model = xgboost.XGBClassifier(subsample= 0.7, seed= 1001, scale_pos_weight= 6,
                                  min_child_weight= 5, max_depth= 9, learning_rate= 0.01, gamma= 0.5, colsample_bytree= 0.6)
    train_model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print('Model 1 XGboost Report %r' % (classification_report(y_test, y_pred)))
    print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, y_pred) * 100))
    
     
    xg_probs=train_model.predict_proba(x_test)[:, 1]
    xg_probs_train=train_model.predict_proba(x_train)[:, 1]

    roc_value = roc_auc_score(y_test, xg_probs)
    roc_value_train = roc_auc_score(y_train, xg_probs_train)

    print("AUC_test: " + str(roc_value))
    print("AUC_train: " + str(roc_value_train))

    plt.figure(figsize=(6,6))
    logit_roc_auc1 = roc_auc_score (y_test, xg_probs)
    fpr1, tpr1, thresholds1 = roc_curve (y_test, xg_probs)
    plt.plot (fpr1, tpr1, label='AUC   = %0.2f' % logit_roc_auc1)
   
    logit_roc_auc2 = roc_auc_score (y_train, xg_probs_train)
    fpr1, tpr1, thresholds1 = roc_curve (y_train, xg_probs_train)
    plt.plot (fpr1, tpr1, label='AUC   = %0.2f' % logit_roc_auc2)
   
    
        
    
    plt.plot ([0, 1], [0, 1], 'r--')
    plt.xlim ([0.0, 1.0])
    plt.ylim ([0.0, 1.05])
    plt.xlabel ('False Positive Rate')
    plt.ylabel ('True Positive Rate')
    #plt.title ('Receiver operating characteristic')
    plt.legend (loc="lower right")
    print("number of features used in model: "+ str(len(features)))
    sensitivity_recall,specificity,ppv,f1=report_metrics(y_test, y_pred)
    
    model.get_booster().feature_names = features

    xgboost.plot_importance(model.get_booster(),max_num_features=15)
    plt.show()
    
    X_importance=pd.DataFrame(x_train,columns=features)
    explainer = shap.TreeExplainer(train_model)
    shap_values = explainer.shap_values(X_importance)
    shap.summary_plot(shap_values, X_importance)
    shap.summary_plot(shap_values, X_importance, plot_type='bar')


    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    
#    X_interaction = X_importance
#    shap_interaction_values = shap.TreeExplainer(train_model).shap_interaction_values(X_interaction)
#    shap.summary_plot(shap_interaction_values, X_interaction)
#
#    plt.figure(figsize=(8,10))

    #y_test=np.where(y_test==1,"readmission","no_readmission")


    #plt.show()
    ax = sns.stripplot(x=y_test, y=xg_probs,jitter=0.05)
    #plt.title("Test (N=71)", fontdict=None, loc='center')
    plt.ylabel ("probabilities")
    plt.show()
    plt.figure(figsize=(8,5))
    
    ax = sns.boxplot(x=y_test, y=xg_probs)
    #plt.title("Test (N=71)", fontdict=None, loc='center')
    plt.ylabel ("probabilities")
    plt.show()


    
    xg_probs_df=pd.Series(xg_probs)
    xg_probs_all=pd.concat([xg_probs_all, xg_probs_df]).groupby(level=0).mean()
    
    roc_value_array.append(roc_value)
    
    avg_roc = np.mean(roc_value_array,axis=0)
    print("average roc test: "+str(avg_roc))
    #    X_importance = x_test
#    explainer = shap.TreeExplainer(train_model)
#    shap_values = explainer.shap_values(X_importance)
#    shap.summary_plot(shap_values, X_importance)
#    plt.show()
#    shap.summary_plot(shap_values, X_importance, plot_type='bar')
#    shap.summary_plot(shap_values, X_importance)
#
#    df_shap_values = pd.DataFrame(shap_values,columns=features)
#    X_importance=pd.DataFrame(x_test,columns=features)
#    shap.summary_plot(df_shap_values, X_importance)
#
#    viz = plot_roc_curve(model, x_test, y_test,
#                         name='ROC fold {}'.format(i),
#                         alpha=0.3, lw=1, ax=ax)
#    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#    interp_tpr[0] = 0.0
#    tprs.append(interp_tpr)
#    aucs.append(viz.roc_auc)
#
#ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#        label='Chance', alpha=.8)
#
#mean_tpr = np.mean(tprs, axis=0)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#std_auc = np.std(aucs)
#ax.plot(mean_fpr, mean_tpr, color='b',
#        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#        lw=2, alpha=.8)
#
#std_tpr = np.std(tprs, axis=0)
#tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                label=r'$\pm$ 1 std. dev.')
#
#ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#       title="Receiver operating characteristic example")
#ax.legend(loc="lower right")
##    

#    
 