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





os.chdir(r"C:\Users\orlyk\readmissions\project\preprocessed\model_input")
file = 'temp_with_dummies.pkl'
df= pd.read_pickle (file)


path_xlsx="O:/research/syncope_giris/analysis/preprocess/prediction_models"
path_figures="O:/research/syncope_giris/analysis/plots/"

df=df.drop(columns=['CaseNum',  'EnterDate', 'ExitDate','LABEL_JUST_ER','enter_month','discharge_month','log_LOS'])


df=filter_input(df)

y=df["LABEL_HOSP"]
x=df.drop(columns=['LABEL_HOSP'])


patnums=x["PatNum"]
x=x.drop(columns=['PatNum'])
#features=list(x.columns)


gkf = list(GroupKFold(n_splits=2 ).split(x,y,patnums))
indx=[x[1] for x in gkf]

roc_value_array = []
xg_probs_all=pd.DataFrame()
xg_probs_df_test=pd.DataFrame()
xg_probs_df_train=pd.DataFrame()
sensitivity_s=pd.Series()

for i in range(len(indx)):  
    x_test=x.iloc[indx[i]]
    x_test_ind=list(x_test.index)
    x_train=x.loc[~x.index.isin(x_test_ind)]
    
    y_test=y.iloc[indx[i]]
    y_test_ind=list(y_test.index)
    y_train=y.loc[~y.index.isin(y_test_ind)]
    
    #feature selection
    fs_list=run_feature_selection(x_train,y_train,200)
    
    x_train=x_train[fs_list]  
    x_test=x_test[fs_list]     
    features=features=list(x_test.columns)
    
    #imputation
    x_train=run_impute(x_train)
    
    #over and/or under sampling
    x_train,y_train=run_over_under_sample(x_train, y_train)

    # scale
    x_train, x_test=run_scaler(x_train,x_test)
   
     
            
        
        
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


    #plot auc curves ####################
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
    
    ######################
    
    
    #print("number of features used in model: "+ str(len(features)))
    sensitivity_recall,specificity,ppv,f1=report_metrics(y_test, y_pred)
    
    #model.get_booster().feature_names = features

    #xgboost.plot_importance(model.get_booster(),max_num_features=15)
    #plt.show()
    # shap importance #######################
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


#    #plt.show()
#    ax = sns.stripplot(x=y_test, y=xg_probs,jitter=0.05)
#    #plt.title("Test (N=71)", fontdict=None, loc='center')
#    plt.ylabel ("probabilities")
#    plt.show()
#    plt.figure(figsize=(8,5))
    
    
    #prob plot####################################
    ax = sns.boxplot(x=y_test, y=xg_probs)
    #plt.title("Test (N=71)", fontdict=None, loc='center')
    plt.ylabel ("probabilities")
    plt.show()


    
    xg_probs=pd.Series(xg_probs)
    xg_probs_df_test=pd.concat([xg_probs_df_test, xg_probs],axis=1)
    xg_probs_df_test=xg_probs_df_test.rename(columns={0: i})
    xg_probs_test_mean=xg_probs_df_test.mean(axis=1)
    
    xg_probs_train=pd.Series(xg_probs_train)
    xg_probs_df_train=pd.concat([xg_probs_df_train, xg_probs_train],axis=1)
    xg_probs_df_test=xg_probs_df_test.rename(columns={0: i})
    xg_probs_train_mean=xg_probs_df_train.mean(axis=1)

#    sensitivity=pd.Series(sensitivity_recall)
#    sensitivity_s=pd.concat([sensitivity_s, sensitivity],axis=0)
#    sensitivity_mean=sensitivity_s.mean()
#    
    def summarize_metrics(param):
        if i==0:
            global param_s
            param_s=pd.Series()
        param=pd.Series(param)
        param_s=pd.concat([param_s, param],axis=0)
        param_mean=param_s.mean()
        
        return param_s,param_mean
    
        
    
    sensitivity_s,sensitivity_mean=summarize_metrics(sensitivity_recall)
    
    
    
    
    
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
 