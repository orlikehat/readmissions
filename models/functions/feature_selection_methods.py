# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:54:16 2021

@author: orlyk
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import xgboost
import shap
import numpy as np

from matplotlib import pyplot

# logistic regression with l1, can be changed to l2
def feature_selection_l1 (x,y,num_feats):
    print ("run l1 fs")
    
    l1_ratio = 0.7  # L1 weight in the Elastic-Net regularization
            
    clf_l1_LR = LogisticRegression(penalty='l1', tol=0.01, solver='saga')
    clf_l2_LR = LogisticRegression(penalty='l2', tol=0.01, solver='saga')
    clf_en_LR = LogisticRegression(penalty='elasticnet', solver='saga',
                                       l1_ratio=l1_ratio, tol=0.01)
    clf_l1_LR.fit(x, y)
    clf_l2_LR.fit(x, y)
    clf_en_LR.fit(x, y)
    
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()
    coef_en_LR = clf_en_LR.coef_.ravel()
    
    cols=list(x.columns)
    cols= pd.Series(cols) 
    
    coefs_l1= pd.Series(coef_l1_LR) 
    coefs_l1=pd.concat([cols,coefs_l1],axis=1)
    
    coefs_l1=coefs_l1[coefs_l1[1]!=0]
    coefs_l1[1]=coefs_l1[1].abs()
    coefs_l1=coefs_l1.sort_values(by=[1],ascending=False)
    coefs_l1=coefs_l1.head(num_feats)
    diag_list_l1 = coefs_l1[0].tolist()
        
   
    coefs_l2= pd.Series(coef_l2_LR) 
    coefs_l2=pd.concat([cols,coefs_l2],axis=1)
    
    coefs_l2=coefs_l2[coefs_l2[1]!=0]
    coefs_l2[1]=coefs_l2[1].abs()
    coefs_l2=coefs_l2.sort_values(by=[1],ascending=False)
    coefs_l2=coefs_l2.head(num_feats)
    diag_list_l2 = coefs_l2[0].tolist()
    
    return diag_list_l1#todo can also return l2 for later

#combined RFE, ridge l2, random forest feature importance - features that are selected in at least 2 of the methods are in
def feature_selection_combined (df,y,num_feats):
    print ("run combined fs")

        
      
     #chi  todo
#        
#    chi_selector = SelectKBest(chi2, k=num_feats)
#    chi_selector.fit(df, y)
#    chi_support = chi_selector.get_support()
#    chi_feature = df.loc[:,chi_support].columns.tolist()
#    print(str(len(chi_feature)), 'selected features - chi2')
#    
#    else:
#        chi_support=False 

    
    #RFE
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=30, verbose=5)
    rfe_selector.fit(df, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = df.loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features - RFE_LR')


    #ridge

    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embeded_lr_selector.fit(df, y)
    
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = df.loc[:,embeded_lr_support].columns.tolist()
    print(str(len(embeded_lr_feature)), 'selected features- l2')

    #random forrest

    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(df, y)
    
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = df.loc[:,embeded_rf_support].columns.tolist()
    print(str(len(embeded_rf_feature)), 'selected features - RF')
   
        
    #combine
    feature_name=list(df.columns)
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                        'Random Forest':embeded_rf_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    
    fs_list=feature_selection_df[feature_selection_df["Total"]>=2]
    fs_list=list(fs_list["Feature"])
    
    return fs_list

#xgboost and SHAP
def feature_selection_shap(x,y,num_feats):
    print ("run SHAP fs")

    model = xgboost.XGBClassifier(scale_pos_weight=11)
    train_model = model.fit(x,y)
    
    X_importance = x
    explainer = shap.TreeExplainer(train_model)
    shap_values = explainer.shap_values(X_importance)
    #shap.summary_plot(shap_values, X_importance, plot_type='bar')
    
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    #importance_df=importance_df[importance_df["shap_importance"]>0]
    importance_df=importance_df.head(num_feats)
#    
    fs_features=list(importance_df.column_name)
    return fs_features

#chi squared
def feature_selection_chi (x,y,num_feats):
    print ("run chi squared fs")

    x=x.astype(str)
    y=y.astype(str)
    diag_cols=list(x.columns)
    ser_p = pd.Series() 
    for col in diag_cols:
        data_crosstab = pd.crosstab(x[col],y)
        stat, p, dof, expected = chi2_contingency(data_crosstab)
        p=pd.Series(p, dtype='float')
        ser_p=ser_p.append(p)
        
    ser_p=ser_p.reset_index()
    ser_p=ser_p.rename(columns={0: "sig"})
    ser_p=ser_p.drop(columns=['index'])
    
    #chi_p=pd.concat([diag_cols,ser_p],axis=1)
    chi_p = pd.Series(diag_cols) 
    chi_p=chi_p.rename("feature")
    chi_p=pd.merge(chi_p,ser_p,left_index=True, right_index=True)
    chi_p=chi_p.sort_values(by=['sig'],ascending=True).head(num_feats)
    diag_list = chi_p["feature"].tolist()
    
    return diag_list




def run_feature_selection(x_train,y_train,num_feats):
    filtering_params=pd.read_csv(r"C:\Users\orlyk\readmissions\project\git_code\readmissions\models\functions\filtering_params.csv")
    filtering_params=dict(zip(list(filtering_params.condition), list(filtering_params.value)))

    if filtering_params['Feature_selection']=="l1":
        x_fs=x_train
        x_fs=x_fs.dropna(thresh=0.9*(x_fs.shape[1]), axis=0)
        x_fs = x_fs.fillna(x_fs.mode().iloc[0])

        x_fs=x_fs.astype(int)
        y_fs=y_train.loc[x_fs.index]
        
        #num_feats=round(x_fs.shape[1]/5)
         
        fs_list=feature_selection_l1(x_fs,y_fs,num_feats)

        #x_train=x_train[fs_list]
        #x_test=x_test[fs_list]
        
    elif filtering_params['Feature_selection']=="multiple":
        x_fs=x_train
        x_fs=x_fs.dropna(thresh=0.9*(x_fs.shape[1]), axis=0)
        x_fs = x_fs.fillna(x_fs.mode().iloc[0])

        #x_fs=x_fs.astype(int)
        y_fs=y_train.loc[x_fs.index]
        
        #num_feats=round(x_fs.shape[1]/5)
        
        fs_list=feature_selection_combined(x_fs,y_fs,num_feats)

    elif filtering_params['Feature_selection']=="shap": 
        fs_list=feature_selection_shap(x_train,y_train,num_feats)
        
    elif filtering_params['Feature_selection']=="chi":
        x_fs=x_train
        x_fs=x_fs.dropna(thresh=0.9*(x_fs.shape[1]), axis=0)
        x_fs = x_fs.fillna(x_fs.mode().iloc[0])

        x_fs=x_fs.astype(int)
        y_fs=y_train.loc[x_fs.index]
        
        #num_feats=round(x_fs.shape[1]/5)
         
        fs_list=feature_selection_chi(x_fs,y_fs,num_feats)
        
    else: 
        print("no feature selection")
        fs_list=list(x_train.columns)
        
    return fs_list
       
       