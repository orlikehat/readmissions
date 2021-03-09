# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:38:05 2021

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

def feature_selection_func (df,y,num_feats):
    
    
     #chi  todo
#        
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(df, y)
    chi_support = chi_selector.get_support()
    chi_feature = df.loc[:,chi_support].columns.tolist()
    print(str(len(chi_feature)), 'selected features - chi2')
#    
#    else:
#        chi_support=False 
#    
    
    
    
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
                                        'Random Forest':embeded_rf_support,"chi":chi_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    feature_selection_df.head(num_feats)
    
    fs_list=feature_selection_df[feature_selection_df["Total"]>=2]
    fs_list=list(fs_list["Feature"])
    print(str(len(fs_list)) + "features remain" )
    return fs_list,feature_selection_df


def feature_selection_l1 (x,y,num_feats):
    
    
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


def feature_selection_chi (x,y,num_feats):

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
    
    chi_p=pd.concat([cols,ser_p],axis=1)
    chi_p=chi_p.sort_values(by=['sig'],ascending=True).head(num_feats)
    diag_list = chi_p[0].tolist()
    
    return diag_list




def feature_selection_shap(x,y,n_features):

#   
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
    importance_df=importance_df.head(n_features)
#    X_interaction = X_importance
#    shap_interaction_values = shap.TreeExplainer(train_model).shap_interaction_values(X_interaction)
#    shap.summary_plot(shap_interaction_values, X_interaction)
    fs_features=list(importance_df.column_name)
    print ("SHAP importance: " + str(len(fs_features)) + " features selected")
    
    
    
    
    
    return fs_features
    






























#df=pd.read_pickle(r"C:\Users\orlyk\readmissions\project\preprocessed\model_input\temp_with_dummies.pkl")

#define categorical features:
#month_cols = [col for col in df.columns if 'month' in col]
#dept_cols=[col for col in df.columns if 'dept' in col]
#entry_cols=[col for col in df.columns if 'entry' in col]
#diag_cols=[col for col in df.columns if 'DIAG' in col]
#vent_cols=[col for col in df.columns if 'VENT' in col]
#cci_cols=[col for col in df.columns if 'bg_cci' in col]
#label_cols=[col for col in df.columns if 'LABEL' in col]
#discharge_type_cols=[col for col in df.columns if 'discharge_type' in col]
#
#cat_list=month_cols+dept_cols+ entry_cols+vent_cols+label_cols+cci_cols+diag_cols+discharge_type_cols
#
#df_cat=df[cat_list]
#df_cat = df_cat.fillna(df_cat.mode().iloc[0])
#
#df_cat = df_cat.astype(str)
#x_cat=df_cat.drop(columns=['LABEL_HOSP','LABEL_JUST_ER'])
#
#df_num=df.drop(columns=cat_list)
#df_num=df_num.drop(columns=['CaseNum','PatNum','EnterDate','ExitDate',
#                        'year','before_2017','LABS_CaseNum'])
#
#x_num=x_num=df_num
#x_num = x_num.fillna(x_num.mode().iloc[0])
#
#x_num_scaled=(x_num-x_num.mean())/x_num.std()
#
#y=df_cat["LABEL_HOSP"]
#
#num_feats=round(df.shape[1]/2)
#num_feats_cat=30
#
#df_features=pd.merge(x_cat,x_num_scaled,left_index=True, right_index=True)
#df_fs_all=feature_selection_methods(df_features,y,num_feats,False)





