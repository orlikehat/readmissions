# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:53:44 2020

@author: orlyk
"""

from sklearn.metrics import confusion_matrix


def report_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]   
    tp=cm[1][1]    
    sensitivity_recall=round(tp/(tp+fn),2)#.round(2)
    specificity=tn/(tn+fp)
    ppv=tp/(tp+fp)
    f1=2*tp/(2*tp+fp+fn)
    
    print("confusion matrix: ")
    print(cm)
    print("---------------------------------")
    print('sensitivity/recall = '+str(sensitivity_recall))
    print('specificity = '+str(specificity))
    print('ppv = '+str(ppv))
    print('f1 = '+str(f1))
    
    return(sensitivity_recall,specificity,ppv,f1)