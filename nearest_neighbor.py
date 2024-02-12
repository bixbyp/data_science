#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:51:58 2022

@author: bixbypeterson
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler as SC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_auc_score as AUC, confusion_matrix

pd.options.display.max_columns = None

if __name__ == '__main__':
    #Read Medical Data into med_data data frame
    med_data = pd.read_csv('/Users/bixbypeterson/desktop/WGU/D208/medical_clean.csv')
    
    med_data.drop(labels=['Customer_id','UID','CaseOrder','Interaction'], axis = 1, inplace = True)
    med_data = pd.get_dummies(med_data)
        
    values = med_data.drop('ReAdmis_Yes', axis = 1).values
    target = med_data['ReAdmis_Yes']
    

    med_select = SelectKBest(score_func=f_regression, k = 8)
    med_fitted = med_select.fit_transform(values,target) # Feature Scaling
    med_select_cols = med_data.iloc[:,med_select.get_support(indices = True)]
    select_values = pd.DataFrame(data = med_data, columns=med_select_cols.columns.values)
            
    print(f'Selected Features:\n{select_values.dtypes}')
    
    #Export Clean Data Set
    select_values.to_csv('/Users/bixbypeterson/desktop/WGU/D209/medical_data_knn_cleaned.csv',sep=',',index=True)
    values = select_values
        
    #Split Train Test Values 
    #Using 80/20 split
    x_train, x_test, y_train, y_test = train_test_split(values,target,test_size=0.2, stratify=target)
    
    #Export Train/Test Data
    x_train.to_csv('/Users/bixbypeterson/desktop/WGU/D209/medical_data_knn_xtrain.csv',sep=',',index=True)
    x_test.to_csv('/Users/bixbypeterson/desktop/WGU/D209/medical_data_knn_xtest.csv',sep=',',index=True)
    y_train.to_csv('/Users/bixbypeterson/desktop/WGU/D209/medical_data_knn_ytrain.csv',sep=',',index=True)
    y_test.to_csv('/Users/bixbypeterson/desktop/WGU/D209/medical_data_knn_ytest.csv',sep=',',index=True)
    
    #Feature Scaling
    # scale_x = SC()
    # x_train = scale_x.fit_transform(x_train)
    # x_test = scale_x.transform(x_test)
    
    #Find Optimal Neighbors
    error_rate = []
    for i in range(1,30):
        knn = KNN(n_neighbors = i)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        error_rate.append(np.mean(pred != y_test))
    
    #Extract Optimal Neighbors
    opt_k_value = error_rate.index(min(error_rate))+1
    if opt_k_value == 1:
        opt_k_value = 10
    else:
        opt_k_value = opt_k_value
        
    print(f'\nOptimal Neighbors: {opt_k_value}')
    
    #KNN using Optimal Neighbors
    knn = KNN(n_neighbors = opt_k_value)
    knn.fit(x_train, y_train)
    med_pred = knn.predict(x_test)
    
    #KNN Results
    print(f'\nKNN Model Test Score / Accuracy: {knn.score(x_test,y_test)}')
    print(f'\nKNN Model Train Score / Accuracy: {knn.score(x_train,y_train)}')
    print(f'\nKNN Area Under Curve: {AUC(y_test, med_pred)}')
    print(f'\nKNN Confusion Matrix: \n{confusion_matrix(y_test,med_pred)}')
    