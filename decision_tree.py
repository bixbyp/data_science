#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:51:58 2022

@author: bixbypeterson
"""

import pandas as pd
# import numpy as np
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, mean_squared_error as mse
from statsmodels.stats.outliers_influence import variance_inflation_factor


pd.options.display.max_columns = None

if __name__ == '__main__':
    #Read Medical Data into med_data data frame
    med_data = pd.read_csv('/Users/bixbypeterson/desktop/WGU/D208/medical_clean.csv')
    
    #Drop Columns not needed for Analysis
    med_data.drop(labels=['Customer_id','Interaction','UID','TimeZone','Population','Children','Income', 'City','State', 'County','CaseOrder','Item1','Item2','Item3','Item4','Item5','Item6','Item7','Item8','Zip','Lat','Lng','Job','Marital','Services', 'Initial_admin', 'Area' , 'HighBlood', 'Stroke', 'Complication_risk', 'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma'], axis = 1, inplace=True)
    
    #Create Dummy Variables for Categorical
    med_data = pd.get_dummies(med_data)
    
    #Feature Selection
    X = med_data[list(med_data.columns[:-2])] # Get Column list for VIF
    vif = pd.DataFrame() # Create a datafram for VIF results
    vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # Add VIF Factor to DataFrame
    vif["features"] = X.columns # Add VIF Features to DataFrame
    vif = vif.query('VIF_Factor < 5 or VIF_Factor == inf') # Find features that have VIF Factor less than 5 or INF (Infinity) 
    init_df = pd.DataFrame(data=med_data,columns=vif['features'].values) #Dataframe for features selected from VIF
    
    #Setting Values / Target values for Decision Tree
    values = init_df.drop('ReAdmis_Yes', axis = 1).values
    target = init_df['ReAdmis_Yes']
        
    #Export Clean Data Set
    init_df.to_csv('/Users/bixbypeterson/desktop/WGU/D209/Task 2/medical_data_dt_cleaned.csv',sep=',',index=True)
        
    #Split Train Test Values 
    #Using 80/20 split
    x_train, x_test, y_train, y_test = tts(values,target,test_size=0.2, stratify=target, random_state=1)

    #Export Train/Test Data
    #Creating Dataframe for x train/test and y train/test -- Avoid numpy array errors on export
    xtrain = pd.DataFrame(x_train, columns=['Age','VitD_levels','Doc_visits','Full_meals_eaten','vitD_supp','Additional_charges','Gender_Female','Gender_Male','Gender_Nonbinary','ReAdmis_No'])
    xtest = pd.DataFrame(x_test, columns=['Age','VitD_levels','Doc_visits','Full_meals_eaten','vitD_supp','Additional_charges','Gender_Female','Gender_Male','Gender_Nonbinary','ReAdmis_No'])
    ytrain = pd.DataFrame(y_train, columns=['Age','VitD_levels','Doc_visits','Full_meals_eaten','vitD_supp','Additional_charges','Gender_Female','Gender_Male','Gender_Nonbinary','ReAdmis_No'])
    ytest = pd.DataFrame(y_test, columns=['Age','VitD_levels','Doc_visits','Full_meals_eaten','vitD_supp','Additional_charges','Gender_Female','Gender_Male','Gender_Nonbinary','ReAdmis_No'])
    
    #Export Dataframes created
    xtrain.to_csv('/Users/bixbypeterson/desktop/WGU/D209/Task 2/medical_data_dt_xtrain.csv',sep=',',index=True)
    xtest.to_csv('/Users/bixbypeterson/desktop/WGU/D209/Task 2/medical_data_dt_xtest.csv',sep=',',index=True)
    ytrain.to_csv('/Users/bixbypeterson/desktop/WGU/D209/Task 2/medical_data_dt_ytrain.csv',sep=',',index=True)
    ytest.to_csv('/Users/bixbypeterson/desktop/WGU/D209/Task 2/medical_data_dt_ytest.csv',sep=',',index=True)

    #Decisiion Tree Classifer    
    dt = dtc(max_depth=2, random_state = 1)
    dt.fit(x_train, y_train) #Fit data
    pred = dt.predict(x_test) #Run predicition
    
    #Print Accuracy metrics
    print(f'\nDecision Tree Accuracy: {accuracy_score(y_test, pred)}')
    print(f'\nDecision Tree MSE: {mse(y_test,pred)**1/2}')