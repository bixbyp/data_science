#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 09:50:07 2022

@author: bixbypeterson
"""

import pandas as pd
from statsmodels.formula.api import ols
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns



def gen_univariate (exp_vars, med_data):
    for i, col in enumerate(exp_vars):
           plt.hist(med_data[f'{col}'], bins = 5 )
           plt.ylabel(f"{col}")
           plt.title(f"Distribution of {col}")
           plt.show()
           plt.clf()

def gen_bivariate (bivar_vars, med_data):
    for i,col in enumerate(bivar_vars):
        if col == 'Initial_days':
            pass
        else:
            # sns.displot(x='Initial_days', col=f'{col}', col_wrap=2, bins=4, data=med_data)
            sns.scatterplot(x='Initial_days', y=f'{col}', data = med_data)
            plt.show()
            plt.clf()

def initial_model (mdl_input, df_orig):
    gross_mdl = ols(mdl_input,data=df_orig).fit()
    print(gross_mdl.summary())
    print(f'\nResidual Error: {np.sqrt(gross_mdl.mse_resid)}')
    reduce_by_p_value = pd.DataFrame(gross_mdl.pvalues)
    reduce_by_p_value.columns=['p_value']
    reduce_by_p_value = reduce_by_p_value.query('p_value <= 0.05')
    print(f'\nFinding p_values <= 0.05\nReduce Model to following:\n{reduce_by_p_value.iloc[:20]}')
    red_values = pd.DataFrame(reduce_by_p_value.index.values, columns=['feature'])
    red_values = red_values.query('feature != "Intercept"')
    return red_values


def reduced_model (df_orig, mdl_input):
    red_mdl = ols(mdl_input, data=df_orig).fit()
    print(red_mdl.summary())
    print(f'\nResidual Error: {np.sqrt(red_mdl.mse_resid)}')
    model_normed_resd = red_mdl.get_influence().resid_studentized_internal
    model_norm_resd_abs_sqrt = np.sqrt(np.abs(model_normed_resd))
    sns.regplot(x=red_mdl.fittedvalues, y=model_norm_resd_abs_sqrt, ci=None, lowess=True)
    plt.show()

if __name__ == '__main__':
    #Read Medical Data into med_data data frame
    med_data = pd.read_csv('/Users/bixbypeterson/desktop/WGU/D208/medical_clean.csv')
    
    #Dependent Variable
    resp_var = 'Initial_days'
            
    #Generate Initial Model
    #Exclude object columns not needed for analysis
    med_data.drop(labels=['Customer_id','Interaction','UID','TimeZone','Population','Children','Income', 'City','State', 'County','CaseOrder','Item1','Item2','Item3','Item4','Item5','Item6','Item7','Item8','Zip','Lat','Lng','Job','Marital','Services', 'Initial_admin', 'Area' , 'HighBlood', 'Stroke', 'Complication_risk', 'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma'], axis = 1, inplace=True)   
    
        
    #Initial Model Prep
    med_data = pd.get_dummies(med_data) # Create Dummy Variables
    X = med_data[list(med_data.columns[:-2])] # Get Column list for VIF
    vif = pd.DataFrame() # Create a datafram for VIF results
    vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # Add VIF Factor to DataFrame
    vif["features"] = X.columns # Add VIF Features to DataFrame
    vif = vif.query('VIF_Factor < 5 or VIF_Factor == inf') # Find features that have VIF Factor less than 5 or INF (Infinity)
    print(vif)
    
    init_df = pd.DataFrame(data=med_data,columns=vif['features'].values) 
    init_df['Initial_days'] = med_data['Initial_days']
    init_df.to_csv('/Users/bixbypeterson/desktop/WGU/D208/medical_data_mr_cleaned.csv',sep=',',index=True)
    df_stats = pd.DataFrame(init_df.describe())
    df_stats.to_csv('/Users/bixbypeterson/desktop/WGU/D208/medical_data_mr_stats.csv',sep=',',index=True)
    
    #Create Univariate Visuals
    # gen_univariate(vif['features'].values, df_stats)
    
    #Create Bivariate Visuals
    # gen_bivariate(vif['features'].values, df_stats)
    
                                                     
    #Call Initial Model function
    # mdl_input = resp_var + ' ~ ' + ' + '.join(list(vif['features'].values))
    # reduced_features = initial_model(mdl_input, med_data)
    
    #Generate Reduced Model
    #Call Reduced Model function
    # mdl_input = resp_var + ' ~ ' + ' + '.join(list(reduced_features['feature'].values))
    # reduced_model(med_data, mdl_input)