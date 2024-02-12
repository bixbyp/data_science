#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:02:20 2021

@author: bixbypeterson
"""

import pandas as pd
from scipy import stats

#Read CSV File into bank_train
med_data = pd.read_csv('/Users/bixbypeterson/desktop/WGU/D206/medical_raw_data.csv')

print('Are any columns completely null')
if med_data.isnull().all().any():
    print('Yes \n')
else:
    print('No \n')


print('Are any rows completely null')
if med_data.isnull().all(axis=1).any():
    print('Yes \n')
else:
    print('No \n')


med_data['age_z'] = stats.zscore(med_data['Age'],nan_policy='omit')
med_data_outliers = med_data.query('age_z > 3 | age_z < -3')
if med_data_outliers.empty:
    print('No Age Outliers \n')
else:
    med_data_sort = med_data.sort_values(['child_z'], ascending = False)
    print('Age Outliers')
    print(med_data_sort[['Customer_id','Age']].head(n=5))


med_data['child_z'] = stats.zscore(med_data['Children'],nan_policy='omit')
med_data_outliers = med_data.query('child_z > 3 | child_z < -3')
if med_data_outliers.empty:
    print('No Children Outliers \n')
else:
    med_data_sort = med_data.sort_values(['child_z'], ascending = False)
    print('Children Outliers')
    print(med_data_sort[['Customer_id','Children']].head(n=5))
    print('\n')
    kids = med_data['Children']    
    kids.plot.box()
    
    
med_data['income_z'] = stats.zscore(med_data['Income'],nan_policy='omit')
med_data_outliers = med_data.query('income_z > 3 | income_z < -3')
if med_data_outliers.empty:
    print('No Income Outliers \n')
else:
    med_data_sort = med_data.sort_values(['income_z'], ascending = False)
    print('Income Outliers')
    print(med_data_sort[['Customer_id','Income']].head(n=5))
    print('\n')


med_data['pop_z'] = stats.zscore(med_data['Population'],nan_policy='omit')
med_data_outliers = med_data.query('pop_z > 3 | pop_z < -3')
if med_data_outliers.empty:
    print('No Population Outliers \n')
else:
    med_data_sort = med_data.sort_values(['pop_z'], ascending = False)
    print('Population Outliers')
    print(med_data_sort[['Zip','Population']].head(n=5))
    print('\n')
    

med_data['vit_z'] = stats.zscore(med_data['VitD_levels'],nan_policy='omit')
med_data_outliers = med_data.query('vit_z > 3 | vit_z < -3')
if med_data_outliers.empty:
    print('No Vitamin D Outliers \n')
else:
    med_data_sort = med_data.sort_values(['vit_z'], ascending = False)
    print('Viatmin D Outliers')
    print(med_data_sort[['Gender','VitD_levels']].head(n=5))
    print('\n')
    med_data.plot.scatter(y = 'VitD_levels', x = 'Gender' )

med_data['vit_z'] = stats.zscore(med_data['VitD_supp'],nan_policy='omit')
med_data_outliers = med_data.query('vit_z > 3 | vit_z < -3')
if med_data_outliers.empty:
    print('No Vitamin D Supplaments Outliers \n')
else:
    med_data_sort = med_data.sort_values(['vit_z'], ascending = False)
    print('Viatmin D Supplaments Outliers')
    print(med_data_sort[['Customer_id','VitD_levels']].head(n=5))
    print('\n')


med_data['doc_z'] = stats.zscore(med_data['Doc_visits'],nan_policy='omit')
med_data_outliers = med_data.query('doc_z > 3 | doc_z < -3')
if med_data_outliers.empty:
    print('No Dr Visit Outliers \n')
else:
    med_data_sort = med_data.sort_values(['doc_z'], ascending = False)
    print('Dr Visit Outliers')
    print(med_data_sort[['Customer_id','Doc_visits']].head(n=5))
    print('\n')

med_data['full_z'] = stats.zscore(med_data['Full_meals_eaten'],nan_policy='omit')
med_data_outliers = med_data.query('full_z > 3 | full_z < -3')
if med_data_outliers.empty:
    print('No Full Meals Eaten Outliers \n')
else:
    med_data_sort = med_data.sort_values(['doc_z'], ascending = False)
    print('Full Meal Eaten Outliers')
    print(med_data_sort[['Customer_id','Full_meals_eaten']].head(n=5))
    print('\n')
    
med_data['full_z'] = stats.zscore(med_data['Initial_days'],nan_policy='omit')
med_data_outliers = med_data.query('full_z > 3 | full_z < -3')
if med_data_outliers.empty:
    print('No Initial Day Outliers \n')
else:
    med_data_sort = med_data.sort_values(['doc_z'], ascending = False)
    print('Initial Day Outliers')
    print(med_data_sort[['Customer_id','Initial_days']].head(n=5))
    print('\n')

med_data['full_z'] = stats.zscore(med_data['TotalCharge'],nan_policy='omit')
med_data_outliers = med_data.query('full_z > 3 | full_z < -3')
if med_data_outliers.empty:
    print('No Total Charge Outliers \n')
else:
    med_data_sort = med_data.sort_values(['doc_z'], ascending = False)
    print('Total Charge Outliers')
    print(med_data_sort[['Customer_id','TotalCharge','doc_z']].head(n=5))
    print('\n')

med_data['full_z'] = stats.zscore(med_data['Additional_charges'],nan_policy='omit')
med_data_outliers = med_data.query('full_z > 3 | full_z < -3')
if med_data_outliers.empty:
    print('No Additional Charges Outliers \n')
else:
    med_data_sort = med_data.sort_values(['doc_z'], ascending = False)
    print('Additional Charges Outliers')
    print(med_data_sort[['Customer_id','Additional_charges','doc_z']].head(n=5))
    print('\n')    
#sns.boxplot(med_data['Age'])
# age , education, marital, gender, stroke, complication_risk



