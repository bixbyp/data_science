#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by:
    Bixby Peterson
    Student Id: 5819654
    Program: Master Data Analytics (May 2021)

Created on:
    Sept 21, 2021

"""

import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import seaborn as sns
from matplotlib import pyplot as plt

#Read Medical Data into med_data data frame
med_data = pd.read_csv('/Users/bixbypeterson/desktop/WGU/D207/medical_clean.csv')


def chi2_calc(med_data):
    md_cross = pd.crosstab(med_data.Gender, med_data.Complication_risk)
    stat, p, dof, expected = chi2_contingency(md_cross)
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    alpha = 1 - prob


#Degree of Freedom
    print('Degrees Of Freedom')
    print('\t Degrees of Freedom = %.0f \n\n' % (dof))

#Interpret Test
    print('Test Statistics')
    print('\t Probability = %.3f \n\t Critical = %.3f \n\t Stats = %.3f \n\n' % (prob, critical, stat))

    print('Statistics Results')
    if abs(stat) >= critical:
        print('\t Dependent: Reject NULL Hypothesis \n \t Stat of: {0} is greater than Critical of: {1}\n'.format(stat, p))
    else:
        print('\t Independent: Do No Reject NULL Hypothesis \n\n')
    
#Interpret p-Value
    print('p-Value Evaluation')
    print('\t p-Value = %.2f \n\t Alpha = %.2f \n\n' % (p, alpha))
    print('p-Value Results')
    if p <= alpha:
        print('\t Dependent: Reject NULL Hypothesis \n\n')
    else:
        print('\t Independent: Do No Reject NULL Hypothesis \n\n')


def univar_2num(med_data):
    #age and total_charges
    df_age = med_data['Age']
    print('Univariate Stats for 2 Continuous \n')
    print('Age: \n{}\n'.format(df_age.describe()))
    print('\n')
    df_tot = med_data['Additional_charges']
    print('Additional Charges: \n{}\n'.format(df_tot.describe()))

    plt.boxplot(med_data['Age'] )
    plt.ylabel("Age")
    plt.title("Distribution of Age")
    plt.show()

    
    plt.clf()
    plt.boxplot(med_data['Additional_charges'])
    plt.ylabel("Additional Charges")
    plt.title("Distribution of Charges")
    plt.show()

    
def univar_2cat(med_data):
    #gender and complication_risk stats
    df_gender = med_data['Gender']
    print('Univariate Stats for 2 Categorical \n')
    print('Gender: \n{}\n'.format(df_gender.describe()))
    print('\n')
    df_risk = med_data['Complication_risk']
    print('Complication Risk: \n{}\n'.format(df_risk.describe()))
    
    # Histogram plots
    plt.clf()
    plt.hist(med_data['Gender'], bins = 5 )
    plt.xlabel("Gender")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gender")
    plt.show()

    plt.clf()
    plt.hist(med_data['Complication_risk'], bins = 5)
    plt.xlabel("Complication Risk")
    plt.ylabel("Frequency")
    plt.title("Distribution of Risk")
    plt.show()

def bivar_2num(med_data):
    plt.clf()
    sns.lmplot(x='Age', y='Additional_charges', hue='Area', fit_reg= True, data = med_data)
    df_bivar = med_data[['Age','Additional_charges']]
    print('{}\n'.format(df_bivar.corr()))

def bivar_2cat(med_data):
    md_cross = pd.crosstab(med_data.Gender, med_data.Complication_risk)
    print('{}\n\n'.format(md_cross))
    plt.clf()    
    sns.heatmap(md_cross.T, annot=True, fmt='.0f', cmap="YlGnBu", cbar=False)

if __name__ == '__main__':
    univar_2num(med_data)
    univar_2cat(med_data)
    bivar_2num(med_data)
    bivar_2cat(med_data)
    chi2_calc(med_data)