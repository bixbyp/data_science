#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 14:47:21 2022

@author: bixbypeterson
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from kneed import KneeLocator #https://pypi.org/project/kneed/

def main():
    #Load Initial Data
    initDataFrame=initial_DataSet()
    
    #Get Component Names (e.g. PCn)
    namedComponents, numberComponents=namePrincipalComponents(initDataFrame, None, None)
    
    #Preprocessing Step: Scale Data
    scaledData=scaleDataSet(initDataFrame)   
    
    #Run PCA
    explainedVariance, modelPCA = PCA_Analysis(initDataFrame, scaledData, numberComponents, namedComponents)
    
    #ScreePlot and locate Elbow
    kneeInfo = screePlot(explainedVariance, numberComponents)
    
    resultsPCA(kneeInfo, modelPCA)
    
    #Get Component Names (e.g. PCn)
    namedComponents, numberComponents=namePrincipalComponents(initDataFrame, 'N', kneeInfo.elbow)
        
    #Run PCA
    explainedVariance, modelPCA = PCA_Analysis(initDataFrame, scaledData, kneeInfo.elbow, namedComponents)
    
def initial_DataSet():
    locale = '/Users/bixbypeterson/desktop/WGU/D212/medical_clean.csv'
    dataset = pd.read_csv(locale)
    initialDataFrame=dataset.select_dtypes(include=['int','float64'])
    initialDataFrame=initialDataFrame[['Age','Income','Initial_days','TotalCharge','Additional_charges','Full_meals_eaten','VitD_levels','Doc_visits']]
    return initialDataFrame


def namePrincipalComponents(initDataFrame, firstRun, loopTerminator):
    princCompNames = []
    
    if firstRun is None:
        for i, col in enumerate(initDataFrame.columns):
            princCompNames.append('PC'+str(i+1))
        
        numComps = i+1
    else:
        for i in np.arange(0,loopTerminator):
            princCompNames.append('PC'+str(i+1))
        numComps = i

    return princCompNames, numComps    

def scaleDataSet(initDataFrame):
    scaler = StandardScaler()
    scaledDataFrame = scaler.fit_transform(initDataFrame)
    dfScaledDataFrame = pd.DataFrame(scaledDataFrame)
    dfScaledDataFrame.to_csv('/Users/bixbypeterson/Desktop/WGU/D212/Task2/pca_prepped_scaled_data.csv')
    return scaledDataFrame



def PCA_Analysis(initDataFrame, scaledDataFrame, numberComponents, namedComponents):
    pca_med = PCA(n_components=numberComponents, random_state=2020)
    pca_med = pca_med.fit(scaledDataFrame)

    
    CoVar = pd.DataFrame.cov(initDataFrame)
    print(f'Covariance Matrix for {numberComponents}:\n{CoVar}')
    
    CoVar.to_csv(f'/Users/bixbypeterson/Desktop/WGU/D212/Task2/covariance_matrix_{numberComponents}comp.csv')
    
    print (f'Variance of ALL {numberComponents} Principle Components = {round(sum(pca_med.explained_variance_ratio_ * 100),2)}\n')
    
    exp_var_rat = pca_med.explained_variance_ratio_ * 100
    var_by_pc = pd.DataFrame(exp_var_rat.round(2), columns=['Variance'], index = namedComponents)
    
    print(f'Captured Variances for ALL {numberComponents} Principle Components: \n{var_by_pc}\n')
    
    exp_var = pca_med.explained_variance_ratio_

    return exp_var, pca_med

def screePlot(explainedVariance, numberComponents):
    kl = KneeLocator(range(0,numberComponents),np.cumsum(explainedVariance).round(2),curve='concave', direction='increasing')
    
    plt.plot(np.cumsum(explainedVariance).round(2))
    kl.plot_knee()
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variances')
    plt.show()
    
    return kl

def resultsPCA(kneeInfo, pcaResults):
    print(f'Elbow Location is: {kneeInfo.elbow} components\n')
    print(f'Total Variance for first {kneeInfo.elbow} components: {np.cumsum(pcaResults.explained_variance_ratio_ * 100)[kneeInfo.elbow -1].round(2)}\n')

if __name__ == '__main__':
    pd.options.display.max_columns = None
    main()