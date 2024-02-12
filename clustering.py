#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:28:35 2022

@author: bixbypeterson
"""

import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans as vqKMeans, whiten, vq
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline #https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
from sklearn.metrics import silhouette_score # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
import matplotlib.pyplot as plt #https://matplotlib.org/
import seaborn as sns #https://seaborn.pydata.org/
from kneed import KneeLocator #https://pypi.org/project/kneed/
import warnings

def ImportDataSet(dataset_locale):
    dataset = pd.read_csv(dataset_locale)

    return dataset

def RefineVariables(rawData):
    refined_data = rawData[['Additional_charges','TotalCharge',]]
    refined_data_ohe = pd.get_dummies(refined_data)
    
    return refined_data, refined_data_ohe

def showRefinedDataSetInfo(dataset_refined):
    print(f'Medical Data - Refined Variable Statistics: \n{dataset_refined.describe()}')

def elbowMethod(dataset):
    distortions = []
                   
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i)
        pipeline = make_pipeline(StandardScaler(),kmeans)
        pipeline = pipeline.fit(dataset)
        distortions.append(pipeline['kmeans'].inertia_)
        
    kl = KneeLocator(range(1,11),distortions,curve='convex', direction='decreasing')
        
    plt.figure
    kl.plot_knee()
    sns.lineplot(range(1,11), distortions, marker='o',color = 'red')
    plt.ylabel('Distortions')
    plt.xlabel('k')
    plt.title('Elbow Method')
    plt.show()
    
    
    return round(kl.elbow,0)

def KMeansClustering(clusters, dataset, orig_dataset):
    kmeans = KMeans(n_clusters = clusters)
    pipeline = make_pipeline(StandardScaler(),kmeans)
    pipeline = pipeline.fit(dataset)
    scaled_data = pd.DataFrame(pipeline['standardscaler'].transform(dataset))
    predicted_labels = pipeline['kmeans'].labels_
    print(f'Cluster Evaluation (silhouette score): {silhouette_score(scaled_data, predicted_labels).round(2)}\n\tRange: -1 (worst) to 1 (best)\n\tNOTE: Values close to 0 represent overlapping clusters')
    scaled_data.to_csv("/Users/bixbypeterson/desktop/WGU/D212/Task1/clustering_prepped_scaled_data.csv")    

    
    return pipeline['kmeans'].labels_, pipeline['kmeans'].cluster_centers_, scaled_data


def vqKmeansClustering(clusters, dataset):
            dataset = whiten(dataset)
            centroids, mean_dist = vqKMeans(dataset, clusters)
            print("Code-book :\n", centroids, "\n")
        
            clusters, dist = vq(dataset, centroids)
            print("Clusters :\n", clusters, "\n")
            
            print(dataset.view())
            return clusters, centroids, dataset
        
def KMeansVisualization(dataset, labels, centers):

    # print(dataset.head(5))
    sns.scatterplot(x=dataset[0], y=dataset[1], c=labels)
    sns.scatterplot(x=centers[:,0], y=centers[:,1], c=['black'])
    plt.show()

def vqKMeansVisualization(dataset, labels, centers):

    # print(dataset.head(5))
    sns.scatterplot(x=dataset[:,0], y=dataset[:,1], c=labels, labels=labels)
    sns.scatterplot(x=centers[:,0], y=centers[:,1], c=['black'])
    plt.show() 


def KMeansDriver(incoming_file):
    importedData = ImportDataSet(incoming_file)
    ReducedData, ReducedData_OHE = RefineVariables(importedData)
    showRefinedDataSetInfo(ReducedData)
    nclusters = elbowMethod(ReducedData_OHE)
    # nclusters = 7
    print(f'\nNumber of Clusters for Analysis: {nclusters}\n')
    cLabels, cCentroids, scaled_data = KMeansClustering(nclusters,ReducedData_OHE, ReducedData)
    print(f'Clusters:\n{cLabels}\n')
    print(f'Code Book:\n{cCentroids}\n')
    KMeansVisualization(scaled_data,cLabels,cCentroids)

if __name__ == '__main__':
    warnings.simplefilter(action='ignore',category=FutureWarning)
    locale = '/Users/bixbypeterson/desktop/WGU/D212/medical_clean.csv'
    KMeansDriver(locale)

    
