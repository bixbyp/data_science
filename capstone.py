#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:07:51 2022

@author: bixbypeterson
"""

# Import modules needed for analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn import metrics

red_wines = pd.read_csv('/Users/bixbypeterson/Desktop/WGU/D214/winequality-red.csv', sep=';', header=0 )
white_wines = pd.read_csv('/Users/bixbypeterson/Desktop/WGU/D214/winequality-white.csv', sep=';', header=0 )
wine_dataset = pd.concat([red_wines,white_wines],ignore_index=True)

#EDA Code
# High Level dataframe stats
print(f'Red Wine -> Number of interactions: {red_wines.shape[0]} \nNumber of variables: {red_wines.shape[1]}\n')
print(f'White Wine -> Number of interactions: {white_wines.shape[0]} \nNumber of variables: {white_wines.shape[1]}\n')
print(f'Combined -> Number of interactions: {wine_dataset.shape[0]} \nNumber of variables: {wine_dataset.shape[1]}\n')


#Identify Columns that are Null / NaN
has_nulls = wine_dataset.isnull().sum()
dc_nulls = pd.Series(has_nulls[has_nulls > 0])
if dc_nulls.empty:
    print('No Null Columns\n')
else:
    print('Dropping Nulln') 
    # Remove Nulls
    wine_dataset.dropna()


# Describe dataset
pd.set_option('display.max_columns',None)
print(wine_dataset.describe())
print(wine_dataset.dtypes)
print(wine_dataset.head(20))

# wine_dataset = wine_dataset.loc[wine_dataset['quality'] != 3 ]
# wine_dataset = wine_dataset.loc[wine_dataset['quality'] != 4 ]
# wine_dataset = wine_dataset.loc[wine_dataset['quality'] != 8 ]
# wine_dataset = wine_dataset.loc[wine_dataset['quality'] != 9 ]

fig, axes =plt.subplots(1,2,figsize=(25,10))

# # View Distribution of data by Quality
sns.countplot(data=wine_dataset, x='quality', ax=axes[0])
axes[0].set_title('Data Distribution')

# Generate Correlations on wine_dataset
wine_corr = wine_dataset.corr()

# Generate correlation heatmap
heat_map = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(wine_corr, vmin = -1, vmax = 1, square = True, cmap = heat_map, ax=axes[1])
axes[1].set_title('Correlation HeatMap')


use_cols = []
for _ ,name in enumerate(wine_dataset.columns[:11]):
    if wine_corr['quality'][name] > 0.05:
        corr_val = wine_corr['quality'][name]
        use_cols.append(name)
        print(f"Correlation for {name} with quality: {corr_val}\n")
                
use_cols.append('quality')

# Reduced dataset
reduced_wine_df = wine_dataset[use_cols]

# Create x and y - Prep for train-test splits
wine_x = reduced_wine_df[use_cols[:len(use_cols)-1]]
wine_y = reduced_wine_df[['quality']]

# Visualize Data is Linear
plt.figure()
sns.pairplot(data=reduced_wine_df, x_vars=reduced_wine_df[use_cols[:len(use_cols)-1]], y_vars='quality', kind = 'reg')

# Generate Train and Test - using 80 / 20
wine_x_train, wine_x_test, \
    wine_y_train, wine_y_test = tts(wine_x, wine_y, 
                                    test_size = 0.2, random_state=42)

# Print Samples of Train and Test Sets
print(f'Wine X Train sample:\n{wine_x_train.head(20)}\n')
print(f'Wine Y Train sample:\n{wine_y_train.head(20)}\n')
print(f'Wine X Test sample:\n{wine_x_test.head(20)}\n')
print(f'Wine Y Test sample:\n{wine_y_test.head(20)}\n')

# Instaniate LinearRegression
wine_lr = lr()

# Fit Regression model with Train data
wine_lr.fit(wine_x_train, wine_y_train)
 
# Predict using the Test data
wine_pred = wine_lr.predict(wine_x_test)

# Display Model evaluations
print(f'Coefficients: {wine_lr.coef_}')
print(f'R-squared: {np.round(wine_lr.score(wine_x_train, wine_y_train),2)*100}')
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(wine_y_test, wine_pred)}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(wine_y_test, wine_pred)}')
print(f'Root Mean Square Error (rMSE): {np.sqrt(metrics.mean_squared_error(wine_y_test, wine_pred))}')

# Plot Test vs Prediction
plt.figure()
sns.regplot(x = wine_y_test, y = wine_pred, ci=None, color = 'red')
