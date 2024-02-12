#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:36:08 2022

@author: bixbypeterson
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

def main():
    print('Import Dataset')
    dataset = import_dataset()
    print('Converting Dataset to List of Lists')
    dataset_list = create_lists(dataset)
    print('One Hot Encoding of Data')
    encoded_data = encode_dataset(dataset_list)
    print('Create Association Rules')
    rules_data = market_basket(encoded_data)
    print('Exporting Data')
    export_data(rules_data,encoded_data)

def import_dataset():
    locale = '/Users/bixbypeterson/desktop/WGU/D212/medical_market_basket.csv'
    dataset = pd.read_csv(locale, header = 1)
    dataset.dropna(axis=0, how='all', inplace=True) # Remove total blank rows from dataset
    
    return dataset


def create_lists(dataset):
    lol = []
    for i in range (1,len(dataset)):
        lol.append([str(dataset.values[i,j]) for j in range(0,20)])

    return lol

def encode_dataset(dataset_lists):
    encoder = TransactionEncoder().fit(dataset_lists)
    encoded = encoder.transform(dataset_lists)
    enc_data = pd.DataFrame(encoded, columns = encoder.columns_)
    enc_data.drop('nan',1,inplace=True)
    print(f'\tNumber of Transactions: {enc_data.shape[0]}')
    return enc_data

def market_basket(enc_dataset):
    freq_items = apriori(enc_dataset, min_support = 0.02, use_colnames = True )
    rules = association_rules(freq_items, metric = 'lift', min_threshold = 1)
    rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
    rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
    print(f'\tNumber of Association Rules: {rules.shape[0]}')
    return rules

def export_data(rules_data,encoded_data):
    encoded_data.to_csv('/Users/bixbypeterson/desktop/WGU/D212/Task3/medical_marketbasket_prepped.csv')
    rules_data[['antecedents','consequents','support','lift','confidence','leverage']].to_csv('/Users/bixbypeterson/desktop/WGU/D212/Task3/medical_rules.csv')       
    
if __name__ == '__main__':
    warnings.simplefilter(action='ignore',category=DeprecationWarning)
    main()