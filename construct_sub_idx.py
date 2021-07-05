#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:58:33 2021

@author: talespadilha
"""

import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA


def pca_analysis(df: pd.DataFrame, n_comp: int):
    """Returns variance ratio and weights for PCA given number of components"""
    pca = PCA(n_components=n_comp)
    pca.fit(df)
    var_ratio = pca.explained_variance_ratio_
    w = (pca.components_)**2
    
    return var_ratio, w


def calculate_weights(data: pd.DataFrame):  
    """Calculates weights based on the PCA framework"""
    # Looping over sub-indices
    sub_idxs = data.columns.get_level_values('subindex').unique()
    final_w = {}
    # Setting number of PCs found in analysis
    PC_n = pd.Series([3,2,3,2,3,2], index = sub_idxs) 
    for idx in sub_idxs:
        # Selecting data
        idx_data = data.xs(idx, axis=1, level='subindex')          
        # Stacking data
        pooled_data = idx_data.stack('country')
        # Droping nas
        all_nonna = pooled_data.dropna(how='any')
        # Running PCA
        vr, w = pca_analysis(all_nonna, PC_n[idx])
        # Getting 
        weights = pd.DataFrame(w, columns=all_nonna.columns)
        # Dropping weights less than 0.1 - not in this version
        weights[weights<0.10] = 0
        # Using PCs to ws to build final weights
        sm =np.zeros(weights.shape[1])
        for i in range(len(vr)):
            sm = sm + vr[i]*weights.loc[i]
        final_w[idx] = sm / sm.sum()
        
    return final_w
    

def calculate_sub(data: pd.DataFrame, weights: dict):
    """Calculates sub-indices given data and weights"""
    sub_idxs = data.columns.get_level_values('subindex').unique()
    final_idxs = {}
    for idx in sub_idxs:
        # Selecting data
        int_data = data.xs(idx, axis=1, level='subindex')
        w_idx = weights[idx]
        idx_data = int_data.reindex(w_idx.index, axis=1, level='variable')
        # Multiplying by weight
        prod_data = idx_data.multiply(w_idx, level='variable')
        #TODO: think if this is the way we want to treat missing values for individial variables
        series = prod_data.fillna(np.inf).groupby(axis=1, level='country').sum().replace(np.inf, np.nan)
        final_idxs[idx] = series
    df = pd.concat(final_idxs, axis=1, names=['subindex'])
    final_df = df.dropna(how='all').dropna(axis=1, how='all')
    
    return final_df
   

if __name__ == '__main__':
    # Setting path and reading data
    os.chdir('/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data')
    data = pd.read_csv('z_scores.csv', header = [0,1,2], index_col = [0], parse_dates=True)
    # Calculating weights
    weights = calculate_weights(data)
    # Calculating sub-indices
    sub_indices = calculate_sub(data, weights)
    # Exporting
    sub_indices.to_csv('sub_indices.csv')
