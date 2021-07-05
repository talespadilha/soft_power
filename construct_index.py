#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:36:04 2021

@author: talespadilha
"""

import pandas as pd
import os


def calc_index(sub_idx):
    """Aggregates final index"""
    stacked_df=sub_idx.stack('country')
    stacked_idx = stacked_df.mean(axis=1, skipna=False)
    #(stacked_df.isna().sum(axis=1)<3).sum()
    df_index = stacked_idx.unstack('country')
    final_df = df_index.dropna(how='all').dropna(axis=1, how='all')
    
    return final_df
    
    
if __name__ == '__main__':
    # Setting path and reading data
    os.chdir('/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data')
    data = pd.read_csv('sub_indices.csv', header = [0,1], index_col = [0], parse_dates=True)
    # Calculating final index
    final_index = calc_index(data)
    # Exporting
    final_index.to_csv('index.csv')
