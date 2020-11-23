#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:06:30 2020

@author: talespadilha
"""

import pandas as pd
import numpy as np


def split_df(df: pd.DataFrame, split_col: str, separate: str) -> pd.DataFrame:
    """Splits values in the same cell into different rows according to the
    separate argument given"""
    split_df = (df.set_index(df.columns.drop(split_col,1).tolist())
                .udnp_code.str.split(separate, expand=True)
                .stack()
                .reset_index()
                .rename(columns={0:split_col})
                .loc[:, df.columns]
                )
    
    return split_df


def wb_series(series: str) -> pd.DataFrame:
    """ Imports an specific series from the WB file"""
    # Setting the path
    path = '/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/Raw Data/'
    # Importing data
    wb_df = pd.read_excel(path+'WB.xlsx', header = [0], index_col = [0, 1, 2, 3])
    wb_df.columns = [x[:4] for x in wb_df.columns]
    wb_df = wb_df.droplevel('Series Code')
    wb_df = wb_df.droplevel('Country Name')
    wb_df = wb_df.T.replace(['..', 0], np.nan)
    # Transforming variables
    wb_df.index = pd.to_datetime(wb_df.index, format='%Y')
    df = wb_df.xs(series, axis=1, level=1).astype(float)
    df.columns.rename(None, inplace=True)
    
    return df 
    