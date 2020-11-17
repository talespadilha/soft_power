#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:44:17 2020

@author: talespadilha
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import support_functions as sf


def wb_import(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from World Bank file

    Args:
        files_path: str with the path for where the raw files are located.

    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing data
    wb_df = pd.read_excel(files_path+'WB.xlsx', header = [0], index_col = [0, 1, 2, 3])
    wb_df.columns = [x[:4] for x in wb_df.columns]
    wb_df = wb_df.droplevel('Series Code')
    wb_df = wb_df.droplevel('Country Name')
    wb_df = wb_df.T.replace(['..', 0], np.nan)
    # Transforming variables
    wb_df.index = pd.to_datetime(wb_df.index, format='%Y')
    wb = {}
    pop = wb_df.xs('Population, total', axis=1, level=1).astype(float)
    gdp = wb_df.xs('GDP (current US$)', axis=1, level=1).astype(float)
    # Tourists as share of population
    col = 'International tourism, number of arrivals'
    wb['int_tourists'] = wb_df.xs(col, axis=1, level=1)/pop
    # Gross tertiary education
    col = 'School enrollment, tertiary (% gross)'
    wb['ter_education'] = wb_df.xs(col, axis=1, level=1)
    # Articles as share of population
    col = 'Scientific and technical journal articles'
    wb['publications'] = wb_df.xs(col, axis=1, level=1)/pop
    # Assistance as share of GDP
    col = 'Net official development assistance and official aid received (current US$)'
    wb['aid'] = wb_df.xs(col, axis=1, level=1)/gdp
    # Refugees as share of population
    col = 'Refugee population by country or territory of asylum'
    wb['refugees'] = wb_df.xs(col, axis=1, level=1)/pop
    # Internet users as share of population
    col = 'Individuals using the Internet (% of population)'
    wb['internet'] = wb_df.xs(col, axis=1, level=1)/pop
    # Mobile phones as share of population
    col = 'Mobile cellular subscriptions'
    wb['cellphones'] = wb_df.xs(col, axis=1, level=1)/pop
    # Trademarks as share of population
    col = 'Trademark applications, total'
    wb['trademarks'] = wb_df.xs(col, axis=1, level=1)/pop
    df = pd.concat(wb, axis=1, names=["variable"])
    df.columns.set_names('country', level='Country Code', inplace=True)
    
    return df 


def icrg_import(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from icrg file

    Args:
        files_path: str with the path for where the raw files are located.

    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing data
    icrg_df = pd.read_excel(files_path+'ICRG.xlsx', header = [0], index_col = [0, 1, 2])
    var_set = ['Bureaucracy Quality (L)', 'Democratic Accountability (K)',
           'Government Stability (A)', 'Law & Order (I)', 'Corruption (F)']
    icrg_df = icrg_df.reindex(var_set, level='Variable').T
    icrg_df.columns = icrg_df.columns.droplevel('Country')
    icrg_df.index = pd.to_datetime(icrg_df.index, format='%m/%Y')
    # Transforming variables
    icrg_df = icrg_df.resample('YS').mean()
    icrg = {}
    icrg["rule_of_law"] = icrg_df.xs('Law & Order (I)', axis=1, level='Variable')
    icrg["gov_stability"] = icrg_df.xs('Government Stability (A)', axis=1, level='Variable')
    icrg["dem_account"] = icrg_df.xs('Democratic Accountability (K)', axis=1, level='Variable')
    icrg["bur_effect"] = icrg_df.xs('Bureaucracy Quality (L)', axis=1, level='Variable')
    icrg["corruption"] = icrg_df.xs('Corruption (F)', axis=1, level='Variable')
    df = pd.concat(icrg, axis=1, names=["variable"])
    df.columns.set_names('country', level='Code', inplace=True)

    return df


def whc_import(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from UNESCO World Heritage Centres file

    Args:
        files_path: str with the path for where the raw files are located.

    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing data
    whc_df = pd.read_excel(files_path+'UNESCO_WHC.xls', header = [0])
    whc_df = whc_df.reindex(['date_inscribed', 'udnp_code'], axis=1)  
    # Splitting multicountry centres
    whc_df = sf.split_df(whc_df, split_col='udnp_code', separate=',')
    whc_df['udnp_code'] = whc_df['udnp_code'].str.upper()
    # Transforming variables
    whc_df = sm.add_constant(whc_df)
    whc_df = whc_df.groupby(['date_inscribed', 'udnp_code']).count()
    whc_df = whc_df.unstack(level='udnp_code') 
    whc_df = whc_df.cumsum().fillna(method='ffill').fillna(0)
    whc_df.index = pd.to_datetime(whc_df.index, format='%Y')
    whc_df.index.name = None
    whc_df.columns = whc_df.columns.droplevel(0)
    whc_df.columns = pd.MultiIndex.from_product([['whc'], whc_df.columns]).set_names(['variable', 'country'])
    
    return whc_df


if __name__ == "__main__":
    # Setting work directory
    os.chdir('/Users/talespadilha/Documents/Projects/soft_power')
    # Setting the path for the raw data files
    raw_path = '/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/Raw Data/'
    # Building the dataset
    wb = wb_import(raw_path)
    icrg = icrg_import(raw_path)
    whc = whc_import(raw_path)
    # Merging and exporting
    df = pd.concat([wb, icrg, whc], axis=1)
    df.to_csv('/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/data.csv')
    
