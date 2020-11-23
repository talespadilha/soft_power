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

os.chdir('/Users/talespadilha/Documents/Projects/soft_power')

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
    pop = wb_df.xs('Population, total', axis=1, level=1).astype(float).fillna(method='ffill')
    gdp = wb_df.xs('GDP (current US$)', axis=1, level=1).astype(float).fillna(method='ffill')
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
    # Migrants as share of population
    col = 'International migrant stock (% of population)'
    wb['migrants'] = wb_df.xs(col, axis=1, level=1)
    # Internet users as share of population
    col = 'Individuals using the Internet (% of population)'
    wb['internet'] = wb_df.xs(col, axis=1, level=1)
    # Mobile phones as share of population
    col = 'Mobile cellular subscriptions'
    wb['cellphones'] = wb_df.xs(col, axis=1, level=1)/pop
    # Trademarks as share of population
    col = 'Trademark applications, total'
    wb['trademarks'] = wb_df.xs(col, axis=1, level=1)/pop
    # Patents as share of population
    col = 'Patent applications, residents'
    wb['patents'] = wb_df.xs(col, axis=1, level=1)/pop
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


def cult_goods_export(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from UNCTAD export of cultural goods file
    
    Args:
        files_path: str with the path for where the raw files are located.
        
    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing 
    cult_df = pd.read_excel(files_path+'cultural_goods.xlsx', header = [0], index_col = [0,1])
    cult_df = cult_df.iloc[1:,:].T.replace('..', np.nan)
    # Transforming
    cult_df.columns = cult_df.columns.droplevel(0)
    cult_df.index = pd.to_datetime(cult_df.index, format='%Y')
    # Getting GDP    
    gdp =  sf.wb_series('GDP (current US$)').fillna(method='ffill')
    # Final df
    df = cult_df/gdp
    df.columns = pd.MultiIndex.from_product([['cult_exp'], df.columns]).set_names(['variable', 'country'])
    
    return df


def olymp_import(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from olympic medals file
    
    Args:
        files_path: str with the path for where the raw files are located.
        
    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing
    medals = pd.read_excel(files_path+'olympics.xlsx', header = [0], index_col = [0,1,2])
    medals = medals.unstack(level='Year').T
    medals.index = medals.index.droplevel(0)
    medals.index = pd.to_datetime(medals.index, format='%Y')
    medals.columns = medals.columns.droplevel('Country')
    medals = medals.drop(columns='Olympic Team')
    # Transforming
    pop =  sf.wb_series('Population, total').fillna(method='ffill')
    df = (medals/pop).fillna(method='ffill')
    df.columns = pd.MultiIndex.from_product([['medals'], df.columns]).set_names(['variable', 'country'])
    
    return df 


def lowy_import(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from lowy embassies file
    
    Args:
        files_path: str with the path for where the raw files are located.
        
    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing
    emb = pd.read_csv(files_path+'lowy.csv', header = [0], index_col = [0,1])
    emb = emb.T
    emb.columns = emb.columns.droplevel('Country')
    emb.index = pd.to_datetime(emb.index, format='%Y')
    emb.columns = pd.MultiIndex.from_product([['emb'], emb.columns]).set_names(['variable', 'country'])

    return emb


def ofi_import(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from outward foreign investment file
    
    Args:
        files_path: str with the path for where the raw files are located.
        
    Returns:
        df: pd.DataFrame with the final output
    """
    # Importing 
    ofi = pd.read_excel(files_path+'ofi.xlsx', header = [0], index_col = [0,1])
    ofi = ofi.T.replace(['..', ['_']], np.nan)
    ofi = ofi.drop(columns=('Country', 'Code'))
    # Transforming
    ofi.columns = ofi.columns.droplevel(0)
    ofi.index = pd.to_datetime(ofi.index, format='%Y')    
    # Getting GDP
    gdp =  sf.wb_series('GDP (current US$)').fillna(method='ffill')
    # Final df
    df = (ofi*10**6)/gdp    
    df.columns = pd.MultiIndex.from_product([['ofi'], df.columns]).set_names(['variable', 'country'])

    return df


def min_max_norm(df_entry: pd.DataFrame) -> pd.DataFrame:
    """Normalises df according to min-max method"""
    df = df_entry.copy()
    subidx = list(set(df.columns.get_level_values('subindex')))
    for idx in subidx:
        si_df = df.xs(idx, axis=1, level='subindex')
        var_set = list(set(si_df.columns.get_level_values('variable')))
        for var in var_set:
            sample = si_df.xs(var, axis=1, level='variable')
            maxi = sample.max(axis=1)
            mini = sample.min(axis=1)
            norm = sample.sub(mini, axis='index').div(maxi-mini, axis='index')
            df[(idx, var)] = norm.copy()
            
    return df 


def z_norm(df_entry: pd.DataFrame) -> pd.DataFrame:
    """Normalises df according to cross sectional z score method"""
    df = df_entry.copy()
    subidx = list(set(df.columns.get_level_values('subindex')))
    for idx in subidx:
        si_df = df.xs(idx, axis=1, level='subindex')
        var_set = list(set(si_df.columns.get_level_values('variable')))
        for var in var_set:
            sample = si_df.xs(var, axis=1, level='variable')
            mean = sample.median(axis=1)
            std = sample.std(axis=1)
            norm = sample.sub(mean, axis='index').div(std, axis='index')
            df[(idx, var)] = norm.copy()
            
    return df 


if __name__ == "__main__":
    # Setting the path for the raw data files
    raw_path = '/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/Raw Data/'
    # Building the dataset
    wb = wb_import(raw_path)
    icrg = icrg_import(raw_path)
    whc = whc_import(raw_path)
    cult_exp = cult_goods_export(raw_path)
    medals = olymp_import(raw_path)
    emb = lowy_import(raw_path)
    ofi = ofi_import(raw_path)
    # Merging
    df ={}
    df['institutions'] = icrg[['rule_of_law', 'gov_stability', 'dem_account', 'bur_effect']]
    df['culture'] = pd.concat([wb[['int_tourists']], whc, cult_exp, medals], axis=1)
    df['comercial'] = pd.concat([wb[['patents', 'trademarks']], icrg[['corruption']], ofi], axis=1)
    df['digital'] = wb[['internet', 'cellphones']]
    df['global_reach'] = pd.concat([wb[['aid', 'migrants', 'refugees']], emb], axis=1)
    df['education'] = wb[['ter_education', 'publications']]
    df = pd.concat(df, axis=1, names=['subindex'])
    # Normalising the data
    z_scores = z_norm(df)
    maxmin = min_max_norm(df)
    # Exporting
    df.to_csv('/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/data.csv')
    z_scores.to_csv('/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/z_scores.csv')
    maxmin.to_csv('/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/maxmin.csv')

