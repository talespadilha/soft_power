#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:17:55 2021

@author: talespadilha
"""
import numpy as np
import pandas as pd

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
    wb_df = wb_df.fillna(method='ffill')
    wb = {}
    # Inflation - Annual growth of rate of country level CPI inflation
    col = 'Inflation, consumer prices (annual %)'
    wb['infla'] = wb_df.xs(col, axis=1, level=1)
    # Goverment Consumption - 5y rolling standard deviation of annual government consumption to GDP ratio
    col = 'General government final consumption expenditure (% of GDP)'
    wb['gov_spending'] = wb_df.xs(col, axis=1, level=1).rolling(5).std()
    # Current Account - Annual current account balance to GDP ratio
    col = 'Current account balance (% of GDP)'
    wb['bca'] = wb_df.xs(col, axis=1, level=1)
    # Trade Openess - Real exports plus real imports divided by real GDP
    col = 'Trade (% of GDP)'
    wb['trade'] = wb_df.xs(col, axis=1, level=1)
    # Domestic Private Credit - The ratio of domestic credit provided by the banking sector to GDP
    col = 'Domestic credit to private sector (% of GDP)'
    wb['credit'] = wb_df.xs(col, axis=1, level=1)
    # Stock Market Capitalization
    col = 'Market capitalization of listed domestic companies (% of GDP)'
    wb['market_cap'] = wb_df.xs(col, axis=1, level=1)
    # Concatenating - The ratio of stock market capitalization to GDP
    df = pd.concat(wb, axis=1, names=["variable"])
    df.columns.set_names('country', level='Country Code', inplace=True)
    
    return df 


def import_tot(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from ToT file

    Args:
        files_path: str with the path for where the raw files are located.

    Returns:
        df: pd.DataFrame with the final output
    """
    tot = pd.read_excel(files_path+'tot.xlsx', header = [0], index_col = [0, 1]).dropna(how='all', axis=1).T
    tot = tot.droplevel(0, axis=1)
    tot.index = pd.to_datetime(tot.index, format="%Y")
    # 5y rolling standard deviation of annual country level terms of trade index growth
    df = pd.concat({'tot': (tot.pct_change()*100).rolling(5).std()}, axis=1)
    df.columns.names = ['variable', 'country']
    
    return df


def import_exp_con(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from Exp Con file

    Args:
        files_path: str with the path for where the raw files are located.

    Returns:
        df: pd.DataFrame with the final output
    """
    exp = pd.read_csv(files_path+'exp_con.csv', encoding='latin-1', header=[0,1], index_col=[0,1]).T
    conc = exp.xs('Concentration Index', level=1).droplevel(0, axis=1)
    conc.index = pd.to_datetime(conc.index, format="%Y")
    # Concentration indices of merchandise exports and imports by country
    df = pd.concat({'concent': conc}, axis=1)
    df.columns.names = ['variable', 'country']
    
    return df


def import_lpi(files_path: str) -> pd.DataFrame:
    """Imports and transforms data from Exp Con file

    Args:
        files_path: str with the path for where the raw files are located.

    Returns:
        df: pd.DataFrame with the final output
    """
    lpi = pd.read_csv(files_path+'lpi.csv',header=[0], index_col=[1,4]).reindex(['obs_value'], axis=1)
    lpi_unstack = lpi.unstack(level=0).xs('obs_value', axis=1)
    lpi_unstack.index = pd.to_datetime(lpi_unstack.index, format="%Y")
    # 5y rolling standard deviation of annual labour productivity growth for each country
    df = pd.concat({'l_product': (lpi_unstack.pct_change()*100).rolling(5).std()}, axis=1)
     
    return df


def import_control(t0, tT):
    control_path = '/Users/talespadilha/Dropbox/Soft Power and FX Prediction/Data/control variables/'
    df = pd.concat([wb_import(control_path), import_tot(control_path), 
                    import_exp_con(control_path), import_lpi(control_path)], axis=1)
    
    return df.loc[t0:tT]
    
    
