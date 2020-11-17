#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:06:30 2020

@author: talespadilha
"""

import pandas as pd


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