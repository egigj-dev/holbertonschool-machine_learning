#!/usr/bin/env python3
"""Script that manipulates a dataframe column"""
import pandas as pd


def rename(df):
    """Renames a dataframe column"""
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
