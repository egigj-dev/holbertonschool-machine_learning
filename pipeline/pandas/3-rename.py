#!/usr/bin/env python3
"""Script that manipulates a dataframe column"""
import pandas as pd
from_file = __import__('2-from_file').from_file

def rename(df):
    """Renames a dataframe column"""
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df[['Datetime', 'Close']]
