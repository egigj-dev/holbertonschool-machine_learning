#!/usr/bin/env python3
"""Script that concatenates two DataFrames"""
import pandas as pd


def concat(df1, df2):
    """
    Concatenates two DataFrames (df1 and df2) with proper indexing and labeling.
    """
    # Import the index function
    index = __import__('10-index').index

    # 1. Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # 2. Select all rows from df2 up to timestamp 1417411920
    df2_selected = df2[df2.index <= 1417411920]

    # 3. Concatenate df2_selected on top of df1
    concatenated = pd.concat([df2_selected, df1], keys=['bitstamp', 'coinbase'])

    return concatenated
