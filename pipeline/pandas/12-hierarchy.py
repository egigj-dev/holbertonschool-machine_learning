#!/usr/bin/env python3
"""Hierarchy concatenation"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two DataFrames with hierarchy.
    """

    # Index both DataFrames by Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Slice by timestamp range
    start = 1417411980
    end = 1417417980

    df1_slice = df1.loc[start:end]
    df2_slice = df2.loc[start:end]

    # Concatenate with keys
    concat_df = pd.concat(
        [df2_slice, df1_slice],
        keys=['bitstamp', 'coinbase']
    )

    # Make Timestamp the first level
    concat_df.index = concat_df.index.swaplevel(0, 1)

    # Sort by timestamp (first level)
    concat_df = concat_df.sort_index(level=0)
    return concat_df
