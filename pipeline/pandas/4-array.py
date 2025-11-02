#!/usr/bin/env python3
"""Script that creates a Numpy array from a Pandas Dataframe"""


def array(df):
    """Creates a Numpy array from a Pandas Dataframe."""
    df = df[['High', 'Close']]
    df = df.dropna()
    arr = df.to_numpy()
    return arr
