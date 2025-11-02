#!/usr/bin/env python3
"""Script that creates a Numpy array from a Pandas Dataframe"""


def array(df):
    """Selects the last 10 rows of High and Close and converts to a numpy array."""
    df_selected = df[['High', 'Close']].tail(10)
    arr = df_selected.to_numpy()
    return arr
