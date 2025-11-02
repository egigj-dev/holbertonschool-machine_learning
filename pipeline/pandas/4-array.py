#!/usr/bin/env python3
"""Script that creates a Numpy array from a Pandas Dataframe"""
import pandas as pd
import numpy as np

def array(df):
    """Creates a Numpy array from a Pandas Dataframe."""
    df = df[['High', 'Close']]
    arr = df.to_numpy()
    return arr
