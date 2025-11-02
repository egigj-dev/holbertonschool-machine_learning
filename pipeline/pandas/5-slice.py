#!/usr/bin/env python3
""" Script that slices a Pandas Dataframe"""


def slice(df):
    """Slices a dataframe"""
    return df.iloc[59][['High', 'Low', 'Close', 'Volume (BTC)']]
