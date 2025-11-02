#!/usr/bin/env python3
""" Script that slices a Pandas Dataframe"""


def slice(df):
    """Slices a dataframe"""
    return df.iloc[::60][['High', 'Low', 'Close', 'Volume_(BTC)']]
