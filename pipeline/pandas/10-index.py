#!/usr/bin/env python3
"""Script that returns a column as the index of a dataframe"""


def index(df):
    """Returns a column the index of a dataframe"""
    return df.set_index('Timestamp')
