#!/usr/bin/env python3
"""Script that sorts a dataframe in descending order"""


def high(df):
    """Sorts a dataframe in descending order"""
    return df.sort_values('High', ascending=False)
