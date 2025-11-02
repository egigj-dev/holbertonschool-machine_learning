#!/usr/bin/env python3
"""Script that removes null values from a column"""


def prune(df):
    """Removes null values from a column"""
    return df.dropna(subset=['Close'])
