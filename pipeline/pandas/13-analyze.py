#!/usr/bin/env python3
"""Compute descriptive statistics for a DataFrame"""


def analyze(df):
    """Computes descriptive statistics for all columns except Timestamp."""
    df_no_ts = df.drop(columns=['Timestamp'], errors='ignore')
    stats = df_no_ts.describe()
    return stats
