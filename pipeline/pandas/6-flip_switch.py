#!/usr/bin/env python3
""" Script that transforms a Pandas Dataframe"""


def flip_switch(df):
    """Flips a dataframe"""
    df = df.sort_values('Timestamp', ascending=False)
    return df.T
