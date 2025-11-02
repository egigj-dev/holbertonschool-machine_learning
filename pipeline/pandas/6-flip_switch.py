#!/usr/bin/env python3
""" Script that transforms a Pandas Dataframe"""
import pandas as pd


def flip_switch(df):
    """Flips a dataframe"""
    df = df.sort_values(df['Timestamp'], ascending=False)
    return df.T
