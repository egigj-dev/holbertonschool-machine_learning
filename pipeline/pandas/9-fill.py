#!/usr/bin/env python3
"""Script that cleans and fills missing data in a dataframe"""


def fill(df):
    """Cleans and fills missing data in the dataframe."""

    # 1. Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # 2. Fill missing Close values with previous row's Close
    df['Close'] = df['Close'].fillna(method='ffill')

    # 3. Fill missing High, Low, Open with the Close value in the same row
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    # 4. Set missing Volume_(BTC) and Volume_(Currency) to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
