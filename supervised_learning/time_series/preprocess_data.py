#!/usr/bin/env python3
"""Preprocess BTC data for forecasting"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(file):
    """Load CSV and clean"""
    df = pd.read_csv(file)

    # Drop timestamp
    df = df.drop(columns=["Timestamp"])

    return df


def create_sequences(data, window=1440, horizon=60):
    """
    window = past 24h (1440 minutes)
    horizon = next 1h (60 minutes)
    """
    X, y = [], []

    for i in range(len(data) - window - horizon):
        X.append(data[i:i + window])
        y.append(data[i + window + horizon - 1, 3])  # close price index

    return np.array(X), np.array(y)


def preprocess(file):
    df = load_data(file)

    # Normalize
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)

    X, y = create_sequences(data)

    return X, y, scaler


if __name__ == "__main__":
    coinbase_file = "coinbase.csv"
    bitstamp_file = "bitstamp.csv"

    X1, y1, scaler1 = preprocess(coinbase_file)
    X2, y2, scaler2 = preprocess(bitstamp_file)

    # Save processed datasets
    np.savez("coinbase_processed.npz", X=X1, y=y1)
    np.savez("bitstamp_processed.npz", X=X2, y=y2)
