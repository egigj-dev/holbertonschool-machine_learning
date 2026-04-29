#!/usr/bin/env python3
"""Forecast BTC using RNN"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


def load_data(file):
    data = np.load(file)
    return data["X"], data["y"]


def create_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    # Load data
    X, y = load_data("coinbase_processed.npz")

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Dataset pipeline
    train_ds = create_dataset(X_train, y_train)
    val_ds = create_dataset(X_val, y_val)

    # Build model
    model = build_model((X.shape[1], X.shape[2]))

    # Train
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Save model
    model.save("btc_forecast_model.h5")
