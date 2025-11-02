#!/usr/bin/env python3
""" Script that creates a Pandas Dataframe from a Numpy array """
import pandas as pd
import numpy as np


def from_numpy(array):
    """Creates a pandas DataFrame from a NumPy ndarray."""
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    num_cols = array.shape[1] if array.ndim > 1 else 1
    columns = [chr(65 + i) for i in range(num_cols)]
    df = pd.DataFrame(array, columns=columns)
    return df
