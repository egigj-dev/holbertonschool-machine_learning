#!/usr/bin/env python3
""" Script that creates a Pandas Dataframe from a file """
import pandas as pd


def from_file(filename, delimiter):
    """Creates a pandas DataFrame from a file."""
    return pd.read_csv(filename, delimiter=delimiter)
