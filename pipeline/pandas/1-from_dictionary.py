#!/usr/bin/env python3
""" Script that creates a Pandas Dataframe from a dictionary """
import pandas as pd


data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four'],
}
df = pd.DataFrame(data)
print(df)