#!/usr/bin/env python3
"""Script that visualizes a dataframe"""
import pandas as pd
import matplotlib.pyplot as plt
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp → Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index on Date
df = df.set_index('Date')

# Fill Close missing values with previous row
df['Close'] = df['Close'].fillna(method='ffill')

# Fill High, Low, Open missing values with same-row Close
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

# Fill Volume_(BTC) and Volume_(Currency) missing values with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Keep only data from 2017 onward
df = df[df.index.year >= 2017]

# Resample daily and aggregate
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
print(df_daily)

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['High'], label='High', linewidth=1.5)
plt.plot(df.index, df['Low'], label='Low', linewidth=1.5)
plt.plot(df.index, df['Open'], label='Open', linewidth=1.5)
plt.plot(df.index, df['Close'], label='Close', linewidth=1.5)
plt.plot(df.index, df['Volume_(BTC)'], label='Volume_(BTC)', linewidth=1.5)
plt.plot(df.index, df['Volume_(Currency)'], label='Volume_(Currency)', linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Bitcoin Daily OHLC and Volume Data')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
