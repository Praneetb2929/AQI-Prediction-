# src/preprocess.py

import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna()
    return df

def get_features(df):
    X = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3']]
    y = df['AQI']
    return X, y