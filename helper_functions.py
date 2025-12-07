import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models

def load_data():
    """
    Load dataset
    """
    prices = pd.read_csv("data/prices_panel.csv", parse_dates = ["Date"])
    funds = pd.read_csv("data/fundamentals_income.csv", parse_dates = ["Report Date", "Publish Date", "Restated Date"])

    # change column names so it easier to type / access them
    prices.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in prices.columns]
    funds.columns = [c.strip().lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace(',', '') for c in funds.columns]

    # drop duplicates
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates()
    funds = funds.sort_values(["ticker", "report_date"]).drop_duplicates()

    # sort double indeces
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    funds = funds.sort_values(["report_date", "ticker"]).reset_index(drop=True)

    # merge price and fudamental data
    df = pd.merge_asof(
        prices,
        funds,
        by="ticker",
        left_on="date",
        right_on="report_date",
        direction="backward",
        )

    return df


def create_raw_features(df):
    """
    Build raw features
    """
    df["ret"] = df.groupby("ticker")['adj_close'].pct_change()
    return df

def create_fundamental_features(df):
    """
    Build fundamental features
    EPS, profit margin, revenue growth, and other ratios derived from reported items.
    """

    df['eps'] = df["net_income"] / df['shares_diluted']

    df['profit_margin'] = df['net_income'] / df['revenue']

    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change()

    return df

def create_engineered_features(df):
    """
    Build engineered features
    Momentum ratios, EMA crossovers, skewness, kurtosis, etc.
    """

    g = df.groupby("ticker")

    df['mean_20'] = g['ret'].transform(lambda x: x.rolling(20).mean())
    df['mean_60'] = g['ret'].transform(lambda x: x.rolling(60).mean())

    df['vol_20'] = g['ret'].transform(lambda x: x.rolling(20).std())
    df['vol_60'] = g['ret'].transform(lambda x: x.rolling(60).std())


    # NOTE: potentially switch to momentum ratio
    df['mom_20'] = g['adj_close'].transform(lambda x: x.pct_change(20))
    df['mom_60'] = g['adj_close'].transform(lambda x: x.pct_change(60))

    df['ema_12'] = g['adj_close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df['ema_26'] = g['adj_close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())

    df["ema_cross"] = df['ema_12'] - df['ema_26']
    # remove lookahead bias
    df["ema_cross"] = g['ema_cross'].shift(1)

    df['skewness_20'] = g['ret'].rolling(20).skew().reset_index(level=0, drop = True)
    df['kurtosis_20'] = g['ret'].rolling(20).kurt().reset_index(level=0, drop = True)

    df['skewness_60'] = g['ret'].rolling(60).skew().reset_index(level=0, drop = True)
    df['kurtosis_60'] = g['ret'].rolling(60).kurt().reset_index(level=0, drop = True)

    return df


def create_binary_labels(df, h):
    # NOTE: improve this function
    # NOTE: in here we are deleting observation which might be a problem

    #labels = [1, 20, 60]

    g = df.groupby('ticker')

    df[f'cumret_{h}'] = g['ret'].transform(
        lambda x: (1 + x.shift(-1))
        .rolling(h, h)
        .apply(lambda r: np.prod(r) - 1).shift(-h + 1))

    # df = df.dropna(subset=[f'cumret_{h}'])

    # df[f'y_{h}'] = (df[f'cumret_{h}'] > 0).astype(int)
    df[f'y_{h}'] = np.where(
        df[f'cumret_{h}'].notna(),
        (df[f'cumret_{h}'] > 0).astype(int),
        np.nan
    )

    #for h in labels:
        #df[f'cumret_{h}'] = g['ret'].transform(lambda x: (1 + x).rolling(h).apply(lambda r: np.prod(r) - 1).shift(-h + 1))

    #df = df.dropna(subset=[f'cumret_{h}' for h in labels])

    #for h in labels:
       # df[f'y_{h}'] = (df[f'cumret_{h}'] > 0).astype(int)

    return df

def time_split(df, train_frac=0.70, val_frac=0.15, date_col='date'):

    # Ensure sorted dates
    dates = df[date_col].sort_values().unique()
    N = len(dates)

    train_dt = dates[int(train_frac * N)]
    val_dt   = dates[int((train_frac + val_frac) * N)]

    train = df[df[date_col] <= train_dt].copy()
    val   = df[(df[date_col] > train_dt) & (df[date_col] <= val_dt)].copy()
    test  = df[df[date_col] > val_dt].copy()

    return train, val, test
