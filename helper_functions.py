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


# load data function

def load_data():
    prices = pd.read_csv("data/prices_panel.csv", parse_dates = ["Date"])
    funds = pd.read_csv("data/fundamentals_income.csv", parse_dates = ["Report Date", "Publish Date", "Restated Date"])

    # change column names so it easier to type / access them
    prices.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in prices.columns]
    funds.columns = [c.strip().lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace(',', '') for c in funds.columns]

    # Drop duplicates

    prices = prices.sort_values(["ticker", "date"]).drop_duplicates()
    funds = funds.sort_values(["ticker", "report_date"]).drop_duplicates()

    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    funds = funds.sort_values(["report_date", "ticker"]).reset_index(drop=True)

    df = pd.merge_asof(
        prices,
        funds,
        by="ticker",
        left_on="date",
        right_on="report_date",
        direction="backward",
        )

    return df


def create_fundamental_features(df):

    df["ret"] = df.groupby("ticker")['adj_close'].pct_change()

    # Build fundamental features
    # EPS, profit margin, revenue growth, and other ratios derived from reported items.

    df['eps'] = df["net_income"] / df['shares_diluted']
    df['profit_margin'] = df['net_income'] / df['revenue']
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change()

    df["income_growth"] = df.groupby("ticker")["net_income"].pct_change()

    # other ratios
    df["gross_margin"] = df["gross_profit"] / df["revenue"]
    df["operating_margin"] = df["operating_income_loss"] / df["revenue"]
    df["sga_ratio"] = df["selling_general_&_administrative"] / df["revenue"]
    df["rd_ratio"] = df["research_&_development"] / df["revenue"]
    df["cost_ratio"] = df["cost_of_revenue"] / df["revenue"]
    df["net_income_per_share"] = df["net_income"] / df["shares_diluted"]
    df["tax_burden"] = df["net_income"] / df["pretax_income_loss_adj"]
    df["nonop_ratio"] = df["non-operating_income_loss"] / df["revenue"]
    df["abnormal_ratio"] = df["abnormal_gains_losses"] / df["revenue"]



    # efficiency features
    df["revenue_per_share"] = df["revenue"] / df["shares_diluted"]
    df["net_income_per_share"] = df["net_income"] / df["shares_diluted"]

    df["da_ratio"] = df["depreciation_&_amortization"] / df["revenue"]
    df["interest_coverage"] = df["operating_income_loss"] / df["interest_expense_net"]
    df["interest_burden"] = df["pretax_income_loss_adj"] / df["operating_income_loss"]

    return df

def create_engineered_features(df):
    # engineered features:  momentum ratios, EMA crossovers, skewness, kurtosis, etc.
    g = df.groupby("ticker")

    # rolling means
    df['mean_20'] = g['ret'].transform(lambda x: x.rolling(20, 10).mean())
    df['mean_60'] = g['ret'].transform(lambda x: x.rolling(60, 20).mean())
    # rolling vol
    df['vol_20'] = g['ret'].transform(lambda x: x.rolling(20, 10).std())
    df['vol_60'] = g['ret'].transform(lambda x: x.rolling(60, 20).std())

    # momentum
    df['mom_5'] = g['adj_close'].transform(lambda x: x / x.shift(5) - 1)
    df['mom_20'] = g['adj_close'].transform(lambda x: x / x.shift(20) - 1)
    df['mom_60'] = g['adj_close'].transform(lambda x: x / x.shift(60) - 1)

    #  EMA  
    df['ema_12'] = g['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['ema_26'] = g['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())

    # EMA crossover
    df['ema_cross'] = df['ema_12'] - df['ema_26']

    # skewness & kurtosis 
    df['skew_20'] = g['ret'].transform(lambda x: x.rolling(20).skew())
    df['skew_60'] = g['ret'].transform(lambda x: x.rolling(60).skew())

    df['kurt_20'] = g['ret'].transform(lambda x: x.rolling(20).kurt())
    df['kurt_60'] = g['ret'].transform(lambda x: x.rolling(60).kurt())

    # volume Z-Score
    df['vol_z'] = g['volume'].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
    )

    return df


def create_binary_labels(df, h):

    # NOTE: in here we are deleting observation which might be a problem

    #labels = [1, 20, 60]

    g = df.groupby('ticker')

    df[f'cumret_{h}'] = g['ret'].transform(lambda x: (1 + x).rolling(h).apply(lambda r: np.prod(r) - 1).shift(-h + 1))

    df = df.dropna(subset=[f'cumret_{h}'])

    df[f'y_{h}'] = (df[f'cumret_{h}'] > 0).astype(int)

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

