import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import tensorflow as tf
# from tensorflow.keras import layers, models

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
    """
    Build engineered features
    Momentum ratios, EMA crossovers, skewness, kurtosis, etc.
    """

    g = df.groupby("ticker")

    # rolling means
    df['mean_20'] = g['ret'].transform(lambda x: x.rolling(20).mean())
    df['mean_60'] = g['ret'].transform(lambda x: x.rolling(60).mean())

    # rolling vol
    df['vol_20'] = g['ret'].transform(lambda x: x.rolling(20).std())
    df['vol_60'] = g['ret'].transform(lambda x: x.rolling(60).std())

    # NOTE: potentially switch to momentum ratio
    df['mom_20'] = g['adj_close'].transform(lambda x: x.pct_change(20))
    df['mom_60'] = g['adj_close'].transform(lambda x: x.pct_change(60))

    # momentum
    # df['mom_5'] = g['adj_close'].transform(lambda x: x / x.shift(5) - 1)
    # df['mom_20'] = g['adj_close'].transform(lambda x: x / x.shift(20) - 1)
    # df['mom_60'] = g['adj_close'].transform(lambda x: x / x.shift(60) - 1)

    # ema
    df['ema_12'] = g['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['ema_26'] = g['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())

    df["ema_cross"] = df['ema_12'] - df['ema_26']
    # remove lookahead bias
    df["ema_cross"] = g['ema_cross'].shift(1)

    # skewness
    df['skew_20'] = g['ret'].transform(lambda x: x.rolling(20).skew())
    df['skew_60'] = g['ret'].transform(lambda x: x.rolling(60).skew())

    # kurtosis
    df['kurt_20'] = g['ret'].transform(lambda x: x.rolling(20).kurt())
    df['kurt_60'] = g['ret'].transform(lambda x: x.rolling(60).kurt())

    # volume Z-score
    df['vol_z'] = g['volume'].transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std())

    return df

def create_engineered_plus_fundamental_features(df):
    df = create_raw_features(df)
    df = create_engineered_features(df)
    df = create_fundamental_features(df)
    df["eps_mom_20"] = df["income_growth"] * df["mom_20"]
    df["eps_mom_60"] = df["income_growth"] * df["mom_60"]

    # 2. Profitability per unit of volatility
    df["profit_vol_adj_20"] = df["profit_margin"] / df["vol_20"]
    df["profit_vol_adj_60"] = df["profit_margin"] / df["vol_60"]

    # 3. Revenue growth × momentum confirmation
    df["rev_growth_mom"] = df["revenue_growth"] * df["mom_20"]

    # 4. Volume-weighted earnings signal (income growth × vol_z)
    df["rev_signal_vol_w"] = df["income_growth"] * df["vol_z"]

    # 5. Quality × EMA trend confirmation
    df["quality_trend"] = df["gross_margin"] * df["ema_cross"]

    return df

def create_binary_labels(df, horizons):

    g = df.groupby('ticker')

    for h in horizons:
        df[f'cumret_{h}'] = g['ret'].transform(
            lambda x: (1 + x.shift(-1))
            .rolling(h, h)
            .apply(lambda r: np.prod(r) - 1).shift(-h + 1))

        df[f'y_{h}'] = np.where(
            df[f'cumret_{h}'].notna(),
            (df[f'cumret_{h}'] > 0).astype(int),
            np.nan
        )

        df.drop(f'cumret_{h}', axis = 1, inplace = True)

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

def prune(df, features, target):
    cols = features + ["date", target]
    data = df.dropna(subset = cols).copy()
    return data
