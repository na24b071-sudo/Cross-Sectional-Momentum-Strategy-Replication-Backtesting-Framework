# src/features.py
import pandas as pd
import numpy as np

def compute_monthly_momentum_from_daily(daily_df, lookback_months=12, skip_months=1):
    """
    Aggregate daily OHLCV to month-end and compute:
      - monthly_volume
      - adj_close (month-end)
      - ret_1m (simple 1-month return)
      - mom_J (compounded J-month return excluding last skip_months)

    Returns DataFrame with columns ['ticker','date','adj_close','monthly_volume','ret_1m','mom_J']
    """
    df = daily_df.copy()

    # normalize date and price columns defensively
    df['date'] = pd.to_datetime(df.get('date', df.get('Date', None)), errors='coerce')
    df = df.dropna(subset=['date'])
    # numeric prices
    if 'adj_close' in df.columns:
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
    elif 'Adj Close' in df.columns:
        df['adj_close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
    elif 'close' in df.columns:
        df['adj_close'] = pd.to_numeric(df['close'], errors='coerce')
    else:
        df['adj_close'] = np.nan

    # numeric volume
    df['volume'] = pd.to_numeric(df.get('volume', df.get('Volume', np.nan)), errors='coerce').fillna(0)

    # group to month-end: get last price in month and sum volume
    monthly = (
        df.groupby(['ticker', pd.Grouper(key='date', freq='ME')])
          .agg(adj_close=('adj_close','last'),
               monthly_volume=('volume','sum'))
          .reset_index()
    )

    monthly = monthly.sort_values(['ticker','date']).reset_index(drop=True)

    # compute 1-month simple return (for diagnostics and for building mom_J)
    monthly['ret_1m'] = monthly.groupby('ticker')['adj_close'].pct_change()

    # compute mom_J: shift by skip_months then rolling compound product over lookback_months
    def _compound_returns(returns):
        shifted = returns.shift(skip_months)
        # rolling product: (1+r1)*(1+r2)*... - 1
        return shifted.rolling(window=lookback_months, min_periods=1).apply(lambda r: np.prod(1 + r) - 1, raw=True)

    monthly['mom_J'] = monthly.groupby('ticker')['ret_1m'].apply(_compound_returns).reset_index(level=0, drop=True)

    # Keep columns in a stable order
    monthly = monthly[['ticker','date','adj_close','monthly_volume','ret_1m','mom_J']]

    return monthly


def compute_monthly_turnover(monthly_df, shares_info_map=None, lookback_months=3):
    """
    Given a monthly DataFrame (output of compute_monthly_momentum_from_daily),
    compute turnover measures:
      - adv_est (monthly_volume / ~21 trading days)
      - shares_outstanding (from shares_info_map if provided; else NaN)
      - turnover_monthly = adv_est / shares_outstanding
      - turn_avg: rolling mean of turnover over lookback_months

    Input monthly_df should contain ['ticker','date','monthly_volume','adj_close'].
    shares_info_map: dict ticker -> {'shares_outstanding': int, 'market_cap': int}

    Returns monthly_df augmented with the turnover columns.
    """
    df = monthly_df.copy()
    # ensure numeric
    df['monthly_volume'] = pd.to_numeric(df.get('monthly_volume', df.get('volume', np.nan)), errors='coerce').fillna(0)

    # estimate adv (daily avg) from monthly volume
    df['adv_est'] = df['monthly_volume'] / 21.0

    # get shares outstanding from provided map if possible
    def _get_shares(row):
        t = row['ticker']
        if isinstance(shares_info_map, dict):
            info = shares_info_map.get(t, {})
            so = info.get('shares_outstanding')
            if so is not None and not pd.isna(so):
                return so
            # fallback: estimate from market cap if available and price present
            mcap = info.get('market_cap')
            price = row.get('adj_close', np.nan)
            if mcap and price and price > 0:
                try:
                    return int(mcap / price)
                except Exception:
                    return np.nan
        return np.nan

    df['shares_outstanding'] = df.apply(_get_shares, axis=1)

    # compute turnover (guard against division by zero / NaN)
    df['turnover_monthly'] = np.where(df['shares_outstanding'] > 0, df['adv_est'] / df['shares_outstanding'], np.nan)

    # rolling average of turnover
    df['turn_avg'] = df.groupby('ticker')['turnover_monthly'].rolling(lookback_months, min_periods=1).mean().reset_index(level=0, drop=True)

    return df


def compute_intraday_features_minute(minute_df, window_minutes=30):
    """
    Build minute-level intraday features from minute bars.
    Accepts a DataFrame with ['datetime','ticker','price','volume'] (or variants).
    Returns:
      ['datetime','ticker','price','ret_1m','ret_5m','vol_roll_sum','vol_zscore','signed_vol_roll']
    """
    df = minute_df.copy()
    df['datetime'] = pd.to_datetime(df.get('datetime', df.get('Datetime', None)), errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['price'] = pd.to_numeric(df.get('price', df.get('Close', np.nan)), errors='coerce')
    df['volume'] = pd.to_numeric(df.get('volume', df.get('Volume', np.nan)), errors='coerce')
    df = df.sort_values(['ticker','datetime']).reset_index(drop=True)

    df['price_lag1'] = df.groupby('ticker')['price'].shift(1)
    df['ret_1m'] = df['price'] / df['price_lag1'] - 1
    df['ret_5m'] = df.groupby('ticker')['ret_1m'].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)

    df['tick_sign'] = (np.sign(df['price'] - df['price_lag1']).fillna(0)).astype(int)
    df['signed_volume'] = df['tick_sign'] * df['volume']

    df['vol_roll_sum'] = df.groupby('ticker')['volume'].rolling(window_minutes, min_periods=1).sum().reset_index(level=0, drop=True)
    df['signed_vol_roll'] = df.groupby('ticker')['signed_volume'].rolling(window_minutes, min_periods=1).sum().reset_index(level=0, drop=True)

    df['vol_roll_mean_60'] = df.groupby('ticker')['vol_roll_sum'].rolling(60, min_periods=1).mean().reset_index(level=0, drop=True)
    df['vol_roll_std_60'] = df.groupby('ticker')['vol_roll_sum'].rolling(60, min_periods=1).std().reset_index(level=0, drop=True).fillna(1.0)
    df['vol_zscore'] = (df['vol_roll_sum'] - df['vol_roll_mean_60']) / df['vol_roll_std_60']

    out_cols = ['datetime','ticker','price','ret_1m','ret_5m','vol_roll_sum','vol_zscore','signed_vol_roll']
    # ensure output columns exist even if some are NaN
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[out_cols].copy()
