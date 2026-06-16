# src/data_io.py
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def cache_path(ticker, freq):
    return os.path.join(DATA_DIR, f"{ticker}_{freq}.csv")

# canonical required columns for daily/intraday
_DAILY_REQUIRED = ['date','ticker','open','high','low','close','adj_close','volume']
_INTRADAY_REQUIRED = ['datetime','ticker','price','volume']

def _drop_duplicate_columns(df):
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def _normalize_daily_columns(df, ticker):
    """
    Normalize input daily DataFrame into canonical lowercase schema.
    Defensive renaming + numeric coercion. Returns DataFrame with _DAILY_REQUIRED columns.
    """
    df = pd.DataFrame(df)  # ensure DataFrame
    df = _drop_duplicate_columns(df)

    # if Adj Close missing but Close exists, create it
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    # ensure ticker column exists
    if 'Ticker' not in df.columns and 'ticker' not in df.columns:
        df['Ticker'] = ticker

    # rename only existing columns to canonical names
    rename_map = {}
    if 'Date' in df.columns: rename_map['Date'] = 'date'
    if 'date' in df.columns: rename_map['date'] = 'date'
    if 'Ticker' in df.columns: rename_map['Ticker'] = 'ticker'
    if 'ticker' in df.columns: rename_map['ticker'] = 'ticker'
    if 'Open' in df.columns: rename_map['Open'] = 'open'
    if 'High' in df.columns: rename_map['High'] = 'high'
    if 'Low' in df.columns: rename_map['Low'] = 'low'
    if 'Close' in df.columns: rename_map['Close'] = 'close'
    if 'Adj Close' in df.columns: rename_map['Adj Close'] = 'adj_close'
    if 'Volume' in df.columns: rename_map['Volume'] = 'volume'

    df = df.rename(columns=rename_map)

    # coerce date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT

    # coerce numeric columns: strings => NaN
    for col in ['open','high','low','close','adj_close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # ensure all required columns present
    for col in _DAILY_REQUIRED:
        if col not in df.columns:
            df[col] = np.nan

    # return canonical order (copy to avoid SettingWithCopy warnings downstream)
    return df[_DAILY_REQUIRED].copy()

def _normalize_intraday_columns(df, ticker):
    """
    Normalize intraday DataFrame into canonical ['datetime','ticker','price','volume'].
    Coerces numeric types and datetimes.
    """
    df = pd.DataFrame(df)
    df = _drop_duplicate_columns(df)

    # map common column names to canonical ones
    if 'Close' in df.columns and 'price' not in df.columns:
        df = df.rename(columns={'Close':'price'})
    if 'Adj Close' in df.columns and 'price' not in df.columns:
        df = df.rename(columns={'Adj Close':'price'})
    if 'Volume' in df.columns and 'volume' not in df.columns:
        df = df.rename(columns={'Volume':'volume'})

    # datetime variants
    if 'Datetime' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'Datetime':'datetime'})
    if 'Date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'Date':'datetime'})

    # ticker casing
    if 'Ticker' in df.columns and 'ticker' not in df.columns:
        df = df.rename(columns={'Ticker':'ticker'})
    if 'ticker' not in df.columns:
        df['ticker'] = ticker

    # coerce datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        df['datetime'] = pd.NaT

    # coerce numeric price/volume
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    else:
        # try common column names
        if 'Close' in df.columns:
            df['price'] = pd.to_numeric(df['Close'], errors='coerce')
        else:
            df['price'] = np.nan

    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    else:
        df['volume'] = np.nan

    # ensure canonical columns
    for col in _INTRADAY_REQUIRED:
        if col not in df.columns:
            df[col] = np.nan

    return df[_INTRADAY_REQUIRED].copy()

def fetch_daily(tickers, start=None, end=None, interval='1d', force_refresh=False, verbose=True):
    """
    Fetch daily OHLCV for tickers using yfinance, caching per-ticker CSVs in data/.
    Returns a concatenated DataFrame with columns:
      ['date','ticker','open','high','low','close','adj_close','volume']

    Parameters:
      tickers: list[str]
      start: str 'YYYY-MM-DD' or None
      end: str 'YYYY-MM-DD' or None
      interval: '1d' typically
      force_refresh: if True, re-download even if cached file exists
      verbose: if True, print progress
    """
    all_rows = []
    for t in tickers:
        try:
            p = cache_path(t, 'daily')
            if os.path.exists(p) and not force_refresh:
                # read cached CSV without assuming exact column-case
                df = pd.read_csv(p, low_memory=False)
            else:
                df = yf.download(t, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
                if df is None or df.empty:
                    if verbose:
                        print(f"[fetch_daily] warning: yfinance returned no data for {t}")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df.to_csv(p, index=False)

            df = _normalize_daily_columns(df, t)
            # drop invalid dates
            df = df.dropna(subset=['date'])
            if df.shape[0] == 0:
                if verbose:
                    print(f"[fetch_daily] warning: after normalization no valid rows for {t}")
                continue

            all_rows.append(df)
            if verbose:
                print(f"[fetch_daily] loaded {t} rows={len(df)}")
            time.sleep(0.05)
        except Exception as e:
            print(f"[fetch_daily] error loading {t}: {repr(e)} — skipping ticker.")
            continue

    if len(all_rows) == 0:
        return pd.DataFrame(columns=_DAILY_REQUIRED)

    return pd.concat(all_rows, ignore_index=True)

def fetch_intraday(tickers, period='7d', interval='1m', force_refresh=False, verbose=True):
    """
    Fetch intraday bars for tickers (yfinance usually supports only recent history).
    Returns concatenated DataFrame with columns ['datetime','ticker','price','volume'].

    Parameters:
      tickers: list[str]
      period: e.g., '7d'
      interval: e.g., '1m'
      force_refresh: if True, re-download
      verbose: print progress
    """
    all_rows = []
    for t in tickers:
        try:
            p = cache_path(t, 'intraday')
            if os.path.exists(p) and not force_refresh:
                df = pd.read_csv(p, low_memory=False)
            else:
                df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False)
                if df is None or df.empty:
                    if verbose:
                        print(f"[fetch_intraday] no intraday data for {t}")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df.to_csv(p, index=False)

            df = _normalize_intraday_columns(df, t)
            df = df.dropna(subset=['datetime'])
            if df.shape[0] == 0:
                if verbose:
                    print(f"[fetch_intraday] after normalization no valid rows for {t}")
                continue

            all_rows.append(df)
            if verbose:
                print(f"[fetch_intraday] loaded {t} rows={len(df)}")
            time.sleep(0.05)
        except Exception as e:
            print(f"[fetch_intraday] error loading {t}: {repr(e)} — skipping ticker.")
            continue

    if len(all_rows) == 0:
        return pd.DataFrame(columns=_INTRADAY_REQUIRED)

    # concatenate — each df already has identical columns
    return pd.concat(all_rows, ignore_index=True)

def get_shares_info(tickers, force_refresh=False, verbose=True):
    """
    Fetch shares outstanding and market cap via yfinance.Ticker.info.
    Returns dict ticker -> {'shares_outstanding': int|None, 'market_cap': int|None}
    """
    info_map = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            inf = tk.info or {}
            shares = inf.get('sharesOutstanding', None)
            mcap = inf.get('marketCap', None)
            info_map[t] = {'shares_outstanding': shares, 'market_cap': mcap}
            if verbose:
                print(f"[get_shares_info] {t}: shares={shares}, mcap={mcap}")
            time.sleep(0.05)
        except Exception as e:
            info_map[t] = {'shares_outstanding': None, 'market_cap': None}
            print(f"[get_shares_info] warning: failed to fetch info for {t}: {repr(e)}")
    return info_map

def minute_fallback_from_daily(daily_df, tickers, minutes_per_day=390):
    """
    Generate synthetic intraday 1-minute bars from daily OHLCV as a fallback.
    Returns a DataFrame with columns ['datetime','ticker','price','volume'].
    """
    rows = []
    daily_df = pd.DataFrame(daily_df)
    # normalize date column
    if 'date' in daily_df.columns:
        daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
    elif 'Date' in daily_df.columns:
        daily_df['date'] = pd.to_datetime(daily_df['Date'], errors='coerce')

    for t in tickers:
        sub = daily_df[daily_df.get('ticker') == t].sort_values('date')
        for _, r in sub.iterrows():
            if pd.isna(r.get('date')):
                continue
            # coerce numeric open/close
            open_p = pd.to_numeric(r.get('open', r.get('Open', np.nan)), errors='coerce')
            close_p = pd.to_numeric(r.get('close', r.get('Close', np.nan)), errors='coerce')
            if pd.isna(open_p) or pd.isna(close_p):
                continue
            day = pd.to_datetime(r['date']).date()
            day_index = pd.date_range(
                start=pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30),
                periods=minutes_per_day,
                freq='min'  # avoid deprecated 'T'
            )
            # linear price path + tiny noise
            path = np.linspace(float(open_p), float(close_p), minutes_per_day) * (1 + np.random.normal(0, 0.0005, minutes_per_day))

            # volume handling
            vol_val = r.get('volume', r.get('Volume', np.nan))
            try:
                vol_val = int(np.nan if pd.isna(vol_val) else vol_val)
            except Exception:
                vol_val = np.nan
            if pd.isna(vol_val) or vol_val <= 0:
                vol_val = 1
            base_vol = np.sin(np.linspace(0, np.pi, minutes_per_day))**2 + 0.1
            base_vol /= base_vol.sum()
            minute_vol = (base_vol * max(1, int(vol_val))).astype(int)

            for dt, p_, v in zip(day_index, path, minute_vol):
                rows.append({'datetime': dt, 'ticker': t, 'price': float(p_), 'volume': int(v)})

    if len(rows) == 0:
        return pd.DataFrame(columns=_INTRADAY_REQUIRED)
    return pd.DataFrame(rows)
