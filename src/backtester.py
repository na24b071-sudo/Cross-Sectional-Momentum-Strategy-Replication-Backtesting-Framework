# src/backtester.py
import numpy as np
import pandas as pd
from collections import defaultdict
from .execution_models import simulate_market_fill

class SimpleEventBacktester:
    def __init__(self, minute_features_df, shares_map, adv_map, vol_map, cash=1_000_000, latency_ms=0):
        self.df = minute_features_df.copy().sort_values(['datetime','ticker']).reset_index(drop=True)
        self.shares_map = shares_map
        self.adv_map = adv_map
        self.vol_map = vol_map
        self.cash = cash
        self.latency_ms = latency_ms
        self.positions = defaultdict(int)
        self.trade_log = []
        self.pnl_series = []
        self.portfolio_value_series = []

    def run_strategy(self, predict_fn, size_shares=100, threshold=0.0):
        grouped = self.df.groupby('datetime')
        last_value = None
        for dt, group in grouped:
            orders = []
            for _, row in group.iterrows():
                ticker = row['ticker']
                price = row['price']
                score = predict_fn(row)
                if score > threshold:
                    orders.append((ticker, +abs(size_shares), price, score))
                elif score < -threshold:
                    orders.append((ticker, -abs(size_shares), price, score))
            # execute orders (immediate market orders with impact)
            for ticker, size, price, score in orders:
                adv = self.adv_map.get(ticker, 100_000)
                vol = self.vol_map.get(ticker, 0.02)
                side = 1 if size > 0 else -1
                executed_price, impact = simulate_market_fill(price, size, adv, vol, side=side)
                # position and cash
                self.positions[ticker] += int(size)
                self.cash -= executed_price * size
                self.trade_log.append({
                    'datetime': dt, 'ticker': ticker, 'size': int(size), 'price': executed_price, 'impact': impact, 'score': score
                })
            # mark-to-market
            pv = self.cash
            for tkr, pos in self.positions.items():
                # use latest price available in group or fallback to last price in df before dt
                latest_prices = group.loc[group['ticker'] == tkr, 'price']
                if len(latest_prices) > 0:
                    p = latest_prices.iloc[-1]
                else:
                    prior = self.df.loc[(self.df['ticker'] == tkr) & (self.df['datetime'] <= dt)]
                    if len(prior) > 0:
                        p = prior.loc[prior['datetime'] == prior['datetime'].max(), 'price'].iloc[-1]
                    else:
                        p = 0.0
                pv += pos * p
            if last_value is None:
                pnl = 0.0
            else:
                pnl = pv - last_value
            self.pnl_series.append({'datetime': dt, 'pnl': pnl})
            self.portfolio_value_series.append(pv)
            last_value = pv

    def results(self):
        pnl_df = pd.DataFrame(self.pnl_series)
        trades_df = pd.DataFrame(self.trade_log)
        return pnl_df, trades_df, dict(self.positions), self.cash, self.portfolio_value_series
