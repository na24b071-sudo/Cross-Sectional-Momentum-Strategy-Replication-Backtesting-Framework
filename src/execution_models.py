# src/execution_models.py
import numpy as np

def square_root_impact(size_shares, adv_shares, volatility, k=0.1, expo=0.5):
    if adv_shares <= 0:
        return 0.0
    return k * volatility * (abs(size_shares) / adv_shares) ** expo

def simulate_market_fill(price, size_shares, adv_shares, volatility, side=1, spread=0.001):
    impact = square_root_impact(size_shares, adv_shares, volatility)
    executed_price = price * (1 + side * (spread/2.0 + impact))
    return executed_price, impact

def simulate_limit_fill(price, size_shares, adv_shares, volatility, aggressiveness=0.5):
    import numpy as np
    p_fill = 0.2 + 0.7 * aggressiveness
    size_frac = min(1.0, abs(size_shares) / max(1.0, adv_shares))
    p_full_fill = p_fill * (1 - 0.5 * size_frac)
    filled = np.random.rand() < p_full_fill
    executed_price = price * (1 - 0.5 * aggressiveness * 0.001)
    expected_slippage = square_root_impact(size_shares, adv_shares, volatility) * (1 - aggressiveness)
    return filled, executed_price, expected_slippage
