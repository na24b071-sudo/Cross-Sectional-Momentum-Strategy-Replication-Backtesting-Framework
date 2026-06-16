import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sharpe(returns, freq_per_year=252):
    rs = np.array(returns)
    if len(rs) == 0:
        return float('nan')
    mean = rs.mean() * freq_per_year
    sd = rs.std(ddof=1) * (freq_per_year ** 0.5)
    if sd == 0:
        return float('nan')
    return mean / sd

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
