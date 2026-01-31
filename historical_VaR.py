import numpy as np
import pandas as pd

def historical_drawdown(price, horizon, percentile):
    df = pd.DataFrame(price, columns=['close'])
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    rolling_log_returns = log_returns.rolling(horizon).sum()
    rolling_simple_returns = np.exp(rolling_log_returns) - 1
    historical_VaR = np.percentile(rolling_simple_returns.dropna(), percentile)
    return historical_VaR
