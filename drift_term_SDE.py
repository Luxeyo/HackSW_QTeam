import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def mc_mu_total_distribution(
    token="SOL-USD",
    btc="BTC-USD",
    days=60,
    horizon_hours=24,
    n_sims=5000,
    arma_order=(1, 0, 1),
    seed=42
):
    rng = np.random.default_rng(seed)

    # 1) Download last `days` of HOURLY close prices
    token_px = yf.Ticker(token).history(period=f"{days}d", interval="1h")["Close"]
    btc_px   = yf.Ticker(btc).history(period=f"{days}d", interval="1h")["Close"]

    # Align timestamps
    df = token_px.to_frame("token").join(btc_px.to_frame("btc"), how="inner").dropna()

    if len(df) < 200:
        raise ValueError("Not enough hourly data returned.")

    # 2) Hourly log returns
    r_token = np.log(df["token"]).diff().dropna().values
    r_btc   = np.log(df["btc"]).diff().dropna().values

    n = min(len(r_token), len(r_btc))
    r_token, r_btc = r_token[-n:], r_btc[-n:]

    # 3) Estimate beta
    X = sm.add_constant(r_btc)
    beta_hat = sm.OLS(r_token, X).fit().params[1]

    # Residuals
    resid = r_token - beta_hat * r_btc

    # 4) ARMA on residuals
    arma = ARIMA(resid, order=arma_order).fit()

    # 5) Monte Carlo mu_total
    mu_total = np.zeros(n_sims)

    for i in range(n_sims):
        btc_future = rng.choice(r_btc, size=horizon_hours, replace=True)
        btc_momentum = btc_future.sum()

        resid_future = arma.simulate(nsimulations=horizon_hours, random_state=rng)
        mu_idio = resid_future.sum()

        mu_total[i] = mu_idio + beta_hat * btc_momentum

    return mu_total, beta_hat


def plot_mu_distribution(mu_dist, token, horizon_hours, bins=60):
    mean = mu_dist.mean()
    p1, p5, p50, p95, p99 = np.percentile(mu_dist, [1, 5, 50, 95, 99])

    plt.figure(figsize=(11, 5))
    plt.hist(mu_dist, bins=bins, density=True, edgecolor="black", alpha=0.75)

    plt.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.4f}")
    plt.axvline(p1, linestyle="--", linewidth=2, label=f"1% = {p1:.4f}")
    plt.axvline(p5, linestyle="--", linewidth=2, label=f"5% = {p5:.4f}")
    plt.axvline(p50, linestyle="--", linewidth=2, label=f"Median = {p50:.4f}")
    plt.axvline(p95, linestyle="--", linewidth=2, label=f"95% = {p95:.4f}")
    plt.axvline(p99, linestyle="--", linewidth=2, label=f"99% = {p99:.4f}")

    plt.title(f"{token} μ_total distribution over {horizon_hours} hours")
    plt.xlabel("μ_total (log return over horizon)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mu_dist, beta_hat = mc_mu_total_distribution(
        token="SOL-USD",
        btc="BTC-USD",
        days=60,
        horizon_hours=24,
        n_sims=5000,
        arma_order=(1, 0, 1),
        seed=42
    )

    p1, p5, p50, p95, p99 = np.percentile(mu_dist, [1, 5, 50, 95, 99])

    print("μ_total distribution (log-return over 24h)")
    print(f"beta_hat: {beta_hat:.3f}")
    print(f"Mean:     {mu_dist.mean():.6f}")
    print(f"1%:       {p1:.6f}")
    print(f"5%:       {p5:.6f}")
    print(f"Median:   {p50:.6f}")
    print(f"95%:      {p95:.6f}")
    print(f"99%:      {p99:.6f}")

    plot_mu_distribution(mu_dist, token="SOL-USD", horizon_hours=24)


