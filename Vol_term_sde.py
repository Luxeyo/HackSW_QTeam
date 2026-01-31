"""
GARCH volatility estimation with visualizations
"""

import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt


def estimate_sigma_garch_with_viz(token_ticker, days=60):
    """
    Estimate σ_GARCH from 60 days of hourly data + create visualizations

    Parameters:
    -----------
    token_ticker : str
        Crypto ticker (e.g., "BTC-USD", "ETH-USD", "SOL-USD")
    days : int
        Lookback period (default 60)

    Returns:
    --------
    sigma_daily : float
        Daily volatility for SDE
    """
    print(f"\n{'=' * 60}")
    print(f"GARCH VOLATILITY ANALYSIS: {token_ticker}")
    print(f"{'=' * 60}\n")

    # Download hourly data
    print(f"📥 Downloading {days} days of hourly data...")
    token = yf.Ticker(token_ticker)
    data = token.history(period=f"{days}d", interval="1h")

    if len(data) < 100:
        raise ValueError(f"Insufficient data for {token_ticker}")

    prices = data["Close"].values
    timestamps = data.index

    # Hourly log returns (in percentage for GARCH numerical stability)
    returns = np.diff(np.log(prices)) * 100

    # Fit GARCH(1,1)
    print(f"🔄 Fitting GARCH(1,1) model...")
    model = arch_model(
        returns,
        vol='Garch',
        p=1,
        q=1,
        mean='Zero',
        rescale=False
    )

    fitted = model.fit(disp='off')

    # Get current conditional volatility
    forecast = fitted.forecast(horizon=1)
    sigma_hourly = np.sqrt(forecast.variance.values[-1, 0]) / 100
    sigma_daily = sigma_hourly * np.sqrt(24)

    # Get conditional volatility time series
    cond_vol_hourly = fitted.conditional_volatility / 100
    cond_vol_daily = cond_vol_hourly * np.sqrt(24)

    # Get GARCH parameters
    omega = fitted.params['omega']
    alpha = fitted.params['alpha[1]']
    beta = fitted.params['beta[1]']
    persistence = alpha + beta

    # Create visualizations
    print(f"🎨 Creating visualizations...")
    create_garch_visualization(
        token_ticker=token_ticker,
        timestamps=timestamps,
        prices=prices,
        returns=returns / 100,  # Back to decimal for plotting
        cond_vol_daily=cond_vol_daily,
        sigma_daily=sigma_daily,
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"GARCH VOLATILITY RESULTS")
    print(f"{'=' * 60}")
    print(f"\n📊 Current Volatility:")
    print(f"  σ_GARCH (daily):   {sigma_daily:.6f}")
    print(f"  σ_GARCH (hourly):  {sigma_hourly:.6f}")
    print(f"  σ_GARCH (%/day):   {sigma_daily * 100:.2f}%")
    print(f"\n⚙️ GARCH(1,1) Parameters:")
    print(f"  ω (omega):         {omega:.6f}")
    print(f"  α (alpha):         {alpha:.6f}")
    print(f"  β (beta):          {beta:.6f}")
    print(f"  Persistence (α+β): {persistence:.6f}")

    if persistence > 0.95:
        print(f"\n⚠️  High persistence → volatility shocks decay slowly")

    print(f"\n✅ Use in your SDE:")
    print(f"  dS_t = μ S_t dt + {sigma_daily:.6f} S_t dW_t + S_t(e^Y-1)dN_t")
    print(f"{'=' * 60}\n")

    return sigma_daily


def create_garch_visualization(
        token_ticker, timestamps, prices, returns, cond_vol_daily,
        sigma_daily, omega, alpha, beta, persistence
):
    """
    Create 4-panel GARCH visualization
    """
    fig = plt.figure(figsize=(18, 10))

    # === 1. PRICE TIME SERIES ===
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(timestamps, prices, linewidth=1.5, color='blue')
    ax1.set_title(f'{token_ticker} Price (60 days, hourly)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(alpha=0.3)

    # === 2. RETURNS TIME SERIES ===
    ax2 = plt.subplot(2, 2, 2)
    returns_timestamps = timestamps[1:]  # Returns start at index 1
    ax2.plot(returns_timestamps, returns * 100, linewidth=0.8, alpha=0.7, color='purple')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Hourly Log Returns', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(alpha=0.3)

    # === 3. CONDITIONAL VOLATILITY (Main plot) ===
    ax3 = plt.subplot(2, 2, 3)
    vol_timestamps = timestamps[1:]  # Volatility aligns with returns
    ax3.plot(vol_timestamps[-len(cond_vol_daily):], cond_vol_daily * 100,
             linewidth=2, color='red', label='GARCH Conditional Vol')
    ax3.axhline(sigma_daily * 100, color='green', linestyle='--', linewidth=2.5,
                label=f'Current σ = {sigma_daily * 100:.2f}%/day')

    # Add shaded regions for high/low vol periods
    mean_vol = np.mean(cond_vol_daily)
    ax3.axhspan(0, mean_vol * 100, alpha=0.1, color='green', label='Low Vol Region')
    ax3.axhspan(mean_vol * 100, max(cond_vol_daily) * 100, alpha=0.1, color='red', label='High Vol Region')

    ax3.set_title('GARCH(1,1) Conditional Volatility', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Volatility (%/day)')
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.3)

    # === 4. RETURNS vs VOLATILITY (Volatility Clustering) ===
    ax4 = plt.subplot(2, 2, 4)

    # Absolute returns (proxy for realized volatility)
    abs_returns = np.abs(returns[-len(cond_vol_daily):])

    ax4.scatter(cond_vol_daily * 100, abs_returns * 100, alpha=0.4, s=20)

    # Add trend line
    z = np.polyfit(cond_vol_daily * 100, abs_returns * 100, 1)
    p = np.poly1d(z)
    vol_sorted = np.sort(cond_vol_daily * 100)
    ax4.plot(vol_sorted, p(vol_sorted), 'r--', linewidth=2, label='Trend')

    ax4.set_title('Volatility Clustering', fontsize=14, fontweight='bold')
    ax4.set_xlabel('GARCH Conditional Vol (%/day)')
    ax4.set_ylabel('Absolute Return (%)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add text box with GARCH parameters
    textstr = f'GARCH(1,1) Parameters:\nω = {omega:.6f}\nα = {alpha:.6f}\nβ = {beta:.6f}\nPersistence = {persistence:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(f'{token_ticker}_garch_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Saved: {token_ticker}_garch_analysis.png")


# ===== USAGE =====

if __name__ == "__main__":
    # User input
    user_ticker = input("Enter crypto ticker (e.g., BTC-USD, ETH-USD, SOL-USD): ").strip().upper()

    if '-USD' not in user_ticker:
        user_ticker = f"{user_ticker}-USD"

    try:
        sigma = estimate_sigma_garch_with_viz(user_ticker, days=60)

        print(f"\n🎯 Final Result: σ_GARCH = {sigma:.6f} (daily)")

    except Exception as e:
        print(f"\n❌ Error: {e}")





