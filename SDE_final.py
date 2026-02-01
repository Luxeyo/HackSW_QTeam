"""
Complete SDE Model with Mean Reversion -  Year Daily Data
EXPANDED: 50+ Popular Crypto Tickers
"""

import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
import requests
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# TICKER MAPPING - 50+ POPULAR COINS
# ==========================================

TICKER_TO_BLOCKCHAIN = {
    # === TOP 10 BY MARKET CAP ===
    "BTC-USD": {"symbol": "BTC", "type": "native", "chain": "bitcoin"},
    "ETH-USD": {"symbol": "ETH", "type": "native", "chain": "ethereum"},
    "USDT-USD": {"symbol": "USDT", "contract": "0xdac17f958d2ee523a2206206994597c13d831ec7", "chain": "ethereum"},
    "BNB-USD": {"symbol": "BNB", "type": "native", "chain": "binance"},
    "SOL-USD": {"symbol": "SOL", "type": "native", "chain": "solana"},
    "USDC-USD": {"symbol": "USDC", "contract": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "chain": "ethereum"},
    "XRP-USD": {"symbol": "XRP", "type": "native", "chain": "ripple"},
    "ADA-USD": {"symbol": "ADA", "type": "native", "chain": "cardano"},
    "AVAX-USD": {"symbol": "AVAX", "type": "native", "chain": "avalanche"},
    "DOGE-USD": {"symbol": "DOGE", "type": "native", "chain": "dogecoin"},

    # === DEFI TOKENS ===
    "UNI-USD": {"symbol": "UNI", "contract": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", "chain": "ethereum"},
    "LINK-USD": {"symbol": "LINK", "contract": "0x514910771af9ca656af840dff83e8264ecf986ca", "chain": "ethereum"},
    "AAVE-USD": {"symbol": "AAVE", "contract": "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", "chain": "ethereum"},
    "CRV-USD": {"symbol": "CRV", "contract": "0xd533a949740bb3306d119cc777fa900ba034cd52", "chain": "ethereum"},
    "MKR-USD": {"symbol": "MKR", "contract": "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2", "chain": "ethereum"},
    "SUSHI-USD": {"symbol": "SUSHI", "contract": "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2", "chain": "ethereum"},
    "COMP-USD": {"symbol": "COMP", "contract": "0xc00e94cb662c3520282e6f5717214004a7f26888", "chain": "ethereum"},
    "SNX-USD": {"symbol": "SNX", "contract": "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f", "chain": "ethereum"},
    "YFI-USD": {"symbol": "YFI", "contract": "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e", "chain": "ethereum"},
    "1INCH-USD": {"symbol": "1INCH", "contract": "0x111111111117dc0aa78b770fa6a738034120c302", "chain": "ethereum"},

    # === LAYER 1 BLOCKCHAINS ===
    "DOT-USD": {"symbol": "DOT", "type": "native", "chain": "polkadot"},
    "MATIC-USD": {"symbol": "MATIC", "contract": "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0", "chain": "ethereum"},
    "ATOM-USD": {"symbol": "ATOM", "type": "native", "chain": "cosmos"},
    "NEAR-USD": {"symbol": "NEAR", "type": "native", "chain": "near"},
    "FTM-USD": {"symbol": "FTM", "type": "native", "chain": "fantom"},
    "ALGO-USD": {"symbol": "ALGO", "type": "native", "chain": "algorand"},
    "TRX-USD": {"symbol": "TRX", "type": "native", "chain": "tron"},
    "VET-USD": {"symbol": "VET", "type": "native", "chain": "vechain"},
    "ICP-USD": {"symbol": "ICP", "type": "native", "chain": "internet_computer"},
    "FIL-USD": {"symbol": "FIL", "type": "native", "chain": "filecoin"},

    # === LAYER 2 & SCALING ===
    "ARB-USD": {"symbol": "ARB", "type": "native", "chain": "arbitrum"},
    "OP-USD": {"symbol": "OP", "type": "native", "chain": "optimism"},
    "IMX-USD": {"symbol": "IMX", "contract": "0xf57e7e7c23978c3caec3c3548e3d615c346e79ff", "chain": "ethereum"},
    "LRC-USD": {"symbol": "LRC", "contract": "0xbbbbca6a901c926f240b89eacb641d8aec7aeafd", "chain": "ethereum"},

    # === MEME COINS ===
    "SHIB-USD": {"symbol": "SHIB", "contract": "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", "chain": "ethereum"},
    "PEPE-USD": {"symbol": "PEPE", "contract": "0x6982508145454ce325ddbe47a25d4ec3d2311933", "chain": "ethereum"},
    "FLOKI-USD": {"symbol": "FLOKI", "contract": "0xcf0c122c6b73ff809c693db761e7baebe62b6a2e", "chain": "ethereum"},
    "BONK-USD": {"symbol": "BONK", "type": "native", "chain": "solana"},

    # === EXCHANGE TOKENS ===
    "CRO-USD": {"symbol": "CRO", "type": "native", "chain": "cronos"},
    "LEO-USD": {"symbol": "LEO", "contract": "0x2af5d2ad76741191d15dfe7bf6ac92d4bd912ca3", "chain": "ethereum"},
    "OKB-USD": {"symbol": "OKB", "contract": "0x75231f58b43240c9718dd58b4967c5114342a86c", "chain": "ethereum"},

    # === ORACLE & DATA ===
    "GRT-USD": {"symbol": "GRT", "contract": "0xc944e90c64b2c07662a292be6244bdf05cda44a7", "chain": "ethereum"},
    "BAND-USD": {"symbol": "BAND", "contract": "0xba11d00c5f74255f56a5e366f4f77f5a186d7f55", "chain": "ethereum"},

    # === NFT & GAMING ===
    "AXS-USD": {"symbol": "AXS", "contract": "0xbb0e17ef65f82ab018d8edd776e8dd940327b28b", "chain": "ethereum"},
    "SAND-USD": {"symbol": "SAND", "contract": "0x3845badade8e6dff049820680d1f14bd3903a5d0", "chain": "ethereum"},
    "MANA-USD": {"symbol": "MANA", "contract": "0x0f5d2fb29fb7d3cfee444a200298f468908cc942", "chain": "ethereum"},
    "ENJ-USD": {"symbol": "ENJ", "contract": "0xf629cbd94d3791c9250152bd8dfbdf380e2a3b9c", "chain": "ethereum"},
    "GALA-USD": {"symbol": "GALA", "contract": "0xd1d2eb1b1e90b638588728b4130137d262c87cae", "chain": "ethereum"},
    "APE-USD": {"symbol": "APE", "contract": "0x4d224452801aced8b2f0aebe155379bb5d594381", "chain": "ethereum"},

    # === PRIVACY COINS ===
    "XMR-USD": {"symbol": "XMR", "type": "native", "chain": "monero"},
    "ZEC-USD": {"symbol": "ZEC", "type": "native", "chain": "zcash"},

    # === STABLECOINS (wrapped versions) ===
    "DAI-USD": {"symbol": "DAI", "contract": "0x6b175474e89094c44da98b954eedeac495271d0f", "chain": "ethereum"},
    "BUSD-USD": {"symbol": "BUSD", "contract": "0x4fabb145d64652a948d72533023f6e7a623c7c53", "chain": "ethereum"},
    "TUSD-USD": {"symbol": "TUSD", "contract": "0x0000000000085d4780b73119b644ae5ecd22b376", "chain": "ethereum"},

    # === INFRASTRUCTURE ===
    "RNDR-USD": {"symbol": "RNDR", "contract": "0x6de037ef9ad2725eb40118bb1702ebb27e4aeb24", "chain": "ethereum"},
    "STX-USD": {"symbol": "STX", "type": "native", "chain": "stacks"},
    "INJ-USD": {"symbol": "INJ", "type": "native", "chain": "injective"},

    # === WRAPPED TOKENS ===
    "WBTC-USD": {"symbol": "WBTC", "contract": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", "chain": "ethereum"},
    "WETH-USD": {"symbol": "WETH", "contract": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "chain": "ethereum"},
}


# ==========================================
# ACTUALLY FETCH REAL DATA
# ==========================================

def fetch_dexscreener_data(symbol):
    """Get REAL liquidity data from DexScreener API"""
    try:
        print(f"   🔍 DexScreener: Searching for {symbol}...")
        url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
        response = requests.get(url, timeout=10).json()

        pairs = response.get('pairs', [])
        if not pairs:
            print(f"   ❌ No pairs found for {symbol}")
            return None

        # Get highest liquidity pair
        best_pair = sorted(pairs, key=lambda x: x.get('liquidity', {}).get('usd', 0), reverse=True)[0]

        liquidity = float(best_pair.get('liquidity', {}).get('usd', 0))
        mcap = float(best_pair.get('fdv', 0))

        print(f"   ✅ Found: Liquidity=${liquidity:,.0f}, MCap=${mcap:,.0f}")

        return {
            'liquidity_usd': liquidity,
            'market_cap_usd': mcap,
            'price_usd': float(best_pair.get('priceUsd', 0)),
            'dex': best_pair.get('dexId', 'unknown')
        }
    except Exception as e:
        print(f"   ⚠️ DexScreener failed: {e}")
        return None


def fetch_etherscan_holders_REAL(contract_address, limit=100):
    """
    ACTUALLY fetch top holders from Etherscan
    Calculate REAL concentration metrics
    """
    try:
        print(f"   🔍 Etherscan: Fetching top {limit} holders...")

        url = "https://api.etherscan.io/api"
        params = {
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": contract_address,
            "page": 1,
            "offset": limit,
            "apikey": ETHERSCAN_API_KEY
        }

        response = requests.get(url, params=params, timeout=15).json()

        if response.get('status') != '1':
            print(f"   ⚠️ Etherscan error: {response.get('message', 'Unknown')}")
            return None

        holders = response.get('result', [])

        if not holders or isinstance(holders, str):
            print(f"   ⚠️ No holder data (might be rate limited)")
            return None

        # Calculate REAL balances
        balances = []
        total_supply = 0

        for h in holders:
            try:
                balance = int(h['TokenHolderQuantity'])
                balances.append(balance)
                total_supply += balance
            except:
                continue

        if total_supply == 0 or len(balances) == 0:
            print(f"   ⚠️ Invalid holder data")
            return None

        # REAL concentration metrics
        top1_pct = balances[0] / total_supply
        top5_pct = sum(balances[:5]) / total_supply
        top10_pct = sum(balances[:10]) / total_supply
        top20_pct = sum(balances[:20]) / total_supply

        print(f"   ✅ Holders: Top1={top1_pct:.1%}, Top10={top10_pct:.1%}, Top20={top20_pct:.1%}")

        return {
            'top1_concentration': top1_pct,
            'top5_concentration': top5_pct,
            'top10_concentration': top10_pct,
            'top20_concentration': top20_pct,
            'num_holders': len(holders),
            'total_analyzed': total_supply
        }

    except Exception as e:
        print(f"   ⚠️ Etherscan failed: {e}")
        return None


def calculate_sybil_score_from_real_data(holder_data, liquidity_usd, market_cap_usd):
    """
    Calculate Sybil score from ACTUAL data, not made-up tiers

    Factors:
    1. Top holder concentration (40% weight)
    2. Top 10 concentration (30% weight)
    3. Liquidity/MCap ratio (20% weight)
    4. Number of holders (10% weight)
    """

    if holder_data is None:
        print("   ⚠️ No holder data - using liquidity-based estimate")
        # Fallback: use liquidity ratio
        if market_cap_usd > 0:
            liq_ratio = liquidity_usd / market_cap_usd
            # Low liquidity = high risk
            sybil_score = max(0.3, min(0.8, 1 - (liq_ratio * 10)))
        else:
            sybil_score = 0.5

        return sybil_score, "estimated_from_liquidity"

    # ACTUAL calculation from holder data
    top1 = holder_data['top1_concentration']
    top10 = holder_data['top10_concentration']
    top20 = holder_data['top20_concentration']

    # Component 1: Top holder (if >50% = instant high risk)
    if top1 > 0.5:
        top1_score = 0.9
    elif top1 > 0.3:
        top1_score = 0.7
    elif top1 > 0.15:
        top1_score = 0.5
    elif top1 > 0.05:
        top1_score = 0.3
    else:
        top1_score = 0.2

    # Component 2: Top 10 holders
    if top10 > 0.8:
        top10_score = 0.9
    elif top10 > 0.6:
        top10_score = 0.7
    elif top10 > 0.4:
        top10_score = 0.5
    else:
        top10_score = 0.3

    # Component 3: Liquidity ratio
    if market_cap_usd > 0:
        liq_ratio = liquidity_usd / market_cap_usd
        if liq_ratio < 0.01:  # < 1% liquidity = very risky
            liq_score = 0.8
        elif liq_ratio < 0.05:
            liq_score = 0.5
        else:
            liq_score = 0.2
    else:
        liq_score = 0.5

    # Component 4: Holder count (more holders = less risk)
    num_holders = holder_data.get('num_holders', 0)
    if num_holders < 10000:
        holder_score = 0.8
    elif num_holders < 100000:
        holder_score = 0.5
    else:
        holder_score = 0.2

    # Weighted average
    sybil_score = (
            0.40 * top1_score +
            0.30 * top10_score +
            0.20 * liq_score +
            0.10 * holder_score
    )

    print(
        f"   📊 Sybil Components: Top1={top1_score:.2f}, Top10={top10_score:.2f}, Liq={liq_score:.2f}, Holders={holder_score:.2f}")
    print(f"   🎯 Final Sybil Score: {sybil_score:.3f}")

    return sybil_score, "calculated_from_real_data"


def auto_fetch_jump_params(ticker):
    """
    ACTUALLY FETCH REAL DATA - No more fake tiers!
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 FETCHING REAL DATA FOR {ticker}")
    print(f"{'=' * 60}")

    if ticker not in TICKER_TO_BLOCKCHAIN:
        print(f"⚠️ {ticker} not in database")
        return None

    token_info = TICKER_TO_BLOCKCHAIN[ticker]
    symbol = token_info['symbol']

    # Step 1: Get liquidity from DexScreener
    print(f"\n📊 Step 1: Fetching liquidity data...")
    dex_data = fetch_dexscreener_data(symbol)

    if dex_data:
        liquidity_usd = dex_data['liquidity_usd']
        market_cap_usd = dex_data['market_cap_usd']
    else:
        print(f"   ⚠️ Using fallback values")
        liquidity_usd = 500_000
        market_cap_usd = 10_000_000

    # Step 2: Get REAL holder data (if ERC-20)
    holder_data = None

    if 'contract' in token_info and token_info['chain'] == 'ethereum':
        print(f"\n👥 Step 2: Fetching REAL holder data from Etherscan...")
        time.sleep(0.3)  # Rate limit respect
        holder_data = fetch_etherscan_holders_REAL(token_info['contract'], limit=100)
    else:
        print(f"\n👥 Step 2: Native coin - skipping holder analysis")

    # Step 3: Calculate REAL Sybil score
    print(f"\n🧮 Step 3: Calculating Sybil score from real data...")
    sybil_score, method = calculate_sybil_score_from_real_data(holder_data, liquidity_usd, market_cap_usd)

    # Step 4: Calculate whale wealth
    if holder_data:
        top1_pct = holder_data['top1_concentration']
        sybil_wealth = market_cap_usd * top1_pct
        print(f"   💰 Top holder controls: {top1_pct:.1%} = ${sybil_wealth:,.0f}")
    else:
        # Estimate based on Sybil score
        estimated_concentration = sybil_score * 0.4  # Rough estimate
        sybil_wealth = market_cap_usd * estimated_concentration
        print(f"   💰 Estimated whale wealth: ${sybil_wealth:,.0f}")

    print(f"\n{'=' * 60}")
    print(f"✅ DATA COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Sybil Score: {sybil_score:.3f} ({method})")
    print(f"Whale Wealth: ${sybil_wealth:,.0f}")
    print(f"Liquidity: ${liquidity_usd:,.0f}")
    print(f"Market Cap: ${market_cap_usd:,.0f}")
    print(f"{'=' * 60}\n")

    return {
        'sybil_score': sybil_score,
        'sybil_wealth': sybil_wealth,
        'liquidity_usd': liquidity_usd,
        'market_cap_usd': market_cap_usd,
        'holder_data': holder_data,
        'method': method
    }


# ==========================================
# 1. DRIFT WITH MEAN REVERSION (1 Year Daily Data)
# ==========================================

def estimate_drift_mean_reversion(token_ticker, btc_ticker="BTC-USD", days=1095, theta=0.3):
    """Estimate drift with mean reversion using 1 year of DAILY data"""
    print(f"📊 Estimating drift with mean reversion (1 year daily data)...")

    token_data = yf.Ticker(token_ticker).history(period=f"{days}d", interval="1d")
    btc_data = yf.Ticker(btc_ticker).history(period=f"{days}d", interval="1d")

    token_px = token_data["Close"]
    btc_px = btc_data["Close"]

    df = token_px.to_frame("token").join(btc_px.to_frame("btc"), how="inner").dropna()

    if len(df) < 100:
        raise ValueError(f"Insufficient data for {token_ticker}")

    r_token = np.log(df["token"]).diff().dropna().values
    r_btc = np.log(df["btc"]).diff().dropna().values

    n = min(len(r_token), len(r_btc))
    r_token, r_btc = r_token[-n:], r_btc[-n:]

    # Calculate BTC beta
    X = sm.add_constant(r_btc)
    beta_hat = sm.OLS(r_token, X).fit().params[1]

    # Long-term vs recent
    long_term_mean = np.mean(r_token)
    recent_ret = np.mean(r_token[-30:])

    # Mean reversion
    mean_reversion_component = theta * (long_term_mean - recent_ret)

    # BTC momentum
    btc_momentum = np.mean(r_btc[-7:])

    # Total drift
    mu_daily = mean_reversion_component + beta_hat * btc_momentum

    print(f"   Data points:         {len(r_token)} days")
    print(f"   Long-term mean:      {long_term_mean:.6f} per day")
    print(f"   Recent (30d):        {recent_ret:.6f} per day")
    print(f"   Mean reversion adj:  {mean_reversion_component:.6f}")
    print(f"   BTC beta:            {beta_hat:.4f}")
    print(f"   → Total drift:       {mu_daily:.6f} per day")

    return mu_daily, beta_hat


# ==========================================
# 2. VOLATILITY ESTIMATION (1 Year Daily Data)
# ==========================================

def estimate_volatility_garch(token_ticker, days=1095):
    """Estimate volatility using GARCH(1,1) on 1 year of DAILY data"""
    print(f"📈 Estimating GARCH volatility (1 year daily data)...")

    token = yf.Ticker(token_ticker)
    data = token.history(period=f"{days}d", interval="1d")

    if len(data) < 100:
        raise ValueError(f"Insufficient data for {token_ticker}")

    prices = data["Close"].values
    returns = np.diff(np.log(prices)) * 100

    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
    fitted = model.fit(disp='off')

    forecast = fitted.forecast(horizon=1)
    sigma_daily = np.sqrt(forecast.variance.values[-1, 0]) / 100

    print(f"   Data points:         {len(returns)} days")
    print(f"   σ_GARCH (daily):     {sigma_daily:.6f}")

    return sigma_daily


# ==========================================
# 3. JUMP PARAMETERS (Merton Model)
# ==========================================

def calculate_merton_jump_params(
    sybil_score, sybil_wealth, liquidity_usd, btc_volatility,
    delta_doxxed=1.0, delta_history=1.0, delta_social=0.5,
    sensitivity_r=1.0, convexity_alpha=2.0, lambda_base=2.0,
    lambda_max=10.0, l_saturation=500_000, beta=1.5, k_cap=1.0
):
    """Calculate Merton jump-diffusion parameters"""
    print(f"🎲 Calculating jump parameters...")

    p = sybil_wealth / (liquidity_usd + 1.0)
    denom = (1 + sensitivity_r * p) ** convexity_alpha
    phi_impact = sybil_score * (1 - (1 / denom))
    phi_impact = min(phi_impact, 0.999)

    mu_j = np.log(1 - phi_impact)
    delta_j = 0.2 + (0.5 * (1 - sybil_score))

    m_rep = delta_doxxed * delta_history * delta_social
    tanh_term = np.tanh((2.65 * liquidity_usd) / l_saturation)
    r_liq = 1 - tanh_term
    market_impact = beta * abs(btc_volatility)
    r_mkt = 1 + min(k_cap, market_impact)

    lambda_total = ((lambda_base * m_rep) + (lambda_max * r_liq)) * r_mkt
    lambda_daily = lambda_total / 1095

    print(f"   Jump intensity (λ):  {lambda_daily:.6f} per day")
    print(f"   Jump mean (μ_j):     {mu_j:.6f}")
    print(f"   Jump vol (σ_j):      {delta_j:.6f}")

    return lambda_daily, mu_j, delta_j


# ==========================================
# 4. MONTE CARLO SIMULATION
# ==========================================

def monte_carlo_sde(S0, mu, sigma, lambda_j, mu_j, sigma_j, T=7, dt=1, n_sims=10000):
    """Monte Carlo simulation of complete SDE"""
    print(f"🔄 Running Monte Carlo: {n_sims} simulations over {T} days...")

    n_steps = int(T / dt)
    S = np.zeros((n_sims, n_steps + 1))
    S[:, 0] = S0

    for i in range(1, n_steps + 1):
        dW = np.random.randn(n_sims) * np.sqrt(dt)
        dS = mu * S[:, i-1] * dt + sigma * S[:, i-1] * dW

        n_jumps = np.random.poisson(lambda_j * dt, n_sims)
        jump_component = np.zeros(n_sims)

        for sim in range(n_sims):
            if n_jumps[sim] > 0:
                for _ in range(n_jumps[sim]):
                    Y = mu_j + sigma_j * np.random.randn()
                    jump_component[sim] += (np.exp(Y) - 1)

        S[:, i] = S[:, i-1] + dS + S[:, i-1] * jump_component
        S[:, i] = np.maximum(S[:, i], 0.01)

    return S


# ==========================================
# 5. RISK METRICS
# ==========================================

def calculate_risk_metrics(S, S0, confidence=0.95):
    """Calculate VaR and Expected Shortfall"""
    terminal_prices = S[:, -1]
    returns = (terminal_prices - S0) / S0
    sorted_returns = np.sort(returns)

    var_index = int((1 - confidence) * len(sorted_returns))
    var_95 = sorted_returns[var_index]
    es_95 = np.mean(sorted_returns[:var_index])

    return {
        'var_95': var_95,
        'es_95': es_95,
        'mean_return': np.mean(returns),
        'median_return': np.median(returns),
        'prob_profit': np.mean(terminal_prices > S0),
        'mean_final_price': np.mean(terminal_prices),
        'terminal_prices': terminal_prices
    }

# ==========================================
# 6. VISUALIZATION (PRETTIER, SAME 2-PANEL LAYOUT)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def _money(x, pos=None):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"


def visualize_results(S, S0, risk_metrics, token_ticker, T):
    """
    Same outputs as your original (paths + terminal distribution),
    but cleaner styling:
      - fan chart (quantile bands) instead of spaghetti emphasis
      - mean + median paths
      - clearer VaR/ES annotations
      - nicer formatting/spacing
    """

    # ---- basic prep
    n_sims, n_steps = S.shape
    t = np.linspace(0, T, n_steps)
    terminal = risk_metrics["terminal_prices"]

    # quantiles for fan chart
    q1, q5, q25, q50, q75, q95, q99 = np.percentile(S, [1, 5, 25, 50, 75, 95, 99], axis=0)
    mean_path = np.mean(S, axis=0)

    # prices for VaR/ES lines
    var_price = S0 * (1 + risk_metrics["var_95"])
    es_price = S0 * (1 + risk_metrics["es_95"])
    mean_final = risk_metrics["mean_final_price"]
    median_final = float(np.median(terminal))

    # ---- style
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#111111",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)

    # =========================
    # LEFT: fan chart + a few paths
    # =========================
    ax1 = axes[0]

    # plot only a small handful of paths to add texture (optional)
    n_show = min(100, n_sims)
    idx = np.random.choice(n_sims, size=n_show, replace=False)
    for i in idx:
        ax1.plot(t, S[i], alpha=0.08, linewidth=0.8)

    # fan chart (wide to narrow)
    ax1.fill_between(t, q1,  q99, alpha=0.10, label="1–99% band")
    ax1.fill_between(t, q5,  q95, alpha=0.16, label="5–95% band")
    ax1.fill_between(t, q25, q75, alpha=0.24, label="25–75% band")

    # central tendency
    ax1.plot(t, q50, linewidth=2.6, label="Median")
    ax1.plot(t, mean_path, linewidth=2.6, linestyle="--", label="Mean")

    # start line
    ax1.axhline(S0, linestyle=":", linewidth=2.0, label=f"Start = ${S0:.2f}")

    ax1.set_title(f"{token_ticker} simulated paths (T={T}d)")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    ax1.yaxis.set_major_formatter(FuncFormatter(_money))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.legend(loc="upper left", frameon=True)

    # =========================
    # RIGHT: terminal distribution with VaR/ES
    # =========================
    ax2 = axes[1]

    ax2.hist(terminal, bins=60, density=True, alpha=0.80, edgecolor="#111111", linewidth=0.6)

    ax2.axvline(S0, linestyle=":", linewidth=2.0, label=f"Start = ${S0:.2f}")
    ax2.axvline(mean_final, linestyle="--", linewidth=2.2, label=f"Mean = ${mean_final:.2f}")
    ax2.axvline(median_final, linestyle="-.", linewidth=2.2, label=f"Median = ${median_final:.2f}")

    ax2.axvline(var_price, linestyle="--", linewidth=2.2, label=f"VaR 95% = ${var_price:.2f}")
    ax2.axvline(es_price,  linestyle="--", linewidth=2.2, label=f"ES 95% = ${es_price:.2f}")

    ax2.set_title(f"Terminal distribution (T={T}d)")
    ax2.set_xlabel("Terminal price")
    ax2.set_ylabel("Density")
    ax2.grid(True)

    ax2.xaxis.set_major_formatter(FuncFormatter(_money))
    ax2.legend(loc="upper right", frameon=True)

    # ---- overall title (nice for demos)
    prob_profit = float(risk_metrics.get("prob_profit", np.mean(terminal > S0)))
    mean_ret = float(risk_metrics.get("mean_return", np.mean((terminal - S0) / S0)))
    median_ret = float(risk_metrics.get("median_return", np.median((terminal - S0) / S0)))

    fig.suptitle(
        f"{token_ticker} | Mean return: {mean_ret*100:.2f}% | Median: {median_ret*100:.2f}% | P(profit): {prob_profit*100:.1f}%",
        fontsize=13,
        fontweight="bold"
    )

    out = f"{token_ticker}_complete_sde_model.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n✅ Saved: {out}")

# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def run_complete_sde_model_auto(token_ticker, T=7, n_sims=10000, theta=0.3):
    """Complete pipeline with mean reversion and 1 year daily data"""
    print(f"\n{'='*70}")
    print(f"COMPLETE SDE MODEL (50+ COINS SUPPORTED)")
    print(f"{token_ticker}")
    print(f"{'='*70}\n")

    S0 = yf.Ticker(token_ticker).history(period="1d")['Close'].iloc[-1]
    print(f"Current Price: ${S0:.2f}\n")

    jump_data = auto_fetch_jump_params(token_ticker)

    if jump_data is None:
        print("\n⚠️ Using default jump parameters")
        sybil_score = 0.5
        sybil_wealth = 1_000_000
        liquidity_usd = 500_000
    else:
        sybil_score = jump_data['sybil_score']
        sybil_wealth = jump_data['sybil_wealth']
        liquidity_usd = jump_data['liquidity_usd']

    mu, beta = estimate_drift_mean_reversion(token_ticker, days=1095, theta=theta)
    sigma = estimate_volatility_garch(token_ticker, days=1095)

    btc_vol = estimate_volatility_garch("BTC-USD", days=1095)
    lambda_j, mu_j, sigma_j = calculate_merton_jump_params(
        sybil_score=sybil_score, sybil_wealth=sybil_wealth,
        liquidity_usd=liquidity_usd, btc_volatility=btc_vol
    )

    print(f"\n{'='*70}")
    print(f"SDE PARAMETERS")
    print(f"{'='*70}")
    print(f"Drift (μ):              {mu:.6f} per day")
    print(f"Volatility (σ_GARCH):   {sigma:.6f} per day")
    print(f"Jump intensity (λ):     {lambda_j:.6f} per day")
    print(f"Jump mean (μ_j):        {mu_j:.6f}")
    print(f"Jump vol (σ_j):         {sigma_j:.6f}")
    print(f"BTC Beta:               {beta:.4f}")
    print(f"Sybil Score:            {sybil_score:.2f}")
    print(f"\nComplete SDE:")
    print(f"dS_t = {mu:.4f}S_t dt + {sigma:.4f}S_t dW_t + S_t(e^Y - 1)dN_t")
    print(f"{'='*70}\n")

    S = monte_carlo_sde(S0, mu, sigma, lambda_j, mu_j, sigma_j, T=T, n_sims=n_sims)
    risk_metrics = calculate_risk_metrics(S, S0)

    print(f"\n{'='*70}")
    print(f"RISK METRICS")
    print(f"{'='*70}")
    print(f"Mean Return:            {risk_metrics['mean_return']:+.2%}")
    print(f"Probability Profit:     {risk_metrics['prob_profit']:.1%}")
    print(f"VaR (95%):              {risk_metrics['var_95']:+.2%}")
    print(f"Expected Shortfall:     {risk_metrics['es_95']:+.2%}")
    print(f"Expected Price (7d):    ${risk_metrics['mean_final_price']:.2f}")
    print(f"{'='*70}\n")

    visualize_results(S, S0, risk_metrics, token_ticker, T)

    return {'mu': mu, 'sigma': sigma, 'lambda': lambda_j, 'risk_metrics': risk_metrics, 'paths': S}


# ==========================================
# 8. USAGE
# ==========================================

if __name__ == "__main__":
    print("\n🎯 Supported Coins (50+):")
    print("Top 10: BTC, ETH, USDT, BNB, SOL, USDC, XRP, ADA, AVAX, DOGE")
    print("DeFi: UNI, LINK, AAVE, CRV, MKR, SUSHI, COMP, SNX, YFI, 1INCH")
    print("L1s: DOT, MATIC, ATOM, NEAR, FTM, ALGO, TRX, VET, ICP, FIL")
    print("L2s: ARB, OP, IMX, LRC")
    print("Meme: SHIB, PEPE, FLOKI, BONK")
    print("NFT/Gaming: AXS, SAND, MANA, ENJ, GALA, APE")
    print("And more!\n")

    user_token = input("Enter crypto ticker (e.g., BTC-USD, SOL-USD, PEPE-USD): ").strip().upper()

    if '-USD' not in user_token:
        user_token = f"{user_token}-USD"

    theta_input = input("Mean reversion speed θ (0-1) [default 0.3]: ").strip()
    theta = float(theta_input) if theta_input else 0.3

    try:
        results = run_complete_sde_model_auto(token_ticker=user_token, T=7, n_sims=10000, theta=theta)
        print("\n✅ Analysis Complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")






