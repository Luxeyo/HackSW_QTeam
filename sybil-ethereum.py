### Sybil attack functions
import numpy as np
import pandas as pd

'''
Severe Sybil risk is modeled as a nonlinear coordination stress index that induces regime-dependent jump intensification 
rather than continuous price distortion.

You have, or can compute, the following normalized inputs in
[0,1] for the top-K non-contract wallets:

dormant_ratio — fraction of wallets inactive for ≥ T days
funding_concentration — Herfindahl-style concentration of funding sources
tx_sync_score — synchrony of transaction timestamps
clustered_ownership — % of supply controlled by clustered wallets

If you cannot normalize these, your upstream data is not ready.

This breaks if:

Wallet clustering is naive
Contract wallets aren’t filtered
Dormancy windows are too short
Sybil score is updated intraday

All four will cause false positives.
'''

### Sybil fetch functions

'''
Critical Warnings (Do Not Ignore)

Etherscan cannot give true wallet clustering — this is proxy-level only
Contract wallets must be filtered upstream
Rate limits will break naive loops
This is Ethereum-only; Solana requires a different pipeline

'''

import requests
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

ETHERSCAN_API_KEY = "C9SI7APBI8TC9PE8C6P7UHI7Z9K6BYWU46"
BASE_URL = "https://api.etherscan.io/api"

# User-facing → ERC-20 mapping
TOKEN_MAP = {
    "BTC": "WBTC",
    "ETH": "WETH",
    "USDC": "USDC",
    "USDT": "USDT"
}

# ERC-20 contract addresses (examples — extend as needed)
TOKEN_CONTRACTS = {
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "WETH": "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7"
}

'''
5. Why This Is Acceptable (And Where It Fails) (clusters)

✔ Captures coordinated funding
✔ Conservative (underestimates control)
✔ Uses observable data

❌ Misses post-funding consolidation
❌ Misses mixers / CEX hops
❌ Cannot detect DAO multisigs

This is fine for a risk advisor — not for attribution.
'''

def funding_clusters(addresses):
    """
    Groups wallets sharing the same initial funder.
    """
    clusters = defaultdict(list)

    for addr in addresses:
        params = {
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": ETHERSCAN_API_KEY
        }
        r = requests.get(BASE_URL, params=params).json()
        txs = r.get("result", [])
        if txs:
            funder = txs[0]["from"].lower()
            clusters[funder].append(addr)

        time.sleep(0.2)

    return clusters

def cluster_ownership_share(contract_address, clusters):
    """
    Returns max cluster ownership share of total circulating supply.
    """
    balances = {}
    total_supply = 0

    for funder, wallets in clusters.items():
        cluster_balance = 0
        for wallet in wallets:
            params = {
                "module": "account",
                "action": "tokenbalance",
                "contractaddress": contract_address,
                "address": wallet,
                "tag": "latest",
                "apikey": ETHERSCAN_API_KEY
            }
            r = requests.get(BASE_URL, params=params).json()
            bal = int(r.get("result", 0))
            cluster_balance += bal
            total_supply += bal

            time.sleep(0.2)

        balances[funder] = cluster_balance

    if total_supply == 0:
        return 0.0

    return max(balances.values()) / total_supply

def get_top_holders(contract_address, limit=50):
    """
    Returns top ERC-20 holders with balances.
    """
    params = {
        "module": "token",
        "action": "tokenholderlist",
        "contractaddress": contract_address,
        "page": 1,
        "offset": limit,
        "apikey": ETHERSCAN_API_KEY
    }
    r = requests.get(BASE_URL, params=params).json()
    return r.get("result", [])

def dormant_wallet_ratio(addresses, dormant_days=30):
    """
    Fraction of wallets inactive for >= dormant_days.
    """
    now = datetime.now(timezone.utc)
    dormant = 0

    for addr in addresses:
        params = {
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": ETHERSCAN_API_KEY
        }
        r = requests.get(BASE_URL, params=params).json()
        txs = r.get("result", [])

        if not txs:
            dormant += 1
        else:
            last_ts = datetime.fromtimestamp(int(txs[0]["timeStamp"]), tz=timezone.utc)
            if (now - last_ts).days >= dormant_days:
                dormant += 1

        time.sleep(0.2)  # rate limit safety

    return dormant / len(addresses)

def funding_source_concentration(addresses):
    """
    Herfindahl-style concentration of initial funding sources.
    """
    sources = []

    for addr in addresses:
        params = {
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": ETHERSCAN_API_KEY
        }
        r = requests.get(BASE_URL, params=params).json()
        txs = r.get("result", [])
        if txs:
            sources.append(txs[0]["from"].lower())

        time.sleep(0.2)

    counts = Counter(sources)
    total = sum(counts.values())

    hhi = sum((c / total) ** 2 for c in counts.values())
    return hhi  # normalize later

def transaction_sync_score(addresses, window_seconds=300):
    """
    Measures how often wallets transact within the same short time window.
    """
    timestamps = []

    for addr in addresses:
        params = {
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": ETHERSCAN_API_KEY
        }
        r = requests.get(BASE_URL, params=params).json()
        txs = r.get("result", [])
        if txs:
            timestamps.append(int(txs[0]["timeStamp"]))

        time.sleep(0.2)

    timestamps.sort()
    if len(timestamps) < 2:
        return 0.0

    clustered = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] <= window_seconds:
            clustered += 1

    return clustered / (len(timestamps) - 1)

def severe_sybil_score(
    dormant_ratio: float,
    funding_concentration: float,
    tx_sync_score: float,
    clustered_ownership: float,
    weights=None
) -> float:
    """
    Computes severe Sybil stress score in [0,1]

    Uses nonlinear convex aggregation so that
    any single extreme signal can dominate.
    """

    # Default equal weights
    if weights is None:
        weights = {
            "dormant": 1.0,
            "funding": 1.0,
            "sync": 1.0,
            "ownership": 1.0
        }

    scores = {
        "dormant": np.clip(dormant_ratio, 0, 1),
        "funding": np.clip(funding_concentration, 0, 1),
        "sync": np.clip(tx_sync_score, 0, 1),
        "ownership": np.clip(clustered_ownership, 0, 1)
    }

    product_term = 1.0
    for k, s in scores.items():
        product_term *= (1.0 - s) ** weights[k]

    return 1.0 - product_term

def sybil_attack_regime(
    sybil_score: float,
    threshold: float = 0.65
) -> int:
    """
    Binary severe Sybil attack regime indicator
    """
    return int(sybil_score >= threshold)

def sybil_adjusted_jump_intensity(
    base_lambda: float,
    sybil_score: float,
    attack_regime: int,
    kappa_continuous: float = 1.5,
    kappa_regime: float = 2.5,
    lambda_cap: float = 10.0
) -> float:
    """
    State-dependent Poisson jump intensity
    """

    lam = base_lambda * np.exp(
        kappa_continuous * sybil_score +
        kappa_regime * attack_regime
    )

    # Hard cap to prevent numerical explosion
    return min(lam, lambda_cap)

def sybil_volatility_multiplier(
    sybil_score: float,
    eta: float = 0.3
) -> float:
    """
    Mild volatility scaling due to coordination risk
    """
    return 1.0 + eta * sybil_score

def historical_drawdown(price, horizon, percentile):
    df = pd.DataFrame(price, columns=['close'])
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    rolling_log_returns = log_returns.rolling(horizon).sum()
    rolling_simple_returns = np.exp(rolling_log_returns) - 1
    historical_VaR = np.percentile(rolling_simple_returns.dropna(), percentile)
    return historical_VaR

print(historical_drawdown([100, 80, 70], 1, 5))

token = TOKEN_MAP.get("BTC", "BTC")
contract = TOKEN_CONTRACTS[token]

holders = get_top_holders(contract, limit=30)
addresses = [h["HolderAddress"] for h in holders]

# Example structural metrics (already normalized)
dormant_ratio = dormant_wallet_ratio(addresses)
funding_concentration = funding_source_concentration(addresses)
tx_sync_score = transaction_sync_score(addresses)
clusters = funding_clusters(addresses)
clustered_ownership = cluster_ownership_share(addresses, clusters)

# Step 1: Sybil stress
S = severe_sybil_score(
    dormant_ratio,
    funding_concentration,
    tx_sync_score,
    clustered_ownership
)

# Step 2: Regime switch
attack_flag = sybil_attack_regime(S)

# Step 3: Jump intensity
lambda_t = sybil_adjusted_jump_intensity(
    base_lambda=0.05,
    sybil_score=S,
    attack_regime=attack_flag
)

print("Sybil score:", S)
print("Attack regime:", attack_flag)
print("Jump intensity:", lambda_t)
