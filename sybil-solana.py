### Sybil attack functions

import numpy as np
import pandas as pd
import requests
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone

SOLANA_RPC = "https://api.mainnet-beta.solana.com"
HEADERS = {"Content-Type": "application/json"}
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

def solana_rpc(method, params):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    r = requests.post(SOLANA_RPC, json=payload, headers=HEADERS).json()
    return r.get("result")

def get_top_spl_holders(mint_address, limit=30):
    """
    Returns top SPL token holders (wallet addresses).
    """
    result = solana_rpc(
        "getTokenLargestAccounts",
        [mint_address]
    )

    accounts = result["value"][:limit]

    holders = []
    for acc in accounts:
        acct_info = solana_rpc(
            "getAccountInfo",
            [acc["address"], {"encoding": "jsonParsed"}]
        )
        owner = acct_info["value"]["data"]["parsed"]["info"]["owner"]
        holders.append({
            "wallet": owner,
            "token_amount": int(acc["amount"])
        })

    return holders

def dormant_wallet_ratio_solana(wallets, dormant_days=30):
    now = datetime.now(timezone.utc)
    dormant = 0

    for w in wallets:
        sigs = solana_rpc(
            "getSignaturesForAddress",
            [w, {"limit": 1}]
        )

        if not sigs:
            dormant += 1
            continue

        ts = sigs[0]["blockTime"]
        last_tx = datetime.fromtimestamp(ts, tz=timezone.utc)

        if (now - last_tx).days >= dormant_days:
            dormant += 1

        time.sleep(0.1)

    return dormant / len(wallets)

def funding_source_concentration_solana(wallets):
    funders = []

    for w in wallets:
        sigs = solana_rpc(
            "getSignaturesForAddress",
            [w, {"limit": 10}]
        )

        if not sigs:
            continue

        # oldest tx = last in list
        oldest_sig = sigs[-1]["signature"]

        tx = solana_rpc(
            "getTransaction",
            [oldest_sig, {"encoding": "jsonParsed"}]
        )

        try:
            instructions = tx["transaction"]["message"]["instructions"]
            for ix in instructions:
                if ix.get("program") == "system" and ix["parsed"]["type"] == "transfer":
                    funders.append(ix["parsed"]["info"]["source"])
                    break
        except:
            pass

        time.sleep(0.1)

    counts = Counter(funders)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    hhi = sum((c / total) ** 2 for c in counts.values())
    return hhi

def transaction_sync_score_solana(wallets, window_seconds=300):
    timestamps = []

    for w in wallets:
        sigs = solana_rpc(
            "getSignaturesForAddress",
            [w, {"limit": 1}]
        )
        if sigs and sigs[0]["blockTime"]:
            timestamps.append(sigs[0]["blockTime"])

        time.sleep(0.1)

    timestamps.sort()

    if len(timestamps) < 2:
        return 0.0

    clustered = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] <= window_seconds:
            clustered += 1

    return clustered / (len(timestamps) - 1)

def funding_clusters_solana(wallets):
    clusters = defaultdict(list)

    for w in wallets:
        sigs = solana_rpc(
            "getSignaturesForAddress",
            [w, {"limit": 10}]
        )
        if not sigs:
            continue

        oldest_sig = sigs[-1]["signature"]
        tx = solana_rpc(
            "getTransaction",
            [oldest_sig, {"encoding": "jsonParsed"}]
        )

        try:
            instructions = tx["transaction"]["message"]["instructions"]
            for ix in instructions:
                if ix.get("program") == "system" and ix["parsed"]["type"] == "transfer":
                    funder = ix["parsed"]["info"]["source"]
                    clusters[funder].append(w)
                    break
        except:
            pass

        time.sleep(0.1)

    return clusters

def cluster_ownership_share_solana(mint_address, clusters):
    cluster_balances = {}
    total_balance = 0

    for funder, wallets in clusters.items():
        bal = 0
        for w in wallets:
            resp = solana_rpc(
                "getParsedTokenAccountsByOwner",
                [w, {"mint": mint_address}, {"encoding": "jsonParsed"}]
            )
            for acct in resp["value"]:
                amt = int(acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
                bal += amt
                total_balance += amt
        cluster_balances[funder] = bal

    if total_balance == 0:
        return 0.0

    return max(cluster_balances.values()) / total_balance

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

MINT = "<SPL_TOKEN_MINT>"

holders = get_top_spl_holders(MINT, limit=30)
wallets = [h["wallet"] for h in holders]

D = dormant_wallet_ratio_solana(wallets)
F = funding_source_concentration_solana(wallets)
T = transaction_sync_score_solana(wallets)

clusters = funding_clusters_solana(wallets)
C = cluster_ownership_share_solana(MINT, clusters)

severe_sybil_score(D, F, T, C)
