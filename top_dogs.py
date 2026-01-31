import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

MORALIS_API_KEY = os.getenv("MORALIS_API_KEY")
if not MORALIS_API_KEY:
    raise SystemExit("Missing MORALIS_API_KEY in .env")

SPECIAL_LIST = {"BTC", "ETH", "SOL"}

CG_BASE = "https://api.coingecko.com/api/v3"

EVM_PLATFORM_TO_MORALIS_CHAIN = {
    "ethereum": "eth",
    "binance-smart-chain": "bsc",
    "polygon-pos": "polygon",
    "arbitrum-one": "arbitrum",
    "optimistic-ethereum": "optimism",
    "base": "base",
    "avalanche": "avalanche",
}

def coingecko_top100():
    r = requests.get(
        f"{CG_BASE}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false",
        },
        timeout=(5, 25),
    )
    r.raise_for_status()
    return r.json()

def coingecko_coin_detail(coin_id: str):
    r = requests.get(
        f"{CG_BASE}/coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false",
        },
        timeout=(5, 25),
    )
    r.raise_for_status()
    return r.json()

def resolve_top100_coin(query: str) -> dict:
    q = query.strip().lower()
    top = coingecko_top100()

    symbol_hits = [c for c in top if (c.get("symbol") or "").lower() == q]
    if len(symbol_hits) == 1:
        return symbol_hits[0]

    name_hits = [c for c in top if (c.get("name") or "").lower() == q]
    if len(name_hits) == 1:
        return name_hits[0]

    contains = [
        c for c in top
        if q in (c.get("name") or "").lower() or q in (c.get("symbol") or "").lower()
    ]
    if len(contains) == 1:
        return contains[0]

    if not (symbol_hits or name_hits or contains):
        raise SystemExit(f"'{query}' not found in CoinGecko top 100.")

    candidates = symbol_hits or name_hits or contains
    print(f"Multiple matches for '{query}':")
    for i, c in enumerate(candidates[:10], start=1):
        print(f"{i}. {c.get('name')} ({(c.get('symbol') or '').upper()})  id={c.get('id')}")
    raise SystemExit("Be more specific (try the symbol or the full name).")

def moralis_solana_top_holders(mint: str, limit: int = 10, network: str = "mainnet"):
    url = f"https://solana-gateway.moralis.io/token/{network}/{mint}/top-holders"
    headers = {"Accept": "application/json", "X-API-Key": MORALIS_API_KEY}
    r = requests.get(url, headers=headers, params={"limit": limit}, timeout=(5, 25))
    r.raise_for_status()
    return r.json()

def moralis_erc20_top_holders(contract: str, chain: str, limit: int = 10):
    url = f"https://deep-index.moralis.io/api/v2.2/erc20/{contract}/owners"
    headers = {"Accept": "application/json", "X-API-Key": MORALIS_API_KEY}
    params = {"chain": chain, "limit": limit, "order": "DESC"}
    r = requests.get(url, headers=headers, params=params, timeout=(5, 25))
    r.raise_for_status()
    return r.json()

def pick_supported_evm_platform(platforms: dict):
    for platform_key, moralis_chain in EVM_PLATFORM_TO_MORALIS_CHAIN.items():
        addr = platforms.get(platform_key)
        if addr:
            return platform_key, moralis_chain, addr
    return None, None, None

def parse_solana_top1_top10(data: dict):
    holders = data.get("result", [])
    if not holders:
        return None, None
    top1 = float(holders[0]["percentageRelativeToTotalSupply"])
    top10 = sum(float(h["percentageRelativeToTotalSupply"]) for h in holders[:10])
    return top1, top10

def parse_evm_top1_top10(data: dict):
    holders = data.get("result", [])
    if not holders:
        return None, None

    def pct(row):
        v = row.get("percentage_relative_to_total_supply")
        return float(v) if v is not None else None

    top1 = pct(holders[0])
    top10_vals = []
    for h in holders[:10]:
        v = pct(h)
        if v is not None:
            top10_vals.append(v)

    if top1 is None or not top10_vals:
        return None, None

    return float(top1), float(sum(top10_vals))

def main():
    query = sys.argv[1].strip() if len(sys.argv) >= 2 else input("Enter a top-100 crypto name/symbol: ").strip()
    if not query:
        raise SystemExit("No input provided")

    if query.strip().upper() in SPECIAL_LIST:
        print("list")
        return

    coin = resolve_top100_coin(query)
    coin_id = coin["id"]

    detail = coingecko_coin_detail(coin_id)
    platforms = detail.get("platforms") or {}

    sol_mint = platforms.get("solana")
    if sol_mint:
        top1, top10 = parse_solana_top1_top10(moralis_solana_top_holders(sol_mint, limit=10, network="mainnet"))
        if top1 is None:
            print("Cant say")
            return
        print(f"Top holder %: {top1:.2f}%")
        print(f"Top 10 %: {top10:.2f}%")
        return

    _, moralis_chain, contract = pick_supported_evm_platform(platforms)
    if contract and moralis_chain:
        top1, top10 = parse_evm_top1_top10(moralis_erc20_top_holders(contract, chain=moralis_chain, limit=10))
        if top1 is None:
            print("Cant say")
            return
        print(f"Top holder %: {top1:.2f}%")
        print(f"Top 10 %: {top10:.2f}%")
        return

    print("Cant say")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled")
