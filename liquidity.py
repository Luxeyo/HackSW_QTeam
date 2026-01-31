import numpy as np
import pandas as pd

### Liquidity functions

'''
1. Assumptions (Explicit)

Pool is constant-product (no concentrated liquidity)
You are selling the token into USDC
Fees are ignored initially (can be added later)
Reserves are known on-chain

'''


def amm_price(x_reserve, y_reserve):
    """
    Spot price from constant-product AMM
    """
    return y_reserve / x_reserve


def price_after_sell(x_reserve, y_reserve, dx):
    """
    New price after selling dx tokens into the pool
    """
    k = x_reserve * y_reserve
    x_new = x_reserve + dx
    y_new = k / x_new
    return y_new / x_new


def price_impact(x_reserve, y_reserve, dx):
    """
    Price impact from selling dx tokens
    """
    p0 = amm_price(x_reserve, y_reserve)
    p1 = price_after_sell(x_reserve, y_reserve, dx)
    return (p0 - p1) / p0


def depth_at_impact(
    x_reserve,
    y_reserve,
    target_impact=0.05,
    tol=1e-6,
    max_iter=100
):
    """
    Binary search for max token amount that causes target price impact
    """
    low, high = 0.0, x_reserve * 0.99

    for _ in range(max_iter):
        mid = (low + high) / 2
        impact = price_impact(x_reserve, y_reserve, mid)

        if abs(impact - target_impact) < tol:
            break
        if impact > target_impact:
            high = mid
        else:
            low = mid

    usd_value = mid * amm_price(x_reserve, y_reserve)
    return {
        "token_amount": mid,
        "usd_depth": usd_value,
        "price_impact": impact
    }

def liquidity_gate(
    x_reserve,
    y_reserve,
    trade_size_usd,
    max_impact=0.08
):
    """
    Returns True if trade passes liquidity gate
    Price impact must stay the same and not be AI-learned because
    it would reinforce survivorship bias
    """
    price = amm_price(x_reserve, y_reserve)
    dx = trade_size_usd / price
    impact = price_impact(x_reserve, y_reserve, dx)

    return {
        "pass": impact <= max_impact,
        "price_impact": impact
    }

# Example pool
token_reserve = 1_000_000      # token units
usdc_reserve = 400_000         # USDC

# Check depth at 5% impact
depth = depth_at_impact(token_reserve, usdc_reserve, target_impact=0.05)
print(depth)

# Gate a $10k trade
gate = liquidity_gate(token_reserve, usdc_reserve, trade_size_usd=100_000)
print(gate)
