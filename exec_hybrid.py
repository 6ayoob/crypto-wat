# -*- coding: utf-8 -*-
"""
Hybrid execution layer for Balanced v3 signals
- Quote-based sizing (USDT) + dynamic risk (score & ATR%)
- Order type by setup:
  * BRK/SWEEP  → IOC-limit (slippage-capped), then safe fallback
  * PULL/RANGE/VBR → Limit at entry_low with short timeout → fallback

Requires: okx_api.py (patched) providing `exchange`, `fetch_balance`, `fetch_price`, `place_market_order`.
Signal shape: the dict returned from Balanced v3 `check_signal(...)` (features.setup, score, features.atr_pct, entries, entry, ...)
"""
from __future__ import annotations
import os, time, math
from typing import Dict, Any, Optional

from okx_api import exchange, fetch_balance, fetch_price, place_market_order

# ======== Config (env-tweakable) ========
RISK_PCT_OF_EQUITY   = float(os.getenv("RISK_PCT_OF_EQUITY", "0.02"))   # 2%
MIN_TRADE_USDT       = float(os.getenv("MIN_TRADE_USDT", "10"))
MAX_TRADE_USDT       = float(os.getenv("MAX_TRADE_USDT", "1200"))
ATR_RISK_SCALER      = float(os.getenv("ATR_RISK_SCALER", "2.0"))

MAX_SLIPPAGE_BPS     = int(os.getenv("MAX_SLIPPAGE_BPS", "25"))        # 0.25%
LIMIT_TIMEOUT_SEC     = int(os.getenv("LIMIT_TIMEOUT_SEC", "10"))        # wait then fallback
POLL_INTERVAL_SEC     = float(os.getenv("POLL_INTERVAL_SEC", "1.0"))

USE_QUOTE_MODE_DEFAULT = True  # use quote_amount for market ops

# ======== Helpers ========

def _score_factor(score: int) -> float:
    if score >= 84: return 1.25
    if score >= 76: return 1.10
    return 1.00


def _atr_factor(atr_pct: float) -> float:
    # Higher ATR% → smaller size (defensive)
    return 1.0 / (1.0 + ATR_RISK_SCALER * max(0.0, float(atr_pct or 0.0)))


def compute_trade_usdt(score: int, atr_pct: float, equity_usdt: float) -> float:
    base = max(0.0, equity_usdt) * RISK_PCT_OF_EQUITY
    out = base * _score_factor(score) * _atr_factor(atr_pct)
    return max(MIN_TRADE_USDT, min(MAX_TRADE_USDT, out))


def _to_precision(symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)


def _create_limit_IOC(symbol: str, side: str, amount: float, price: float) -> Dict[str, Any] | None:
    params = {"timeInForce": "IOC"}
    amt = _to_precision(symbol, amount)
    if amt <= 0: return None
    try:
        return exchange.create_order(symbol, type="limit", side=side, amount=amt, price=float(price), params=params)
    except Exception as e:
        print("[IOC-limit][err]", e)
        return None


def _create_limit_GTC(symbol: str, side: str, amount: float, price: float, post_only: bool = False) -> Dict[str, Any] | None:
    params = {"timeInForce": "GTC"}
    if post_only:
        params["postOnly"] = True
    amt = _to_precision(symbol, amount)
    if amt <= 0: return None
    try:
        return exchange.create_order(symbol, type="limit", side=side, amount=amt, price=float(price), params=params)
    except Exception as e:
        print("[limit][err]", e)
        return None


def _fetch_status(order: Dict[str, Any]) -> str:
    try:
        st = (order.get("status") or order.get("info", {}).get("state") or "").lower()
        # ccxt/okx states often: "open"|"live"|"partially_filled"|"filled"|"canceled"|"cancelled"
        if st in ("filled", "closed"): return "filled"
        if st in ("canceled", "cancelled"): return "canceled"
        if st in ("live", "open", "partially_filled", "partial"): return "open"
    except Exception:
        pass
    return "unknown"


def _cancel(order: Dict[str, Any]) -> None:
    try:
        exchange.cancel_order(order.get("id"), order.get("symbol"))
    except Exception:
        pass


def _filled_amount(order: Dict[str, Any]) -> float:
    try:
        f = order.get("filled")
        if f is None and isinstance(order.get("info"), dict):
            f = order["info"].get("accFillSz") or order["info"].get("fillSz")
        return float(f or 0.0)
    except Exception:
        return 0.0

# ======== Core execution ========

def ioc_protected_buy(symbol: str, notional_usdt: float, max_slip_bps: int = MAX_SLIPPAGE_BPS) -> Dict[str, Any] | None:
    """Buy using IOC-limit cap = last * (1 + max_slip). If not filled, fallback to market (quote mode)."""
    last = float(fetch_price(symbol) or 0.0)
    if last <= 0: return None
    cap = last * (1.0 + max_slip_bps / 1e4)
    qty = notional_usdt / cap
    order = _create_limit_IOC(symbol, "buy", qty, cap)
    if order is None: return None
    st = _fetch_status(order)
    if st == "filled" or _filled_amount(order) > 0:
        return order
    # fallback: check current price still within cap
    now = float(fetch_price(symbol) or 0.0)
    if now <= cap:
        return place_market_order(symbol, "buy", None, quote_amount=float(notional_usdt))
    return None


def limit_entry_buy_with_timeout(symbol: str, notional_usdt: float, limit_price: float, timeout_sec: int = LIMIT_TIMEOUT_SEC) -> Dict[str, Any] | None:
    """Place GTC limit at price; wait a short window; if not filled, cancel and fallback to IOC-protected."""
    if limit_price <= 0: return None
    qty = notional_usdt / limit_price
    order = _create_limit_GTC(symbol, "buy", qty, limit_price, post_only=False)
    if order is None: return None
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        st = _fetch_status(order)
        if st == "filled" or _filled_amount(order) > 0:
            return order
        time.sleep(POLL_INTERVAL_SEC)
        # try fetch refreshed order
        try:
            order = exchange.fetch_order(order.get("id"), order.get("symbol"))
        except Exception:
            pass
    # timeout
    _cancel(order)
    return ioc_protected_buy(symbol, notional_usdt, MAX_SLIPPAGE_BPS)


# ======== Public API ========

def execute_signal_hybrid(signal: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Execute a v3 signal dict using hybrid rules. Returns order dict (or None).
    """
    try:
        symbol = signal.get("symbol")
        score = int(signal.get("score", 0))
        atr_pct = float(signal.get("features", {}).get("atr_pct", 0.0))
        setup = (signal.get("features", {}).get("setup") or signal.get("strategy_code") or "").upper()
        entries = signal.get("entries")
        entry_low = float(entries[0]) if isinstance(entries, (list, tuple)) and len(entries) >= 1 else float(signal.get("entry", 0.0))
    except Exception:
        return None

    if not symbol:
        return None

    equity = float(fetch_balance("USDT") or 0.0)
    trade_usdt = compute_trade_usdt(score, atr_pct, equity)

    if trade_usdt < MIN_TRADE_USDT:
        return None

    # Route by setup
    if setup in ("BRK", "SWEEP"):
        return ioc_protected_buy(symbol, trade_usdt, MAX_SLIPPAGE_BPS)
    else:  # PULL / RANGE / VBR
        return limit_entry_buy_with_timeout(symbol, trade_usdt, limit_price=entry_low, timeout_sec=LIMIT_TIMEOUT_SEC)


# ======== Tiny demo ========
if __name__ == "__main__":
    # Example mock usage (requires a real signal dict)
    demo_signal = {
        "symbol": "BTC/USDT",
        "score": 82,
        "entry": 65000.0,
        "entries": [64920.0, 65100.0],
        "features": {"atr_pct": 0.009, "setup": "PULL"},
    }
    print("Running demo (will likely fail without real keys/market):")
    print(execute_signal_hybrid(demo_signal))
