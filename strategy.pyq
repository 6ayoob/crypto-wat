# -*- coding: utf-8 -*-
"""
strategy_open_v3.4_pro.py — Spot-only (v3.4 PRO, unified, stable, **open/lenient profile**) 
- MTF (HTF/LTF) gates + breadth-aware thresholds
- Softer defaults per user's env (RVOL/ATR/Notional) + Auto‑Relax
- Smart time exit, multi‑targets, trailing, hourly/DD risk block, daily loss limit
- Safe OKX execution (minNotional/minQty/step/tick) with partial-fill handling
- Market orders by default; optional LIMIT if okx_api.place_limit_order is available; fallback to market
- Telegram notifications optional (STRAT_TG_SEND)

Notes:
- This file honors many ENV flags you provided; defaults below mirror your values.
- SLIPPAGE_MAX_PCT is auto-normalized: if >1 we treat it as percent (e.g., 1.2 -> 0.012).
"""
from __future__ import annotations

import os, json, requests, logging, time, math, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

# ===== OKX API hooks =====
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
try:
    # Optional; will be used only if present and USE_LIMIT_ON_* enabled
    from okx_api import place_limit_order as _place_limit_order  # type: ignore
except Exception:
    _place_limit_order = None  # graceful fallback

# ===== Config imports =====
from config import (
    TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME,
    MIN_NOTIONAL_USDT  # ← platform minNotional (used in filters/partial exits)
)

# ===================== ENV helpers =====================
def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

def _env_str(name, default=""):
    v = os.getenv(name)
    return default if v is None else str(v)

# ===================== Base settings (mirroring your env) =====================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# From your ENV (defaults mirror your values)
DEBUG_LOG_SIGNALS      = _env_bool("DEBUG_LOG_SIGNALS", False)
STRAT_TG_SEND          = _env_bool("STRAT_TG_SEND", False)
ENABLE_MTF_STRICT      = _env_bool("ENABLE_MTF_STRICT", False)
IGNORE_SIGNALS         = _env_bool("IGNORE_SIGNALS", False)

# Timeframes (config-driven)
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME  # you set 1h
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME  # you set 5m

# Indicator windows
EMA_FAST, EMA_SLOW, EMA_TREND, EMA_LONG = 9, 21, 50, 200
VOL_MA = 20
ATR_PERIOD = 14
RVOL_WINDOW = _env_int("RVOL_WINDOW", 30)   # you set 30
NR_WINDOW = 10
NR_FACTOR = 0.75
HTF_EMA_TREND_PERIOD = 50
HTF_SR_WINDOW = 50

# Trade mgmt & generic limits
TP1_FRACTION = 0.5
TRAIL_MIN_STEP_RATIO = _env_float("TRAIL_MIN_STEP_RATIO", 0.001)
MAX_TRADES_PER_DAY       = _env_int("MAX_TRADES_PER_DAY", 20)
MAX_CONSEC_LOSSES        = _env_int("MAX_CONSEC_LOSSES", 3)
DAILY_LOSS_LIMIT_USDT    = _env_float("DAILY_LOSS_LIMIT_USDT", 200.0)

# Sizing & execution
TRADE_BASE_USDT  = _env_float("TRADE_BASE_USDT", 12.0)          # you set 12
MIN_TRADE_USDT   = _env_float("MIN_TRADE_USDT", 12.0)           # you set 12
DRY_RUN          = _env_bool("DRY_RUN", False)
EXEC_USDT_RESERVE= _env_float("EXEC_USDT_RESERVE", 5.0)
EXEC_MIN_FREE_USDT = _env_float("EXEC_MIN_FREE_USDT", 12.0)

# Normalize slippage: accept 1.2 as percent = 0.012
_def_slip = _env_float("SLIPPAGE_MAX_PCT", 0.012)
if _def_slip > 1.0:
    _def_slip = _def_slip / 100.0
SLIPPAGE_MAX_PCT = max(0.0005, min(0.05, _def_slip))

# Feature flags
USE_EMA200_TREND_FILTER   = _env_bool("USE_EMA200_TREND_FILTER", True)
ENABLE_GOLDEN_CROSS_ENTRY = _env_bool("ENABLE_GOLDEN_CROSS_ENTRY", True)
GOLDEN_CROSS_RVOL_BOOST   = _env_float("GOLDEN_CROSS_RVOL_BOOST", 1.10)

# Scoring
SCORE_THRESHOLD = _env_int("SCORE_THRESHOLD", 33)

# ======= Auto-Relax =======
AUTO_RELAX_AFTER_HRS_1 = _env_float("AUTO_RELAX_AFTER_HRS_1", 6)
AUTO_RELAX_AFTER_HRS_2 = _env_float("AUTO_RELAX_AFTER_HRS_2", 12)
RELAX_RVOL_DELTA_1 = _env_float("RELAX_RVOL_DELTA_1", 0.05)
RELAX_RVOL_DELTA_2 = _env_float("RELAX_RVOL_DELTA_2", 0.10)
RELAX_ATR_MIN_SCALE_1 = _env_float("RELAX_ATR_MIN_SCALE_1", 0.9)
RELAX_ATR_MIN_SCALE_2 = _env_float("RELAX_ATR_MIN_SCALE_2", 0.85)
RELAX_RESET_SUCCESS_TRADES = _env_int("RELAX_RESET_SUCCESS_TRADES", 2)

# ======= Market Breadth =======
BREADTH_MIN_RATIO = _env_float("BREADTH_MIN_RATIO", 0.55)
BREADTH_TF        = _env_str("BREADTH_TF", "1h")
BREADTH_TTL_SEC   = _env_int("BREADTH_TTL_SEC", 180)
BREADTH_SYMBOLS_ENV = os.getenv("BREADTH_SAMPLE", "") or os.getenv("BREADTH_SYMBOLS", "")

# ======= Soft schedule =======
SOFT_SCHEDULE_ENABLE      = _env_bool("SOFT_SCHEDULE_ENABLE", True)
SOFT_SCHEDULE_HRS         = _env_str("SOFT_SCHEDULE_HRS", "00:00-23:59")
SOFT_SCHEDULE_WEEKDAYS    = _env_str("SOFT_SCHEDULE_WEEKDAYS", "0,1,2,3,4,5,6")
SOFT_SCALE_TIME_ONLY      = _env_float("SOFT_SCALE_TIME_ONLY", 0.85)
SOFT_SCALE_MARKET_WEAK    = _env_float("SOFT_SCALE_MARKET_WEAK", 0.90)
SOFT_SEVERITY_STEP        = _env_float("SOFT_SEVERITY_STEP", 0.10)
SOFT_MSG_ENABLE           = _env_bool("SOFT_MSG_ENABLE", True)
SOFT_BREADTH_ENABLE       = _env_bool("SOFT_BREADTH_ENABLE", True)
SOFT_BREADTH_SIZE_SCALE   = _env_float("SOFT_BREADTH_SIZE_SCALE", 0.5)

# Exhaustion guards (lenient but safe)
EXH_RSI_MAX = _env_float("EXH_RSI_MAX", 76)
EXH_EMA50_DIST_ATR = _env_float("EXH_EMA50_DIST_ATR", 2.8)

# Multi-targets
ENABLE_MULTI_TARGETS = _env_bool("ENABLE_MULTI_TARGETS", True)
MAX_TP_COUNT = _env_int("MAX_TP_COUNT", 5)
TP_ATR_MULTS_TREND = tuple(float(x) for x in os.getenv("TP_ATR_MULTS_TREND", "1.2,2.2,3.5,4.5,6.0").split(","))
TP_ATR_MULTS_VBR   = tuple(float(x) for x in os.getenv("TP_ATR_MULTS_VBR",   "0.6,1.2,1.8,2.4").split(","))

# Dynamic Max Bars to TP1
USE_DYNAMIC_MAX_BARS = _env_bool("USE_DYNAMIC_MAX_BARS", True)
MAX_BARS_BASE = _env_int("MAX_BARS_BASE", 40)  # lenient window; TP1 timeout soft exit uses this

# Tunables via ENV (relax ATR/RVOL/Notional rejections)
MIN_BAR_NOTIONAL_USD = _env_float("MIN_BAR_NOTIONAL_USD", 25000)
ATR_MIN_BASE = _env_float("ATR_MIN_BASE", 0.0016)
ATR_MIN_NEW  = _env_float("ATR_MIN_NEW",  0.0024)
ATR_MIN_BRT  = _env_float("ATR_MIN_BRT",  0.0018)
# Optional per-trend ATRs
ATR_MIN_FOR_TREND_BASE = _env_float("ATR_MIN_FOR_TREND_BASE", 0.0018)
ATR_MIN_FOR_TREND_NEW  = _env_float("ATR_MIN_FOR_TREND_NEW",  0.0022)
ATR_MIN_FOR_TREND_BRT  = _env_float("ATR_MIN_FOR_TREND_BRT",  0.0020)

RVOL_MIN_NEW = _env_float("RVOL_MIN_NEW", 1.0)
RVOL_MIN_BRT = _env_float("RVOL_MIN_BRT", 1.05)
# Global RVOL baselines/floor (fed into regime thresholds)
RVOL_NEED_BASE_ENV = _env_float("RVOL_NEED_BASE", 0.85)
RVOL_NEED_FLOOR    = _env_float("RVOL_NEED_FLOOR", 0.70)

# Breadth/ATR adaptive thresholds input
USE_DYNAMIC_RISK = _env_bool("USE_DYNAMIC_RISK", True)

# ======= Caches + metrics =======
_HTF_CACHE: Dict[str, Dict[str, Any]] = {}
_HTF_TTL_SEC = _env_int("HTF_CACHE_TTL_SEC", 150)

_OHLCV_CACHE: Dict[tuple, list] = {}
_METRICS = {"ohlcv_api_calls": 0, "ohlcv_cache_hits": 0, "ohlcv_cache_misses": 0, "htf_cache_hits": 0, "htf_cache_misses": 0}

# ======= Rejection counters =======
_REJ_COUNTS = {"atr_low": 0, "rvol_low": 0, "notional_low": 0}
_REJ_SUMMARY: Dict[str, int] = {}

# ===== Logger =====
logger = logging.getLogger("strategy")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def _print(s: str):
    try:
        print(s, flush=True)
    except Exception:
        try:
            import sys
            sys.stdout.write(str(s) + "\n"); sys.stdout.flush()
        except Exception:
            pass

# ================== Basics/helpers ==================
def now_riyadh(): return datetime.now(RIYADH_TZ)

def _today_str(): return now_riyadh().strftime("%Y-%m-%d")

def _hour_key(dt: datetime) -> str: return dt.strftime("%Y-%m-%d %H")

def _atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _df(data):
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    try:
        if len(df) and df["timestamp"].iloc[-1] < 10**12:
            df["timestamp"] = df["timestamp"] * 1000
    except Exception:
        pass
    return df

def _finite_or(default, *vals):
    for v in vals:
        try:
            f = float(v)
            if math.isfinite(f):
                return f
        except Exception:
            pass
    return default

def _split_symbol_variant(symbol: str):
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower().strip()
        if variant in ("srr+", "srrplus", "srr_plus"): variant = "srr_plus"
        elif variant not in ("old","new","srr","brt","vbr","srr_plus"): variant = "new"
        return base, variant
    return symbol, "new"

# ================== Telegram helpers ==================
def _tg(text, parse_mode="HTML"):
    if not STRAT_TG_SEND: return
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        if parse_mode: data["parse_mode"] = parse_mode
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass

_MSG_DEDUP: Dict[str, float] = {}

def _tg_once(key: str, text: str, ttl_sec: int = 900, parse_mode: str = "HTML"):
    now = time.time(); last = _MSG_DEDUP.get(key, 0.0)
    if now - last < ttl_sec: return
    _MSG_DEDUP[key] = now
    _tg(text, parse_mode=parse_mode)

# ================== Indicators ==================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff(); gain = d.where(d > 0, 0.0); loss = -d.where(d < 0, 0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    rs = ag / al; return 100 - (100 / (1 + rs))

def macd_cols(df, fast=12, slow=26, signal=9):
    df["ema_fast"], df["ema_slow"] = ema(df["close"], fast), ema(df["close"], slow)
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_indicators(df):
    df["ema9"]   = ema(df["close"], EMA_FAST)
    df["ema21"]  = ema(df["close"], EMA_SLOW)
    df["ema50"]  = ema(df["close"], EMA_TREND)
    df["ema200"] = ema(df["close"], EMA_LONG)
    df["rsi"] = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA).mean()
    df = macd_cols(df)
    return df

def _ensure_ltf_indicators(df):
    df = add_indicators(df.copy())
    ts = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Riyadh")
    day_changed = ts.dt.date
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tpv"] = tp * df["volume"]
    df["cum_vol"] = df.groupby(day_changed)["volume"].cumsum()
    df["cum_tpv"] = df.groupby(day_changed)["tpv"].cumsum()
    df["vwap"] = (df["cum_tpv"] / df["cum_vol"]).replace([pd.NA, pd.NaT], None)

    vol_ma = df["volume"].rolling(RVOL_WINDOW).mean().replace(0, 1e-9)
    df["rvol"] = df["volume"] / vol_ma

    rng = df["high"] - df["low"]; rng_ma = rng.rolling(NR_WINDOW).mean()
    df["is_nr"] = rng < (NR_FACTOR * rng_ma)

    df["body"] = (df["close"] - df["open"]).abs()
    df["avg_body20"] = df["body"].rolling(20).mean()
    return df

def _atr_from_df(df, period=ATR_PERIOD):
    c = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-c).abs(),
        (df["low"]-c).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-2])

# ================== Symbol filters & balances ==================
def get_usdt_free() -> float:
    try:
        return float(fetch_balance("USDT") or 0.0)
    except Exception:
        return 0.0

# Exchange symbol filters (minQty/minNotional/stepSize/tickSize)
def fetch_symbol_filters(base: str) -> dict:
    try:
        info = {}
        step = float(info.get("stepSize", 0.000001)) or 0.000001
        min_qty = float(info.get("minQty", 0.0)) or 0.0
        min_notional = float(info.get("minNotional", MIN_NOTIONAL_USDT)) or MIN_NOTIONAL_USDT
        tick = float(info.get("tickSize", 0.00000001)) or 0.00000001
    except Exception:
        step, min_qty, min_notional, tick = 0.000001, 0.0, MIN_NOTIONAL_USDT, 0.00000001
    return {"stepSize": step, "minQty": min_qty, "minNotional": min_notional, "tickSize": tick}

def _round_to_tick(px: float, tick: float) -> float:
    if tick <= 0: return float(px)
    return math.floor(float(px) / tick) * tick

# ================== HTF Context ==================
_HTF_MIN = {"5m":5, "15m":15, "30m":30, "45m":45, "1h":60, "2h":120, "4h":240, "1d":1440}

def _tf_minutes(tf: str) -> int: return _HTF_MIN.get(tf.lower(), 60)

def api_fetch_ohlcv(symbol: str, tf: str, bars: int) -> list:
    _METRICS["ohlcv_api_calls"] += 1
    # retry with simple backoff
    last_exc = None
    for i in range(3):
        try:
            return fetch_ohlcv(symbol, tf, bars)
        except Exception as e:
            last_exc = e
            time.sleep(min(6.0, (1.2 * (2 ** i)) * (0.9 + 0.2 * np.random.rand())))
    if last_exc: raise last_exc
    return []

def get_ohlcv_cached(symbol: str, tf: str, bars: int) -> list:
    key = (symbol, tf, bars)
    if key in _OHLCV_CACHE:
        _METRICS["ohlcv_cache_hits"] += 1
        return _OHLCV_CACHE[key]
    _METRICS["ohlcv_cache_misses"] += 1
    data = api_fetch_ohlcv(symbol, tf, bars)
    if data: _OHLCV_CACHE[key] = data
    return data

_def_neutral_pass = True  # will be overridden by regime

def _get_htf_context(symbol):
    base = symbol.split("#")[0]
    now = now_riyadh()
    ent = _HTF_CACHE.get(base)
    if ent and (now - ent["t"]).total_seconds() <= _HTF_TTL_SEC:
        _METRICS["htf_cache_hits"] += 1
        return ent["ctx"]

    _METRICS["htf_cache_misses"] += 1
    data = get_ohlcv_cached(base, HTF_TIMEFRAME, 200)
    if not data: return None
    df = _df(data); df["ema50_htf"] = ema(df["close"], HTF_EMA_TREND_PERIOD)
    if len(df) < HTF_SR_WINDOW + 3: return None

    df_prev = df.iloc[:-2]; w = min(HTF_SR_WINDOW, len(df_prev))
    resistance = _finite_or(None, df_prev["high"].rolling(w).max().iloc[-1])
    support    = _finite_or(None, df_prev["low"].rolling(w).min().iloc[-1])

    closed = df.iloc[-2]
    ema_series = ema(df["close"], HTF_EMA_TREND_PERIOD)
    ema_now  = _finite_or(float(closed["close"]), (ema_series.iloc[-2] if len(ema_series) >= 2 else None))
    ema_prev = _finite_or(ema_now, (ema_series.iloc[-7] if len(ema_series) >= 7 else None))

    ctx: Dict[str, Any] = {
        "close": float(closed["close"]),
        "ema50_now": float(ema_now),
        "ema50_prev": float(ema_prev),
        "support": support,
        "resistance": resistance,
        "mtf": {}
    }

    if ENABLE_MTF_STRICT:
        def _tf_info(tf, bars=160):
            try:
                d = get_ohlcv_cached(base, tf, bars)
                if not d or len(d) < 80: return None
                _dfx = _df(d); _dfx[f"ema{HTF_EMA_TREND_PERIOD}"] = ema(_dfx["close"], HTF_EMA_TREND_PERIOD)
                row = _dfx.iloc[-2]
                e = _finite_or(float(row["close"]), row.get(f"ema{HTF_EMA_TREND_PERIOD}"))
                return {"tf": tf, "price": float(row["close"]), "ema": float(e), "trend_up": bool(float(row["close"]) > float(e))}
            except Exception:
                return None
        mtf = {HTF_TIMEFRAME: {"tf": HTF_TIMEFRAME, "price": ctx["close"], "ema": ctx["ema50_now"], "trend_up": bool(ctx["close"] > ctx["ema50_now"])}}
        for tf in ("1h","4h","1d"):
            info = _tf_info(tf)
            if info: mtf[tf] = info
        ctx["mtf"] = mtf

    _HTF_CACHE[base] = {"t": now, "ctx": ctx}
    return ctx

# ================== Breadth helpers ==================
_BREADTH_CACHE = {"t": 0.0, "ratio": None}

def _breadth_refs() -> List[str]:
    if BREADTH_SYMBOLS_ENV.strip():
        out = []
        for s in BREADTH_SYMBOLS_ENV.split(","):
            s = s.strip()
            if s:
                out.append(s.replace("-", "/").upper().split("#")[0])
        return out
    uniq, seen = [], set()
    for s in SYMBOLS:
        base = s.split("#")[0].replace("-", "/").upper()
        if base not in seen:
            uniq.append(base); seen.add(base)
        if len(uniq) >= 12: break
    if not uniq:
        uniq = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT"]
    return uniq

def _row_is_recent_enough(df: pd.DataFrame, tf: str, bars_back: int = 2) -> bool:
    try:
        last_ts = int(df["timestamp"].iloc[-bars_back])
        if last_ts < 10**12: last_ts *= 1000
        now_ms = int(time.time()*1000)
        return (now_ms - last_ts) <= (2 * _tf_minutes(tf) * 60 * 1000)
    except Exception:
        return False

def _compute_breadth_ratio() -> Optional[float]:
    refs = _breadth_refs()
    if not refs: return None
    ok, tot = 0, 0
    for sym in refs:
        try:
            d = get_ohlcv_cached(sym, BREADTH_TF, 140)
            if not d or len(d) < 60: 
                continue
            df = _df(d)
            if not _row_is_recent_enough(df, BREADTH_TF, bars_back=2):
                continue
            df["ema50"] = ema(df["close"], 50)
            row = df.iloc[-2]
            c = float(row["close"]); e = float(row["ema50"])
            if math.isfinite(c) and math.isfinite(e):
                tot += 1
                if c > e: ok += 1
        except Exception:
            continue
    if tot < 5: return None
    ratio = ok / float(tot)
    if ratio <= 0.05: return None
    return ratio

def _get_breadth_ratio_cached() -> Optional[float]:
    now_s = time.time()
    if _BREADTH_CACHE["ratio"] is not None and (now_s - _BREADTH_CACHE["t"]) <= BREADTH_TTL_SEC:
        return _BREADTH_CACHE["ratio"]
    r = _compute_breadth_ratio()
    _BREADTH_CACHE["ratio"] = r
    _BREADTH_CACHE["t"] = now_s
    return r

# ================== Soft schedule ==================

def _parse_time_hhmm(s: str):
    h, m = s.split(":"); return int(h)*60 + int(m)

def _parse_soft_hours(expr: str):
    spans = []
    for chunk in (expr or "").split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk: continue
        a, b = [x.strip() for x in chunk.split("-", 1)]
        try:
            sa = _parse_time_hhmm(a); sb = _parse_time_hhmm(b)
            spans.append((sa, sb))
        except Exception:
            continue
    return spans

def _is_minute_in_span(mins: int, span: tuple) -> bool:
    sa, sb = span
    if sa == sb: return True
    if sa < sb:  return sa <= mins < sb
    return (mins >= sa) or (mins < sb)

def _is_within_soft_window(dt_local: datetime) -> bool:
    try:
        wd_allowed = {int(x) for x in (SOFT_SCHEDULE_WEEKDAYS or "").split(",") if x.strip().isdigit()}
        if not wd_allowed: wd_allowed = set(range(7))
    except Exception:
        wd_allowed = set(range(7))
    spans = _parse_soft_hours(SOFT_SCHEDULE_HRS)
    if not spans: return False
    wd = dt_local.weekday()
    if wd not in wd_allowed: return False
    mins = dt_local.hour*60 + dt_local.minute
    return any(_is_minute_in_span(mins, sp) for sp in spans)

def _soft_scale_by_time_and_market(br: Optional[float], eff_min: float) -> tuple[float, str]:
    if not SOFT_SCHEDULE_ENABLE: return 1.0, ""
    in_window = _is_within_soft_window(now_riyadh())
    if not in_window: return 1.0, ""
    scale = SOFT_SCALE_TIME_ONLY
    note_parts = [f"TimeSoft×{SOFT_SCALE_TIME_ONLY:.2f}"]
    if br is not None and br < eff_min:
        scale *= SOFT_SCALE_MARKET_WEAK
        note_parts.append(f"MarketWeak×{SOFT_SCALE_MARKET_WEAK:.2f}")
    scale = min(1.0, max(0.50, float(scale)))
    note = " • ".join(note_parts)
    return float(scale), note

# ================== Dynamic thresholds ==================

def _effective_breadth_min() -> float:
    base = BREADTH_MIN_RATIO
    try:
        d = get_ohlcv_cached("BTC/USDT", "4h", 220)
        if not d or len(d) < 100: 
            return base
        df = _df(d); df["ema50"] = ema(df["close"], 50)
        row = df.iloc[-2]
        above = float(row["close"]) > float(row["ema50"])
        rsi_btc = float(rsi(df["close"], 14).iloc[-2])
        if above and rsi_btc >= 55:  return max(0.40, base - 0.15)
        if (not above) or rsi_btc <= 45: return min(0.75, base + 0.10)
    except Exception:
        pass
    return base

def _round_relax_factors():
    f_atr, f_rvol = 1.0, 1.0
    notional_min = MIN_BAR_NOTIONAL_USD
    c = _REJ_COUNTS
    if c["atr_low"] >= 10: f_atr = 0.92
    if c["atr_low"] >= 30: f_atr = 0.85
    if c["rvol_low"]    >= 10: f_rvol = 0.96
    if c["rvol_low"]    >= 30: f_rvol = 0.92
    if c["notional_low"]>= 10: notional_min *= 0.85
    return f_atr, f_rvol, notional_min

# Main regime thresholds (lenient profile)

def regime_thresholds(breadth_ratio: float | None, atrp_now: float) -> dict:
    br = 0.5 if breadth_ratio is None else float(breadth_ratio)
    # Start with a lenient base, then apply env overrides and relax logic
    if br >= 0.60:
        thr = {"ATRP_MIN_MAJ":0.0014,"ATRP_MIN_ALT":0.0016,"ATRP_MIN_MICRO":0.0020,
               "RVOL_NEED_BASE":max(1.05, RVOL_NEED_BASE_ENV),
               "NOTIONAL_AVG_MIN":18000,"NOTIONAL_MINBAR":max(14000, MIN_BAR_NOTIONAL_USD*0.6),
               "NEUTRAL_HTF_PASS":True}
    elif br >= 0.50:
        thr = {"ATRP_MIN_MAJ":0.0016,"ATRP_MIN_ALT":0.0019,"ATRP_MIN_MICRO":0.0023,
               "RVOL_NEED_BASE":max(1.10, RVOL_NEED_BASE_ENV),
               "NOTIONAL_AVG_MIN":21000,"NOTIONAL_MINBAR":max(18000, MIN_BAR_NOTIONAL_USD*0.9),
               "NEUTRAL_HTF_PASS":True}
    else:
        thr = {"ATRP_MIN_MAJ":0.0019,"ATRP_MIN_ALT":0.0023,"ATRP_MIN_MICRO":0.0027,
               "RVOL_NEED_BASE":max(1.18, RVOL_NEED_BASE_ENV),
               "NOTIONAL_AVG_MIN":25000,"NOTIONAL_MINBAR":max(22000, MIN_BAR_NOTIONAL_USD),
               "NEUTRAL_HTF_PASS":False}

    if atrp_now >= 0.01:
        thr["RVOL_NEED_BASE"] = max(1.03, thr["RVOL_NEED_BASE"] - 0.05)

    f_atr, f_rvol, notional_min = _round_relax_factors()
    thr["RVOL_NEED_BASE"] *= f_rvol
    thr["ATRP_MIN_ALT"]  *= f_atr
    thr["ATRP_MIN_MAJ"]  *= f_atr
    thr["ATRP_MIN_MICRO"]*= f_atr
    thr["NOTIONAL_MINBAR"] = max(thr["NOTIONAL_MINBAR"]*0.95, notional_min*0.95)

    # Respect floors
    thr["RVOL_NEED_BASE"] = max(RVOL_NEED_FLOOR, thr["RVOL_NEED_BASE"])

    # Soft scaling during allowed hours / weak market
    try:
        eff_min = _effective_breadth_min()
        scale, _note = _soft_scale_by_time_and_market(br, eff_min)
        if scale < 1.0:
            ease = 1.0 - (1.0 - scale) * 0.3
            thr["RVOL_NEED_BASE"] = max(RVOL_NEED_FLOOR, thr["RVOL_NEED_BASE"] * ease)
            thr["ATRP_MIN_ALT"]   = thr["ATRP_MIN_ALT"]  * ease
            thr["ATRP_MIN_MAJ"]   = thr["ATRP_MIN_MAJ"]  * ease
            thr["ATRP_MIN_MICRO"] = thr["ATRP_MIN_MICRO"]* ease
    except Exception:
        pass

    return thr

# ================== SR & patterns ==================

def _bullish_engulf(prev, cur):
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and \
           (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])

# ================== Positions storage ==================

def _pos_path(symbol):
    os.makedirs(POSIONS_DIR := POSITIONS_DIR, exist_ok=True)
    return f"{POSIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol): return _read_json(_pos_path(symbol), None)

def save_position(symbol, position): _atomic_write(_pos_path(symbol), position)

def clear_position(symbol):
    try:
        p = _pos_path(symbol)
        if os.path.exists(p): os.remove(p)
    except Exception:
        pass

def count_open_positions():
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions(): return _read_json(CLOSED_POSITIONS_FILE, [])

def save_closed_positions(lst): _atomic_write(CLOSED_POSITIONS_FILE, lst)

# ================== Gates & scoring ==================

def _ensure_htf_indicators(df):
    try:
        if "ema21" not in df: df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        if "ema50" not in df: df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        if "rsi14" not in df:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss.replace(0, 1e-9))
            df["rsi14"] = 100 - (100 / (1 + rs))
    except Exception:
        pass
    return df

def macd_rsi_gate(prev_row, closed_row, policy):
    if not policy: return True
    rsi_now = float(closed_row.get("rsi", 50.0))
    rsi_up  = rsi_now > float(prev_row.get("rsi", rsi_now))
    macd_h_now  = float(closed_row.get("macd_hist", 0.0))
    macd_h_prev = float(prev_row.get("macd_hist", 0.0))
    macd_pos    = macd_h_now > 0
    macd_up     = macd_h_now > macd_h_prev
    flags = []
    if rsi_now > 50: flags.append("RSI>50")
    if rsi_up:       flags.append("RSI↑")
    if macd_pos:     flags.append("MACD_hist>0")
    if macd_up:      flags.append("MACD_hist↑")
    k = len(flags)
    if policy == "lenient": return k >= 1
    if policy == "strict":  return ("RSI>50" in flags and "MACD_hist>0" in flags and "MACD_hist↑" in flags)
    return k >= 2

# --- Config profiles per variant ---
BASE_CFG = {
    "ENTRY_MODE": os.getenv("ENTRY_MODE", "pullback").lower(),  # you set pullback
    "HYBRID_ORDER": [x.strip() for x in os.getenv("HYBRID_ORDER", "pullback,breakout").split(",")],
    "PULLBACK_VALUE_REF": "vwap",           # vwap preferred in open profile
    "PULLBACK_CONFIRM": "bos",             # or bullish_engulf / sweep_reclaim
    "RVOL_MIN": RVOL_MIN_NEW,
    "ATR_MIN_FOR_TREND": ATR_MIN_FOR_TREND_NEW,
    "USE_FIB": True,
    "SWING_LOOKBACK": 60,
    "FIB_TOL": 0.004,
    "BREAKOUT_BUFFER_LTF": 0.0018,
    "RSI_GATE_POLICY": "lenient",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.003,
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 8,
}

SRR_OVERRIDES = {
    "ENTRY_MODE": "pullback",
    "PULLBACK_VALUE_REF": "ema21",
    "PULLBACK_CONFIRM": "bullish_engulf",
    "RVOL_MIN": 1.20,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.004,
    "MAX_HOLD_HOURS": 8,
    "SYMBOL_COOLDOWN_MIN": 10,
}

BRT_OVERRIDES = {
    "ENTRY_MODE": "breakout",
    "RVOL_MIN": RVOL_MIN_BRT,
    "ATR_MIN_FOR_TREND": ATR_MIN_FOR_TREND_BRT,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True, "SL_ATR_MULT": 0.9, "TP1_ATR_MULT": 1.4, "TP2_ATR_MULT": 2.4,
    "TRAIL_AFTER_TP1": True, "TRAIL_ATR_MULT": 1.1,
    "LOCK_MIN_PROFIT_PCT": 0.004, "MAX_HOLD_HOURS": 8, "SYMBOL_COOLDOWN_MIN": 10,
}

VBR_OVERRIDES = {
    "ENTRY_MODE": "pullback",
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True, "SL_ATR_MULT": 0.8, "TP1_ATR_MULT": 1.2, "TP2_ATR_MULT": 2.0,
    "TRAIL_AFTER_TP1": True, "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.003, "MAX_HOLD_HOURS": 6, "SYMBOL_COOLDOWN_MIN": 8,
}

PER_STRAT_MGMT = {
    "new": {"SL":"atr", "SL_MULT":0.9, "TP1":"sr_or_atr", "TP1_ATR":1.2, "TP2_ATR":2.2,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":1.0, "TIME_HRS":6},
    "old": {"SL":"pct", "SL_PCT":0.02, "TP1_PCT":0.03, "TP2_PCT":0.06,
             "TRAIL_AFTER_TP1":False, "TIME_HRS":12},
    "srr_plus": {"SL":"atr_below_sweep", "SL_MULT":0.8, "TP1":"sr_or_atr", "TP1_ATR":1.2, "TP2_ATR":2.3,
                  "TRAIL_AFTER_TP1":True, "TRAIL_ATR":1.0, "TIME_HRS":6},
    "brt": {"SL":"atr_below_retest", "SL_MULT":1.0, "TP1":"range_or_atr", "TP1_ATR":1.5, "TP2_ATR":2.5,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":0.9, "TIME_HRS":8},
    "vbr": {"SL":"atr", "SL_MULT":1.0, "TP1":"vwap_or_sr", "TP2_ATR":1.8,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":0.8, "TIME_HRS":3},
}

def _mgmt(variant: str): return PER_STRAT_MGMT.get(variant, PER_STRAT_MGMT["new"])

# ---------- Opportunity scoring (lightweight) ----------

def _opportunity_score(df, prev, closed):
    score, why, pattern = 0, [], ""
    try:
        if closed["close"] > closed["open"]: score += 8; why.append("BullishClose")
        if closed["close"] > closed.get("ema21", closed["close"]): score += 8; why.append("AboveEMA21")
        if closed["close"] > closed.get("ema50", closed["close"]): score += 8; why.append("AboveEMA50")
        rvol = float(closed.get("rvol", 0) or 0)
        if rvol >= 1.3: score += 14; why.append("HighRVOL")
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        if nr_recent and (closed["close"] > hi_range):
            score += 18; why.append("NR_Breakout"); pattern = "NR_Breakout"
        if _bullish_engulf(prev, closed):
            score += 18; why.append("BullishEngulf"); pattern = pattern or "BullishEngulf"
        ref = _finite_or(None, closed.get("vwap"), closed.get("ema21"))
        if ref:
            rng14 = (df["high"]-df["low"]).rolling(14).mean().iloc[-2]
            if rng14 and abs(float(closed["close"]) - ref) <= 0.4 * float(rng14):
                score += 6; why.append("NearValue")
    except Exception:
        pass
    return score, ", ".join(why), (pattern or "Generic")

# ---------- Entry logics ----------

def _swing_points(df, left=2, right=2):
    highs, lows = df["high"], df["low"]
    idx = len(df) - 3
    swing_high = swing_low = None
    for i in range(max(0, idx-10), idx+1):
        if i-left < 0 or i+right >= len(df): continue
        if highs[i] == max(highs[i-left:i+right+1]): swing_high = float(highs[i])
        if lows[i]  == min(lows[i-left:i+right+1]):  swing_low  = float(lows[i])
    return swing_high, swing_low

def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    ref_val = _finite_or(None, (closed.get("vwap") if cfg.get("PULLBACK_VALUE_REF","vwap").lower()=="vwap" else closed.get("ema21")), closed.get("ema21"))
    ref_val = _finite_or(float(closed["close"]), ref_val, closed.get("ema50"), closed.get("close"))
    close_v = _finite_or(None, closed.get("close")); low_v = _finite_or(None, closed.get("low"))
    if close_v is None or low_v is None: return False
    near_val = (close_v >= ref_val) and (low_v <= ref_val)
    if not near_val: return False
    # Lenient MACD/RSI gate
    if not macd_rsi_gate(prev, closed, cfg.get("RSI_GATE_POLICY")): return False
    confirm = (cfg.get("PULLBACK_CONFIRM") or "").lower()
    if confirm == "bullish_engulf":
        return _bullish_engulf(prev, closed)
    if confirm == "bos":
        swing_high, _ = _swing_points(df)
        sh = _finite_or(None, swing_high)
        return bool(sh is not None and close_v > sh)
    if confirm == "sweep_reclaim":
        try:
            lows_window = df["low"].iloc[-22:-2]; ll = float(lows_window.min())
            cur_low = float(closed["low"]); cur_close = float(closed["close"])
            swept = (cur_low <= ll * (1.0 - 0.0012)) or (cur_low <= ll)
            reclaimed = cur_close >= float(ref_val)
            bullish_body = (cur_close > float(closed["open"])) or _bullish_engulf(prev, closed)
            return bool(swept and reclaimed and bullish_body)
        except Exception:
            return False
    return True

def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    try:
        n = int(cfg.get("SWING_LOOKBACK", 60)); buf = float(cfg.get("BREAKOUT_BUFFER_LTF", 0.0015))
        if len(df) < max(40, n+3): return False
        hi = float(df["high"].iloc[-n-2:-2].max())
        close_v = float(closed["close"]); open_v = float(closed["open"])
        vwap_v  = _finite_or(None, closed.get("vwap"), closed.get("ema21"))
        bo = (close_v > hi * (1.0 + buf)) and (close_v > open_v)
        rvol_v = float(_finite_or(1.0, closed.get("rvol"), 1.0))
        need_rvol = float(cfg.get("RVOL_MIN", 1.2))
        if bo and (rvol_v >= max(need_rvol, 1.05)): return True
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        if nr_recent and (close_v > hi): return True
        if bo and vwap_v is not None:
            rng14 = (df["high"] - df["low"]).rolling(14).mean().iloc[-2]
            if rng14 and abs(close_v - vwap_v) <= 0.6 * float(rng14):
                return True
        return False
    except Exception:
        return False

# ================== Rejections tracking ==================
_LAST_REJECT: Dict[str, Any] = {}
_CURRENT_SYMKEY: Optional[str] = None

def _rej(stage, **kv):
    if stage in _REJ_COUNTS: _REJ_COUNTS[stage] += 1
    try:
        if _CURRENT_SYMKEY:
            _LAST_REJECT[_CURRENT_SYMKEY] = {
                "stage": stage,
                "details": {k: (float(v) if isinstance(v, (int,float)) else v) for k,v in kv.items()},
                "ts": now_riyadh().isoformat(timespec="seconds")
            }
    except Exception: pass
    if DEBUG_LOG_SIGNALS:
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"[REJECT] {stage} | {kvs}")
    return None

def _pass(stage, **kv):
    if DEBUG_LOG_SIGNALS:
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"[PASS]   {stage} | {kvs}")

# ================== Signal check ==================

def check_signal(symbol: str):
    if IGNORE_SIGNALS:
        return _rej("ignored")
    global _CURRENT_SYMKEY
    _CURRENT_SYMKEY = symbol
    try:
        # --- HTF ---
        htf_ctx = _get_htf_context(symbol)
        if not htf_ctx: return _rej("data_unavailable")

        # --- LTF ---
        ltf = get_ohlcv_cached(symbol, LTF_TIMEFRAME, 140)
        if not ltf or len(ltf) < 80: return _rej("no_ltf")
        df = _df(ltf); df = _ensure_ltf_indicators(df)
        if len(df) < 60: return _rej("no_ltf")
        closed = df.iloc[-2]; prev = df.iloc[-3]

        # ATR%
        atr_val = _finite_or(None, _atr_from_df(df))
        price   = _finite_or(None, closed.get("close"))
        if atr_val is None or price is None or price <= 0: return _rej("atr_calc")
        atrp = float(atr_val) / float(price)

        # bucket (maj/alt)
        base, variant = _split_symbol_variant(symbol)
        bucket = "maj" if base.split("/")[0] in ("BTC","ETH","BNB","SOL") else "alt"

        # Light liquidity context (lenient, uses LTF with conservative 1h cap)
        sym_ctx = {
            "bucket": bucket,
            "atrp_q35_lookback": float(df["close"].pct_change().rolling(35).std().iloc[-1] or 0),
            "price": float(price),
            "notional_avg_30": float(df["volume"].iloc[-30:].mean() * float(price)),
            "notional_min_30": float(df["volume"].iloc[-30:].min()  * float(price)),
            "is_meme": False,
        }
        try:
            d1h = get_ohlcv_cached(base, "1h", 80)
            if d1h and len(d1h) >= 35:
                df1h = _df(d1h)
                px1h  = float(df1h["close"].iloc[-2])
                vol30 = float(df1h["volume"].iloc[-30:].mean())
                minbar30 = float(df1h["volume"].iloc[-30:].min())
                avg_notional_30_h1 = vol30 * px1h
                min_notional_30_h1 = minbar30 * px1h
                sym_ctx["notional_avg_30"] = float(min(sym_ctx["notional_avg_30"], avg_notional_30_h1))
                sym_ctx["notional_min_30"] = float(min(sym_ctx["notional_min_30"], min_notional_30_h1))
        except Exception:
            pass

        # LTF helpers
        rvol = float(_finite_or(1.0, closed.get("rvol"), 1.0))
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_slice = df["high"].iloc[-NR_WINDOW-2:-2]
        if len(hi_slice) < 3: return _rej("no_ltf")
        hi_range = float(hi_slice.max())
        if not math.isfinite(hi_range): return _rej("no_ltf")
        is_breakout = bool((float(closed["close"]) > hi_range) and (nr_recent or float(closed["close"]) > _finite_or(float(closed["close"]), closed.get("vwap"), closed.get("ema21"))))

        # LTF trend via EMA200
        try:
            ema200_val = float(closed.get("ema200"))
            if float(closed["close"]) > ema200_val: ema200_trend = "up"
            elif float(closed["close"]) < ema200_val: ema200_trend = "down"
            else: ema200_trend = "flat_up"
        except Exception:
            ema200_trend = "flat_up"

        ltf_ctx = {"rvol": rvol, "is_breakout": is_breakout, "ema200_trend": ema200_trend,
                   "pullback_ok": (float(closed["low"]) <= _finite_or(float(closed["close"]), closed.get("vwap"), closed.get("ema21")) <= float(closed["close"]))}

        # Thresholds by breadth/ATR
        br = _get_breadth_ratio_cached()
        thr = regime_thresholds(br, atrp)

        # HTF trend gate
        trend = "neutral"
        try:
            if float(htf_ctx["close"]) > float(htf_ctx["ema50_now"]): trend = "up"
            elif float(htf_ctx["close"]) < float(htf_ctx["ema50_now"]): trend = "down"
        except Exception:
            trend = "neutral"
        neutral_ok = bool(thr.get("NEUTRAL_HTF_PASS", True))
        eff_min = _effective_breadth_min()
        if trend == "down":
            if not (br is not None and br >= max(0.58, eff_min + 0.04)):
                return _rej("htf_trend", trend=trend)
        elif trend == "neutral" and not neutral_ok:
            if not (br is not None and br >= eff_min):
                return _rej("htf_trend", trend=trend)

        # ATR% check (bucket-aware)
        bucket_map = {"maj": thr["ATRP_MIN_MAJ"], "alt": thr["ATRP_MIN_ALT"], "micro": thr["ATRP_MIN_MICRO"]}
        need_atrp = float(bucket_map.get(bucket, thr["ATRP_MIN_ALT"]))
        if float(atrp) < need_atrp:
            return _rej("atr_low", atrp=atrp, need=need_atrp)

        # RVOL check
        r_need = thr["RVOL_NEED_BASE"]
        if ltf_ctx.get("is_breakout"): r_need = max(RVOL_NEED_FLOOR, r_need - 0.05)
        if ENABLE_GOLDEN_CROSS_ENTRY and float(closed.get("ema50",0)) > float(prev.get("ema50",0)) and float(closed.get("ema21",0)) > float(prev.get("ema21",0)):
            r_need = max(RVOL_NEED_FLOOR, r_need - (GOLDEN_CROSS_RVOL_BOOST - 1.0))
        if float(rvol) < r_need:
            return _rej("rvol_low", rvol=rvol, need=r_need)

        # Notional check (lenient)
        avg_not, minbar = float(sym_ctx["notional_avg_30"]), float(sym_ctx["notional_min_30"])
        if not (avg_not >= thr["NOTIONAL_AVG_MIN"] and minbar >= thr["NOTIONAL_MINBAR"]):
            return _rej("notional_low", avg=avg_not, minbar=minbar)

        # Entry mode
        cfg = get_cfg(_split_symbol_variant(symbol)[1])
        mode_pref = cfg.get("ENTRY_MODE", "pullback")
        chosen_mode = None
        order = cfg.get("HYBRID_ORDER", ["pullback","breakout"]) if mode_pref == "hybrid" else [mode_pref]
        for m in (order + [x for x in ["pullback","breakout"] if x not in order]):
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr_val, htf_ctx, cfg):
                chosen_mode = "pullback"; break
            if m == "breakout" and _entry_breakout_logic(df, closed, prev, atr_val, htf_ctx, cfg):
                chosen_mode = "breakout"; break
        if not chosen_mode:
            return _rej("entry_mode", mode=mode_pref)

        # Score
        score, why_str, pattern = _opportunity_score(df, prev, closed)
        if SCORE_THRESHOLD and int(score) < int(SCORE_THRESHOLD):
            return _rej("score_low", score=score, need=SCORE_THRESHOLD)

        _pass("buy", mode=chosen_mode, score=int(score))
        return {
            "decision": "buy",
            "mode": chosen_mode,
            "score": int(score),
            "reasons": why_str,
            "pattern": pattern,
            "features": {
                "atrp": float(atrp),
                "rvol": float(rvol),
                "breadth_ratio": (None if br is None else float(br)),
                "htf_ok": bool(trend in ("up","neutral")),
                "notional_avg_30": float(avg_not),
                "notional_min_30": float(minbar),
            }
        }
    except Exception as e:
        return _rej("error", err=str(e))
    finally:
        _CURRENT_SYMKEY = None

# ================== Config merging ==================

def get_cfg(variant: str):
    cfg = dict(BASE_CFG)
    v = (variant or "new").lower()
    if v == "new":
        pass  # BASE_CFG already represents open profile for new
    elif v == "srr":
        cfg.update(SRR_OVERRIDES)
    elif v == "srr_plus":
        cfg.update(SRR_OVERRIDES)
        cfg["PULLBACK_CONFIRM"] = "sweep_reclaim"
        cfg["RSI_GATE_POLICY"] = "strict"
        cfg["SL_ATR_MULT"] = 0.8
    elif v == "brt":
        cfg.update(BRT_OVERRIDES)
    elif v == "vbr":
        cfg.update(VBR_OVERRIDES)
    elif v == "old":
        # classic
        cfg.update({"ENTRY_MODE":"pullback","PULLBACK_VALUE_REF":"ema21","PULLBACK_CONFIRM":"bullish_engulf",
                    "USE_ATR_SL_TP": False, "STOP_LOSS_PCT": 0.02, "TP1_PCT": 0.03, "TP2_PCT": 0.06,
                    "TRAIL_AFTER_TP1": False, "LOCK_MIN_PROFIT_PCT": 0.0, "MAX_HOLD_HOURS": 12})
    else:
        pass
    return cfg

# ================== Entry plan & execution ==================

def _atr_latest(symbol_base: str, tf: str, bars: int = 180) -> tuple[float, float, float]:
    data = get_ohlcv_cached(symbol_base, tf, bars)
    if not data: raise RuntimeError("no LTF data")
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 50: raise RuntimeError("ltf too short")
    closed = df.iloc[-2]; px = float(closed["close"]) 
    atr_abs = _atr_from_df(df)
    if not atr_abs or atr_abs <= 0: raise RuntimeError("atr invalid")
    atrp = atr_abs / max(px, 1e-9)
    return px, atr_abs, atrp

def _partials_for(score: int, tp_count: int, atrp: float) -> list:
    tp_count = max(1, min(int(tp_count), MAX_TP_COUNT))
    base = [1.0] if tp_count == 1 else [0.5, 0.3, 0.2, 0.0, 0.0][:tp_count]
    if score >= 55 and tp_count >= 3: base = [0.45, 0.30, 0.25, 0.0, 0.0][:tp_count]
    if atrp >= 0.008 and tp_count >= 3: base = [0.40, 0.30, 0.30, 0.0, 0.0][:tp_count]
    s = sum(base); return [round(x/s, 6) for x in base]

def _build_entry_plan(symbol: str, sig: dict | None) -> dict:
    base, variant = _split_symbol_variant(symbol)
    cfg = get_cfg(variant)

    if sig is None:
        r = check_signal(symbol)
        if not (isinstance(r, dict) and r.get("decision") == "buy"):
            raise RuntimeError("no buy signal")
        sig = r

    price, atr_abs, atrp = _atr_latest(base, LTF_TIMEFRAME)

    mgmt = _mgmt(variant)
    if mgmt.get("SL") in ("atr","atr_below_sweep","atr_below_retest"):
        sl_mult = float(mgmt.get("SL_MULT", 1.0)); sl = float(price - sl_mult * atr_abs)
    elif mgmt.get("SL") == "pct":
        sl = float(price * (1.0 - float(mgmt.get("SL_PCT", 0.02))))
    else:
        sl = float(price - 1.0 * atr_abs)

    tps: list[float] = []
    if ENABLE_MULTI_TARGETS:
        mults = []
        for k in ("TP1_ATR", "TP2_ATR"):
            if k in mgmt: mults.append(float(mgmt[k]))
        if not mults: mults = list(TP_ATR_MULTS_TREND)[:3]
        for m in mults[:MAX_TP_COUNT]:
            tps.append(float(price + float(m) * atr_abs))
    else:
        tps.append(float(price + float(mgmt.get("TP1_ATR", 1.2)) * atr_abs))
        tps.append(float(price + float(mgmt.get("TP2_ATR", 2.2)) * atr_abs))

    partials = _partials_for(int(sig.get("score", SCORE_THRESHOLD)), len(tps), atrp)

    max_bars = None
    if USE_DYNAMIC_MAX_BARS:
        if atrp >= 0.01: max_bars = MAX_BARS_BASE + 6
        elif atrp >= 0.006: max_bars = MAX_BARS_BASE + 3
        else: max_bars = MAX_BARS_BASE

    sig = dict(sig)
    sig["sl"] = float(sl)
    sig["targets"] = [float(x) for x in tps]
    sig["partials"] = partials
    sig["atrp"] = float(atrp)
    sig["max_bars_to_tp1"] = max_bars
    sig.setdefault("messages", {})
    return sig

# ----- LIMIT helper (optional) -----

def _maybe_place_limit_or_market(side: str, base: str, want_amount: float, ref_price: float, mode: str):
    use_limit_pull = _env_bool("USE_LIMIT_ON_PULLBACK", False)
    use_limit_bo   = _env_bool("USE_LIMIT_ON_BREAKOUT", False)
    if (_place_limit_order is None) or (mode == "pullback" and not use_limit_pull) or (mode == "breakout" and not use_limit_bo):
        # Market fallback
        if DRY_RUN:
            return {"id": f"dry_{side}_{int(time.time())}", "average": ref_price, "filled": float(want_amount)}
        return place_market_order(base, side, want_amount)

    try:
        off_bps = _env_int("LIMIT_OFFSET_BPS_PULLBACK", 3) if mode == "pullback" else _env_int("LIMIT_OFFSET_BPS_BREAKOUT", 5)
        px = ref_price * (1.0 + (off_bps / 10000.0) * (1 if side=="buy" else -1))
        if DRY_RUN:
            return {"id": f"dry_limit_{side}_{int(time.time())}", "average": px, "filled": float(want_amount)}
        # Best‑effort: place limit and assume immediate/near fill; if not supported, exception -> fallback handled by caller
        return _place_limit_order(base, side, want_amount, px)
    except Exception:
        # Fallback to market
        if DRY_RUN:
            return {"id": f"dry_{side}_{int(time.time())}", "average": ref_price, "filled": float(want_amount)}
        return place_market_order(base, side, want_amount)

# ================== Execute BUY ==================

def execute_buy(symbol: str, sig: dict | None = None):
    base, variant = _split_symbol_variant(symbol)
    sig = _build_entry_plan(symbol, sig)

    # Max open positions
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 تم بلوغ الحد الأقصى للمراكز المفتوحة."

    # Risk block gate is handled in the main loop typically; you can add here if desired

    # Balance checks
    usdt_free = get_usdt_free()
    if usdt_free < EXEC_MIN_FREE_USDT:
        return None, f"🚫 رصيد USDT غير كافٍ ({usdt_free:.2f}$ < {EXEC_MIN_FREE_USDT:.2f}$)."
    max_affordable = max(0.0, usdt_free - EXEC_USDT_RESERVE)
    if max_affordable <= 0:
        return None, f"🚫 الاحتياطي محجوز ({EXEC_USDT_RESERVE:.2f}$)."

    # Sizing with breadth soft scale
    trade_usdt = float(TRADE_BASE_USDT)
    br = _get_breadth_ratio_cached(); eff_min = _effective_breadth_min()
    if br is not None:
        if br < 0.45:   trade_usdt *= 0.72
        elif br < 0.55: trade_usdt *= 0.88
    if SOFT_BREADTH_ENABLE and (br is not None) and (br < eff_min):
        scale, note = _soft_scale_by_time_and_market(br, eff_min)
        trade_usdt *= scale
        if SOFT_MSG_ENABLE:
            sig.setdefault("messages", {})
            sig["messages"]["breadth_soft"] = f"⚠️ Soft breadth: ratio={br:.2f} < min={eff_min:.2f} → size×{scale:.2f}"

    # Affordability
    trade_usdt = min(trade_usdt, max_affordable)

    # Platform filters
    f = fetch_symbol_filters(base)
    step = f["stepSize"]; min_qty = f["minQty"]; min_notional = f["minNotional"]; tick = f["tickSize"]

    price = float(fetch_price(base))
    if not (price > 0):
        return None, "⚠️ لا يمكن جلب سعر صالح."

    raw_amount = max(MIN_TRADE_USDT, trade_usdt) / price
    amount = math.floor(raw_amount / step) * step

    if min_qty and amount < min_qty:
        amount = min_qty
        trade_usdt = amount * price
        if trade_usdt > max_affordable:
            return None, "🚫 لا يمكن بلوغ الحد الأدنى للكمية ضمن الرصيد."

    if amount * price < min_notional:
        need_amt = math.ceil((min_notional / price) / step) * step
        trade_usdt = need_amt * price
        if trade_usdt > max_affordable:
            _tg_once(f"warn_min_notional:{base}", f"⚠️ <b>قيمة الصفقة أقل من الحد الأدنى</b>\nالقيمة: <code>{amount*price:.2f}$</code> • الحد الأدنى: <code>{min_notional:.2f}$</code>.", ttl_sec=900)
            return None, "🚫 قيمة الصفقة أقل من الحد الأدنى ضمن رصيدك المتاح."
        amount = need_amt

    # Execute (market by default / optional limit)
    try:
        order = _maybe_place_limit_or_market("buy", base, amount, price, sig.get("mode","pullback"))
    except Exception as e:
        _tg_once(f"buy_fail_{base}", f"❌ فشل شراء {base}: {e}", ttl_sec=600)
        return None, "⚠️ فشل تنفيذ الصفقة (استثناء)."
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    fill_px = float(order.get("average") or order.get("price") or price)
    filled_amt = float(order.get("filled") or amount)
    if filled_amt <= 0:
        return None, "⚠️ لم يتم تنفيذ أي كمية."
    amount = filled_amt

    # Slippage guard & rollback (best‑effort)
    slippage = abs(fill_px - price) / price
    if slippage > SLIPPAGE_MAX_PCT:
        try:
            if not DRY_RUN:
                place_market_order(base, "sell", amount)
        except Exception:
            pass
        return None, f"🚫 انزلاق مرتفع وتم التراجع عن العملية ({slippage:.2%} > {SLIPPAGE_MAX_PCT:.2%})."

    # Save position
    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": _round_to_tick(fill_px, tick),
        "stop_loss": float(sig["sl"]),
        "targets": [float(x) for x in sig["targets"]],
        "partials": list(sig.get("partials") or []),
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": _split_symbol_variant(symbol)[1],
        "htf_stop": sig.get("stop_rule"),
        "max_bars_to_tp1": sig.get("max_bars_to_tp1"),
        "messages": sig.get("messages", {}),
        "tp_hits": [False] * len(sig["targets"]),
        "score": sig.get("score"),
        "pattern": sig.get("pattern"),
        "reason": sig.get("reasons"),
        "max_hold_hours": _mgmt(_split_symbol_variant(symbol)[1]).get("TIME_HRS"),
    }

    # Cache ATR at entry for SmartExit
    try:
        df_ltf = _df(get_ohlcv_cached(base, LTF_TIMEFRAME, 120))
        if len(df_ltf) >= 40:
            pos["atr_entry"] = float(_atr_from_df(df_ltf))
    except Exception as e:
        _print(f"[execute_buy] atr_entry save error {symbol}: {e}")

    save_position(symbol, pos)

    try:
        if STRAT_TG_SEND:
            msg = (
                f"✅ دخول {symbol}\n"
                f"🎯 <b>Mode</b>: {sig.get('mode','-')} • <b>Score</b>: {sig.get('score','-')} • <b>Pattern</b>: {sig.get('pattern','-')}\n"
                f"🟢 <b>Entry</b>: <code>{pos['entry_price']:.6f}</code>\n"
                f"🛡️ <b>SL</b>: <code>{pos['stop_loss']:.6f}</code>\n"
                f"🎯 <b>TPs</b>: {', '.join(str(round(t,6)) for t in pos['targets'])}\n"
                f"💰 <b>الحجم</b>: {trade_usdt:.2f}$"
            )
            if pos["messages"].get("breadth_soft"): msg += f"\n{pos['messages']['breadth_soft']}"
            _tg(msg)
    except Exception:
        pass

    return order, f"✅ شراء {symbol} | SL: {pos['stop_loss']:.6f} | 💰 {trade_usdt:.2f}$"

# ================== Safe SELL ==================

def _safe_sell(base_symbol: str, want_qty: float):
    try:
        avail = float(fetch_balance(base_symbol.split("/")[0]) or 0.0)
    except Exception:
        avail = 0.0
    if avail <= 0.0:
        _tg_once(f"warn_insuff_{base_symbol}", f"⚠️ لا توجد كمية متاحة للبيع لـ {base_symbol}.", ttl_sec=600)
        return None, None, 0.0

    f = fetch_symbol_filters(base_symbol)
    step = float(f.get("stepSize", 0.000001)) or 0.000001
    min_qty = float(f.get("minQty", 0.0)) or 0.0
    min_notional = float(f.get("minNotional", MIN_NOTIONAL_USDT)) or MIN_NOTIONAL_USDT

    try:
        price_now = float(fetch_price(base_symbol) or 0.0)
    except Exception:
        price_now = 0.0

    raw = max(0.0, min(float(want_qty or 0.0), avail))
    qty = math.floor(raw / step) * step

    if min_qty and qty < min_qty:
        qty = math.floor(min(avail, min_qty) / step) * step

    if price_now <= 0 or (qty * price_now) < min_notional:
        qty = math.floor(avail / step) * step
        if price_now <= 0 or (qty * price_now) < min_notional or qty <= 0:
            _tg_once(
                f"sell_skip_small_{base_symbol}",
                (f"⚠️ تخطّي البيع {base_symbol}: القيمة {qty*price_now:.2f}$ أقل من الحد الأدنى "
                 f"{min_notional:.2f}$ أو السعر غير متاح."),
                ttl_sec=600
            )
            return None, None, 0.0

    # No LIMIT sell here — market only (robust)
    if DRY_RUN:
        px = float(fetch_price(base_symbol) or 0.0)
        return {"id": f"dry_sell_{int(time.time())}", "average": px}, px, qty

    try:
        order = place_market_order(base_symbol, "sell", qty)
    except Exception as e:
        msg = str(e)
        if "51008" in msg or "insufficient" in msg.lower():
            try:
                avail2 = float(fetch_balance(base_symbol.split("/")[0]) or 0.0)
            except Exception:
                avail2 = qty * 0.95
            qty2 = math.floor(max(0.0, min(qty*0.98, avail2)) / step) * step
            if qty2 <= 0 or (price_now > 0 and qty2 * price_now < min_notional):
                _tg_once(f"sell_fail_{base_symbol}", f"❌ بيع متعذّر بعد 51008 — كمية غير كافية/أقل من الحد.", ttl_sec=600)
                return None, None, 0.0
            try:
                order = place_market_order(base_symbol, "sell", qty2)
                qty = qty2
            except Exception:
                _tg_once(f"sell_fail_{base_symbol}", f"❌ فشل بيع {base_symbol} بعد إعادة المحاولة (51008).", ttl_sec=600)
                return None, None, 0.0
        else:
            _tg_once(f"sell_fail_{base_symbol}", f"❌ فشل بيع {base_symbol} (استثناء): {e}", ttl_sec=600)
            return None, None, 0.0

    if not order:
        _tg_once(f"sell_fail_{base_symbol}", f"❌ فشل بيع {base_symbol} (أمر السوق).", ttl_sec=600)
        return None, None, 0.0

    exit_px = float(order.get("average") or order.get("price") or fetch_price(base_symbol) or 0.0)
    return order, exit_px, qty

# ================== Position management ==================

def manage_position(symbol):
    pos = load_position(symbol)
    if not pos:
        return False

    # Backward-compat
    if "amount" not in pos and "qty" in pos:
        try: pos["amount"] = float(pos["qty"]) 
        except Exception: pos["amount"] = float(pos.get("amount", 0.0))

    targets = pos.get("targets") or []
    if targets and not pos.get("tp_hits"): pos["tp_hits"] = [False] * len(targets)

    if "stop_loss" not in pos:
        try:
            base = pos["symbol"].split("#")[0]
            price_now = float(fetch_price(base) or pos.get("entry_price", 0.0))
            data = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
            if data and price_now > 0:
                df = _df(data)
                atr_abs = _atr_from_df(df)
                pos["stop_loss"] = float(max(0.0, price_now - 1.0 * atr_abs)) if atr_abs and atr_abs > 0 else float(pos.get("entry_price", price_now) * 0.97)
            else:
                pos["stop_loss"] = float(pos.get("entry_price", price_now) * 0.97)
        except Exception:
            pos["stop_loss"] = float(pos.get("entry_price", 0.0) * 0.97)
        save_position(symbol, pos)

    base = pos["symbol"].split("#")[0]
    current = float(fetch_price(base))
    entry   = float(pos["entry_price"])
    amount  = float(pos.get("amount", pos.get("qty", 0.0)))
    targets = pos.get("targets") or []
    partials = pos.get("partials") or []
    variant = pos.get("variant", "new")
    mgmt = _mgmt(variant)

    if amount <= 0:
        clear_position(symbol); return False

    # (A) Time-based TP1 exit (Smart)
    max_bars = pos.get("max_bars_to_tp1")
    if max_bars and isinstance(max_bars, int) and max_bars > 0:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            bar_min = _tf_minutes(LTF_TIMEFRAME)
            bars_passed = int((now_riyadh() - opened_at) // timedelta(minutes=bar_min))
        except Exception:
            bars_passed = 0
        if bars_passed >= max_bars and pos.get("tp_hits") and not pos["tp_hits"][0]:
            order, exit_px, sold_qty = _safe_sell(base, amount)
            if order and sold_qty > 0:
                pnl_net = (exit_px - entry) * sold_qty - (entry + exit_px) * sold_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
                p = load_position(symbol) or {}
                p["amount"] = max(0.0, p.get("amount", 0.0) - float(sold_qty))
                save_position(symbol, p); pos = p
                if pos["amount"] <= 0:
                    close_trade(symbol, exit_px, pnl_net, reason="TIME_EXIT"); return True
                else:
                    register_trade_result(pnl_net); return True

    # (B) Max hold hours
    try:
        max_hold_hours = float(pos.get("max_hold_hours") or mgmt.get("TIME_HRS") or 0)
    except Exception:
        max_hold_hours = 0.0
    if max_hold_hours > 0:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            hold_expired = (now_riyadh() - opened_at) >= timedelta(hours=max_hold_hours)
        except Exception:
            hold_expired = False
        if hold_expired:
            order, exit_px, sold_qty = _safe_sell(base, amount)
            if order and sold_qty > 0:
                pnl_net = (exit_px - entry) * sold_qty - (entry + exit_px) * sold_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
                p = load_position(symbol) or {}
                p["amount"] = max(0.0, p.get("amount", 0.0) - float(sold_qty))
                save_position(symbol, p); pos = p
                if pos["amount"] <= 0:
                    close_trade(symbol, exit_px, pnl_net, reason="TIME_HOLD_MAX"); return True
                else:
                    register_trade_result(pnl_net); return True

    # (C) Targets + partials + trailing
    if targets and partials and len(targets) == len(partials):
        current = float(fetch_price(base))
        for i, tp in enumerate(targets):
            if not pos["tp_hits"][i] and current >= tp and pos["amount"] > 0:
                part_qty = pos["amount"] * float(partials[i])
                if part_qty * current < MIN_NOTIONAL_USDT: part_qty = pos["amount"]
                order, exit_px, sold_qty = _safe_sell(base, part_qty)
                if order and sold_qty > 0:
                    pnl_net = (exit_px - entry) * sold_qty - (entry + exit_px) * sold_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
                    p = load_position(symbol) or {}
                    p["amount"] = max(0.0, float(p.get("amount", 0.0)) - float(sold_qty))
                    if p.get("tp_hits") and i < len(p["tp_hits"]): p["tp_hits"][i] = True
                    save_position(symbol, p); pos = p
                    register_trade_result(pnl_net)
                    # Lock SL after TP1
                    if i == 0 and pos["amount"] > 0:
                        lock_sl = entry * (1.0 + float(get_cfg(variant).get("LOCK_MIN_PROFIT_PCT", 0.0)))
                        if lock_sl > pos["stop_loss"]:
                            pos["stop_loss"] = float(lock_sl); save_position(symbol, pos)

    # (D) Trailing after any TP
    if _mgmt(variant).get("TRAIL_AFTER_TP1") and pos["amount"] > 0 and any(pos.get("tp_hits", [])):
        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
        if data_for_atr:
            df_atr = _df(data_for_atr)
            atr_val3 = _atr_from_df(df_atr)
            if atr_val3 and atr_val3 > 0:
                current = float(fetch_price(base))
                new_sl = current - _mgmt(variant).get("TRAIL_ATR", 1.0) * atr_val3
                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                    pos["stop_loss"] = float(new_sl); save_position(symbol, pos)

    # (E) Hard SL
    current = float(fetch_price(base))
    if current <= pos["stop_loss"] and pos["amount"] > 0:
        sellable = float(pos["amount"])
        order, exit_px, sold_qty = _safe_sell(base, sellable)
        if order and sold_qty > 0:
            pnl_gross = (exit_px - entry) * sold_qty
            fees = (entry + exit_px) * sold_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            p = load_position(symbol) or {}
            p["amount"] = max(0.0, p.get("amount", 0.0) - float(sold_qty))
            save_position(symbol, p); pos = p
            if pos["amount"] <= 0:
                close_trade(symbol, exit_px, pnl_net, reason="SL"); return True
            else:
                register_trade_result(pnl_net); return True

    return False

# ================== Risk & reports ==================

def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0, "trades_today": 0,
            "blocked_until": None, "hourly_pnl": {}, "last_signal_ts": None, "relax_success_count": 0}

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str(): s = _default_risk_state(); save_risk_state(s)
    if "hourly_pnl" not in s or not isinstance(s["hourly_pnl"], dict): s["hourly_pnl"] = {}
    if "last_signal_ts" not in s: s["last_signal_ts"] = None
    if "relax_success_count" not in s: s["relax_success_count"] = 0
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)

def register_trade_opened():
    s = load_risk_state(); s["trades_today"] = int(s.get("trades_today", 0)) + 1; save_risk_state(s)

def register_trade_result(pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1
    if pnl_usdt > 0:
        s["relax_success_count"] = int(s.get("relax_success_count", 0)) + 1
        if s["relax_success_count"] >= RELAX_RESET_SUCCESS_TRADES:
            s["relax_success_count"] = 0
            s["last_signal_ts"] = now_riyadh().isoformat(timespec="seconds")
            try: _tg("✅ صفقتان ناجحتان — إلغاء التخفيف (عودة للوضع الطبيعي).")
            except Exception: pass
    else:
        s["relax_success_count"] = 0

    hk = _hour_key(now_riyadh()); s["hourly_pnl"][hk] = float(s["hourly_pnl"].get(hk, 0.0)) + float(pnl_usdt)

    if s["consecutive_losses"] >= _env_int("MAX_CONSEC_LOSSES", 3):
        save_risk_state(s); _set_block(90, reason="خسائر متتالية"); return
    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s); _set_block(minutes, reason="تجاوز حد الخسارة اليومي"); return

    if _env_bool("HOURLY_DD_BLOCK_ENABLE", True):
        try:
            equity = float(fetch_balance("USDT") or 0.0)
            hour_pnl = float(s["hourly_pnl"].get(hk, 0.0))
            HOURLY_DD_PCT = float(os.getenv("HOURLY_DD_PCT", "0.05"))
            if equity > 0 and (hour_pnl <= -abs(HOURLY_DD_PCT) * equity):
                save_risk_state(s); _set_block(60, reason=f"هبوط {HOURLY_DD_PCT*100:.1f}%/ساعة"); return
        except Exception:
            pass
    save_risk_state(s)

def _set_block(minutes, reason="risk"):
    s = load_risk_state(); until = now_riyadh() + timedelta(minutes=minutes)
    s["blocked_until"] = until.isoformat(timespec="seconds"); save_risk_state(s)
    _tg(f"⛔️ <b>حظر مؤقت</b> ({reason}) حتى <code>{until.strftime('%H:%M')}</code>.")

# Reports (compact)

def build_daily_report_text():
    closed = load_closed_positions(); today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    s = load_risk_state()

    def f6(x): 
        try: return "{:,.6f}".format(float(x))
        except Exception: return str(x)
    def f2(x): 
        try: return "{:,.2f}".format(float(x))
        except Exception: return str(x)

    if not todays:
        extra = (f"\nوضع المخاطر: {('محظور' if s.get('blocked_until') else 'سماح')} • صفقات اليوم: {int(s.get('trades_today', 0))}"
                 f" • PnL اليومي: {float(s.get('daily_pnl', 0.0)):.2f}$")
        return f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم.{extra}"

    total_pnl = sum(float(t.get("profit", 0.0)) for t in todays)
    wins = [t for t in todays if float(t.get("profit", 0.0)) > 0]
    win_rate = round(100 * len(wins) / max(1, len(todays)), 2)

    headers = ["الرمز#النسخة", "الكمية", "دخول", "خروج", "P/L$", "P/L%", "Score", "نمط", "سبب", "TP_hits", "Exit"]
    rows = []
    for t in todays:
        tp_hits = []
        for i in range(1, 8):
            if t.get(f"tp{i}_hit"): tp_hits.append(f"T{i}")
        tp_str = ",".join(tp_hits) if tp_hits else "-"
        pnl_pct = ((float(t.get('exit_price',0))/float(t.get('entry_price',1)))-1.0) if float(t.get('entry_price',0))>0 else 0.0
        rows.append([
            t.get("symbol", "-"), f6(t.get('amount', 0)), f6(t.get('entry_price', 0)), f6(t.get('exit_price', 0)),
            f2(t.get('profit', 0)), f"{round(float(pnl_pct)*100, 2)}%",
            str(t.get("score", "-")), t.get("pattern", "-"),
            (t.get("entry_reason", t.get("reason", "-"))[:40] + ("…" if len(str(t.get('entry_reason', t.get('reason', '')))) > 40 else "")),
            tp_str, t.get("reason", "-")
        ])

    # Simple monospaced table
    def _fmt_table(rows, headers):
        widths = [len(str(h)) for h in headers]
        for r in rows:
            for i, c in enumerate(r): widths[i] = max(widths[i], len(str(c)))
        def fmt_row(r): return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
        return "<pre>" + fmt_row(headers) + "\n" + "\n".join(fmt_row(r) for r in rows) + "</pre>"

    table = _fmt_table(rows, headers)
    summary = (f"📊 <b>تقرير اليوم {today}</b>\n" 
               f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b> • نسبة الفوز: <b>{win_rate}%</b>\n")
    return summary + table

# ================== Close & archive ==================

def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos: return
    closed = load_closed_positions()

    entry = float(pos.get("entry_price", 0.0))
    amount = float(pos.get("amount", 0.0))
    pnl_pct = ((float(exit_price) / entry) - 1.0) if entry else 0.0

    tp_hits = {}
    try:
        if "targets" in pos and "tp_hits" in pos and isinstance(pos["tp_hits"], list):
            for i, hit in enumerate(pos["tp_hits"], start=1): tp_hits[f"tp{i}_hit"] = bool(hit)
    except Exception:
        pass

    closed.append({
        "symbol": pos.get("symbol", symbol),
        "entry_price": float(entry),
        "exit_price": float(exit_price),
        "amount": float(amount),
        "profit": float(pnl_net),
        "pnl_pct": round(float(pnl_pct), 6),
        "reason": reason,
        "opened_at": pos.get("opened_at"),
        "closed_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": pos.get("variant"),
        "score": pos.get("score"),
        "pattern": pos.get("pattern"),
        "entry_reason": pos.get("reason"),
        **tp_hits
    })
    save_closed_positions(closed)
    clear_position(symbol)

# ================== Diagnostics ==================

def get_last_reject(symbol: str):
    if symbol in _LAST_REJECT: return _LAST_REJECT[symbol]
    base, variant = _split_symbol_variant(symbol)
    key1 = f"{base}|{variant}"
    if key1 in _LAST_REJECT: return _LAST_REJECT[key1]
    if base in _LAST_REJECT: return _LAST_REJECT[base]
    return None

def check_signal_debug(symbol: str):
    r = check_signal(symbol)
    if isinstance(r, dict) and r.get("decision") == "buy":
        return r, ["buy_ok"]
    last = get_last_reject(symbol)
    if last:
        stg = last.get("stage", "no_buy"); det = last.get("details", {})
        return None, [f"{stg}:{det}"]
    return None, ["no_buy"]
