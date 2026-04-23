# -*- coding: utf-8 -*-
# strategy.py - Spot-only (v4.2 — إصلاح الغبار النهائي + تطوير الأداء)
# الإصلاحات في هذه النسخة:
# 1. [FIX] _safe_sell: لوج تشخيصي كامل + تحقق دقيق من avail
# 2. [FIX] حساب الكمية مع stepSize (منع التقريب لصفر)
# 3. [FIX] fetch_balance مع retry وتفصيل الأخطاء
# 4. [FIX] تحقق من min_notional قبل وبعد التقريب
# 5. [FIX] fallback لسعر العملة إذا فشل fetch_price
# 6. [IMPROVE] لوج موحّد لكل عمليات البيع

from __future__ import annotations

import os, json, requests, logging, time, math, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from okx_api import (
    fetch_ohlcv,
    fetch_price,
    place_market_order,
    fetch_balance,
    fetch_symbol_filters,
)

# [NEW v4.2] العقل المدبر
try:
    from market_brain import get_brain_directives as _get_brain_directives
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    def _get_brain_directives():
        return {"entry_allowed": True, "score_threshold_override": None,
                "size_multiplier": 1.0, "max_open_positions_override": None,
                "blocked_patterns": [], "preferred_modes": [], "notes": []}

from config import (
    TRADE_AMOUNT_USDT,
    MAX_OPEN_POSITIONS,
    SYMBOLS,
    FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    STRAT_LTF_TIMEFRAME,
    STRAT_HTF_TIMEFRAME,
    MIN_NOTIONAL_USDT,
)

# ==== ATR & RVOL dynamic defaults ====
ATR_NEED_BASE   = float(os.getenv("ATR_NEED_BASE",   0.0022))
ATR_BR_WEAK     = float(os.getenv("ATR_BR_WEAK",     0.45))
ATR_BR_STRONG   = float(os.getenv("ATR_BR_STRONG",   0.65))
ATR_NEED_WEAK   = float(os.getenv("ATR_NEED_WEAK",   0.0028))
ATR_NEED_STRONG = float(os.getenv("ATR_NEED_STRONG", 0.0018))

RVOL_BASE              = float(os.getenv("RVOL_BASE",              1.15))
RVOL_RELAX_FOR_LEADERS = float(os.getenv("RVOL_RELAX_FOR_LEADERS", 0.90))
RVOL_RELAX_BR_MIN      = float(os.getenv("RVOL_RELAX_BR_MIN",      0.55))

LEADERS_ENV = os.getenv("LEADERS", "BTC,ETH,SOL,BNB,LINK")
LEADERS_SET = {s.strip().upper() for s in LEADERS_ENV.split(",") if s.strip()}

# ===================== ENV helpers =====================
def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def _env_float(name, default):
    try:    return float(os.getenv(name, str(default)))
    except: return float(default)

def _env_int(name, default):
    try:    return int(float(os.getenv(name, str(default))))
    except: return int(default)

def _env_str(name, default=""):
    v = os.getenv(name)
    return default if v is None else str(v)

def get_usdt_free():
    try:    return float(fetch_balance("USDT") or 0.0)
    except: return 0.0

def _round_to_tick(px, tick):
    if tick <= 0: return float(px)
    return math.floor(float(px) / tick) * tick

# ===== Strategy logger =====
logger = logging.getLogger("strategy")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEBUG_LOG_SIGNALS = _env_bool("DEBUG_LOG_SIGNALS", False)

def _print(s):
    try:    print(s, flush=True)
    except: pass

# ================== Constants ==================
RIYADH_TZ          = timezone(timedelta(hours=3))
POSITIONS_DIR       = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE     = "risk_state.json"
DUST_LOG_FILE       = "dust_cleaned.json"

try:    os.makedirs(POSITIONS_DIR, exist_ok=True)
except: pass

STRAT_TG_SEND = _env_bool("STRAT_TG_SEND", False)
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME

# ===== Indicator windows =====
EMA_FAST, EMA_SLOW = 9, 21
EMA_TREND, EMA_LONG = 50, 200
VOL_MA    = 20
SR_WINDOW = 50
ATR_PERIOD = 14

RVOL_WINDOW_FAST = _env_int("RVOL_WINDOW_FAST", 24)
RVOL_WINDOW_SLOW = _env_int("RVOL_WINDOW_SLOW", 30)
RVOL_BLEND       = _env_float("RVOL_BLEND", 0.55)

NR_WINDOW = 10
NR_FACTOR = 0.75

HTF_EMA_TREND_PERIOD = 50
HTF_SR_WINDOW = 50

# ===== Trade management =====
TP1_FRACTION       = 0.5
TRAIL_MIN_STEP_RATIO = 0.001

MAX_TRADES_PER_DAY    = _env_int("MAX_TRADES_PER_DAY",    15)
MAX_CONSEC_LOSSES     = _env_int("MAX_CONSEC_LOSSES",      4)
DAILY_LOSS_LIMIT_USDT = _env_float("DAILY_LOSS_LIMIT_USDT", 40.0)
MAX_OPEN_POSITIONS    = _env_int("MAX_OPEN_POSITIONS",     5)

TRADE_BASE_USDT   = _env_float("TRADE_BASE_USDT",   25.0)
MIN_TRADE_USDT    = _env_float("MIN_TRADE_USDT",    10.0)
MIN_NOTIONAL_USDT = _env_float("MIN_NOTIONAL_USDT",  5.0)
DRY_RUN           = _env_bool("DRY_RUN", False)

# ===== Early Scout =====
EARLY_SCOUT_ENABLE       = _env_bool("EARLY_SCOUT_ENABLE", True)
EARLY_SCOUT_SIZE_MULT    = _env_float("EARLY_SCOUT_SIZE_MULT", 0.35)
EARLY_SCOUT_SCORE_MIN    = _env_int("EARLY_SCOUT_SCORE_MIN",   30)
EARLY_SCOUT_MAX_ATR_DIST = _env_float("EARLY_SCOUT_MAX_ATR_DIST", 0.5)
EARLY_SCOUT_BR_MIN       = _env_float("EARLY_SCOUT_BR_MIN", 0.55)

# ===== Feature flags =====
USE_EMA200_TREND_FILTER   = _env_bool("USE_EMA200_TREND_FILTER",   True)
ENABLE_GOLDEN_CROSS_ENTRY = _env_bool("ENABLE_GOLDEN_CROSS_ENTRY", True)
USE_EMA100_LTF_FILTER     = _env_bool("USE_EMA100_LTF_FILTER",     True)
GOLDEN_CROSS_RVOL_BOOST   = _env_float("GOLDEN_CROSS_RVOL_BOOST",  1.10)

# ===== Scoring =====
SCORE_THRESHOLD = _env_int("SCORE_THRESHOLD", 50)

# ===== Exhaustion filter =====
EXH_RSI_MAX        = _env_float("EXH_RSI_MAX",        78.0)  # [FIX v4.2] رُفع من 75
EXH_EMA50_DIST_PCT = _env_float("EXH_EMA50_DIST_PCT",  0.08)
EXH_EMA50_DIST_ATR = _env_float("EXH_EMA50_DIST_ATR",  3.5)  # [FIX v4.2] رُفع من 2.8

# ===== Break-even =====
BREAKEVEN_ENABLE      = _env_bool("BREAKEVEN_ENABLE", True)
BREAKEVEN_TRIGGER_ATR = _env_float("BREAKEVEN_TRIGGER_ATR", 0.3)
BREAKEVEN_OFFSET_PCT  = _env_float("BREAKEVEN_OFFSET_PCT",  0.002)

# ===== Aggressive Mode =====
AGGR_MODE_ENABLE     = _env_bool("AGGR_MODE_ENABLE",    True)
AGGR_SCORE_MIN       = _env_int("AGGR_SCORE_MIN",        65)
AGGR_SCORE_STRONG    = _env_int("AGGR_SCORE_STRONG",     75)
AGGR_MAX_RISK_MULT   = _env_float("AGGR_MAX_RISK_MULT",  1.8)
AGGR_BREAKOUT_ONLY   = _env_bool("AGGR_BREAKOUT_ONLY",  True)

# ===== Market Sizing =====
LEADER_SIZE_MULT      = _env_float("LEADER_SIZE_MULT",      0.80)
LEADER_DONT_DOWNSCALE = _env_bool("LEADER_DONT_DOWNSCALE", False)
MAX_SYMBOL_EXPOSURE_MULT = _env_float("MAX_SYMBOL_EXPOSURE_MULT", 1.6)
ADDON_BREAKOUT_MULT      = _env_float("ADDON_BREAKOUT_MULT",      0.30)
SCOUT_TO_FULL_MIN_SCORE  = _env_int("SCOUT_TO_FULL_MIN_SCORE", SCORE_THRESHOLD)

# ===== Auto-Relax =====
AUTO_RELAX_AFTER_HRS_1     = _env_float("AUTO_RELAX_AFTER_HRS_1",    6)
AUTO_RELAX_AFTER_HRS_2     = _env_float("AUTO_RELAX_AFTER_HRS_2",   12)
RELAX_RVOL_DELTA_1         = _env_float("RELAX_RVOL_DELTA_1",      0.05)
RELAX_RVOL_DELTA_2         = _env_float("RELAX_RVOL_DELTA_2",      0.10)
RELAX_ATR_MIN_SCALE_1      = _env_float("RELAX_ATR_MIN_SCALE_1",   0.90)
RELAX_ATR_MIN_SCALE_2      = _env_float("RELAX_ATR_MIN_SCALE_2",   0.85)
RELAX_RESET_SUCCESS_TRADES = _env_int("RELAX_RESET_SUCCESS_TRADES",   2)

# ===== Breadth =====
BREADTH_MIN_RATIO   = _env_float("BREADTH_MIN_RATIO", 0.55)
BREADTH_TF          = os.getenv("BREADTH_TF", "1h")
BREADTH_TTL_SEC     = _env_int("BREADTH_TTL_SEC", 180)
BREADTH_SYMBOLS_ENV = os.getenv("BREADTH_SYMBOLS", "")

# ===== Soft schedule =====
SOFT_SCHEDULE_ENABLE    = _env_bool("SOFT_SCHEDULE_ENABLE",  False)
SOFT_SCHEDULE_HRS       = _env_str("SOFT_SCHEDULE_HRS",  "09:30-16:00")
SOFT_SCHEDULE_WEEKDAYS  = _env_str("SOFT_SCHEDULE_WEEKDAYS", "")
SOFT_SCALE_TIME_ONLY    = _env_float("SOFT_SCALE_TIME_ONLY",    0.80)
SOFT_SCALE_MARKET_WEAK  = _env_float("SOFT_SCALE_MARKET_WEAK",  0.85)
SOFT_SEVERITY_STEP      = _env_float("SOFT_SEVERITY_STEP",      0.10)
SOFT_MSG_ENABLE         = _env_bool("SOFT_MSG_ENABLE", True)
SOFT_BREADTH_ENABLE     = _env_bool("SOFT_BREADTH_ENABLE",  True)
SOFT_BREADTH_SIZE_SCALE = _env_float("SOFT_BREADTH_SIZE_SCALE", 0.5)

# ===== Multi-targets =====
ENABLE_MULTI_TARGETS = _env_bool("ENABLE_MULTI_TARGETS", True)
MAX_TP_COUNT         = _env_int("MAX_TP_COUNT", 5)
TP_ATR_MULTS_TREND   = tuple(float(x) for x in os.getenv(
    "TP_ATR_MULTS_TREND", "1.2,2.2,3.5,4.5,6.0").split(","))
TP_ATR_MULTS_VBR     = tuple(float(x) for x in os.getenv(
    "TP_ATR_MULTS_VBR", "0.6,1.2,1.8,2.4").split(","))

# ===== Dynamic max bars =====
USE_DYNAMIC_MAX_BARS = _env_bool("USE_DYNAMIC_MAX_BARS", True)
MAX_BARS_BASE        = _env_int("MAX_BARS_TO_TP1_BASE",  12)

# ===== Tunables =====
MIN_BAR_NOTIONAL_USD = _env_float("MIN_BAR_NOTIONAL_USD",    25000)
ATR_MIN_BASE         = _env_float("ATR_MIN_FOR_TREND_BASE",  0.0020)
ATR_MIN_NEW          = _env_float("ATR_MIN_FOR_TREND_NEW",   0.0026)
ATR_MIN_BRT          = _env_float("ATR_MIN_FOR_TREND_BRT",   0.0022)
RVOL_MIN_NEW         = _env_float("RVOL_MIN_NEW", 1.25)
RVOL_MIN_BRT         = _env_float("RVOL_MIN_BRT", 1.30)

# ===== Caches =====
_HTF_CACHE:   Dict[str, Dict[str, Any]] = {}
_HTF_TTL_SEC  = _env_int("HTF_CACHE_TTL_SEC", 150)
_OHLCV_CACHE: Dict[tuple, list] = {}
_METRICS = {
    "ohlcv_api_calls":    0,
    "ohlcv_cache_hits":   0,
    "ohlcv_cache_misses": 0,
    "htf_cache_hits":     0,
    "htf_cache_misses":   0,
}
_REJ_COUNTS  = {"atr_low": 0, "rvol_low": 0, "notional_low": 0}
_REJ_SUMMARY: Dict[str, int] = {}

# ===== MTF strict =====
try:    ENABLE_MTF_STRICT
except: ENABLE_MTF_STRICT = _env_bool("ENABLE_MTF_STRICT", False)

# ================== Telegram ==================
def _tg(text, parse_mode="HTML"):
    if not STRAT_TG_SEND: return False
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return False
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": str(TELEGRAM_CHAT_ID), "text": text,
                  "disable_web_page_preview": True,
                  **({"parse_mode": parse_mode} if parse_mode else {})},
            timeout=10)
        return r.ok
    except: return False

_TG_ONCE_CACHE = {}
def _tg_once(key, text, ttl_sec=600):
    if not STRAT_TG_SEND: return False
    now_s = int(time.time())
    if (now_s - _TG_ONCE_CACHE.get(key, 0)) < ttl_sec: return False
    _TG_ONCE_CACHE[key] = now_s
    try: _tg(text); return True
    except: return False

# ================== Storage utils ==================
def now_riyadh(): return datetime.now(RIYADH_TZ)
def _today_str(): return now_riyadh().strftime("%Y-%m-%d")
def _hour_key(dt): return dt.strftime("%Y-%m-%d %H")

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
    except: pass
    return default

def _df(data):
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    try:
        if len(df) and df["timestamp"].iloc[-1] < 10**12:
            df["timestamp"] = df["timestamp"] * 1000
    except: pass
    for c in ("open","high","low","close","volume"):
        try: df[c] = pd.to_numeric(df[c], errors="coerce")
        except: pass
    return df

def _finite_or(default, *vals):
    for v in vals:
        try:
            f = float(v)
            if math.isfinite(f): return f
        except: pass
    return default

def _split_symbol_variant(symbol):
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower().strip()
        if variant in ("srr+","srrplus","srr_plus"): variant = "srr_plus"
        elif variant not in ("old","new","srr","brt","vbr","srr_plus","alpha"):
            variant = "new"
        return base, variant
    return symbol, "new"

# ================== Positions ==================
def _pos_path(symbol):
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return f"{POSITIONS_DIR}/{symbol.replace('/','_')}.json"

def load_position(symbol):    return _read_json(_pos_path(symbol), None)
def save_position(symbol, p): _atomic_write(_pos_path(symbol), p)
def clear_position(symbol):
    try:
        p = _pos_path(symbol)
        if os.path.exists(p): os.remove(p)
    except: pass

def count_open_positions():
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions():     return _read_json(CLOSED_POSITIONS_FILE, [])
def save_closed_positions(lst):  _atomic_write(CLOSED_POSITIONS_FILE, lst)

# ================== Indicators ==================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    gain = d.where(d > 0, 0.0)
    loss = -d.where(d < 0, 0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    return 100 - (100 / (1 + ag/al))

def macd_cols(df, fast=12, slow=26, signal=9):
    df["ema_fast"]   = ema(df["close"], fast)
    df["ema_slow"]   = ema(df["close"], slow)
    df["macd"]       = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"]= df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]  = df["macd"] - df["macd_signal"]
    return df

def add_indicators(df):
    df["ema9"]   = ema(df["close"], EMA_FAST)
    df["ema21"]  = ema(df["close"], EMA_SLOW)
    df["ema50"]  = ema(df["close"], EMA_TREND)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], EMA_LONG)
    df["rsi"]    = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA, min_periods=1).mean()
    df = macd_cols(df)
    return df

def _ensure_ltf_indicators(df):
    df = add_indicators(df.copy())
    ts = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Riyadh")
    day_changed = ts.dt.date
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tpv"]     = tp * df["volume"]
    df["cum_vol"] = df.groupby(day_changed, sort=False)["volume"].cumsum()
    df["cum_tpv"] = df.groupby(day_changed, sort=False)["tpv"].cumsum()
    df["vwap"]    = df["cum_tpv"] / df["cum_vol"].replace(0, np.nan)
    df["vwap"]    = df.groupby(day_changed, sort=False)["vwap"].transform(
                        lambda s: s.ffill().bfill())
    vol_ma_f = df["volume"].rolling(RVOL_WINDOW_FAST,
                   min_periods=max(1, RVOL_WINDOW_FAST//3)).mean()
    vol_ma_s = df["volume"].rolling(RVOL_WINDOW_SLOW,
                   min_periods=max(1, RVOL_WINDOW_SLOW//3)).mean()
    df["rvol_fast"] = df["volume"] / vol_ma_f.replace(0, 1e-9)
    df["rvol_slow"] = df["volume"] / vol_ma_s.replace(0, 1e-9)
    df["rvol"]      = df["rvol_fast"] * RVOL_BLEND + df["rvol_slow"] * (1.0 - RVOL_BLEND)
    rng    = df["high"] - df["low"]
    rng_ma = rng.rolling(NR_WINDOW, min_periods=max(3, NR_WINDOW//2)).mean()
    df["is_nr"]     = rng < (NR_FACTOR * rng_ma)
    df["body"]      = (df["close"] - df["open"]).abs()
    df["avg_body20"]= df["body"].rolling(20, min_periods=5).mean()
    return df

def _atr_from_df(df, period=ATR_PERIOD):
    if len(df) < max(5, period + 2): return float("nan")
    c  = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-c).abs(),
        (df["low"]-c).abs()
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-2])

def atr(h, l, c, period=14):
    h, l, c = pd.Series(h), pd.Series(l), pd.Series(c)
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(),(h-prev_c).abs(),(l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ===== Pattern helpers =====
def _bullish_engulf(prev, cur):
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and \
           (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])

def _sweep_then_reclaim(df, prev, closed, ref_val, lookback=20, tol=0.0012):
    try:
        if ref_val is None or len(df) < lookback + 5: return False
        lows_w  = df["low"].iloc[-(lookback+2):-2]
        if lows_w.empty: return False
        ll = float(lows_w.min())
        swept    = float(closed["low"]) <= ll * (1.0 - tol) or float(closed["low"]) <= ll
        reclaimed= float(closed["close"]) >= float(ref_val)
        bullish  = (float(closed["close"]) > float(closed["open"])) or _bullish_engulf(prev, closed)
        return bool(swept and reclaimed and bullish)
    except: return False

def _swing_points(df, left=2, right=2):
    highs, lows = df["high"], df["low"]
    idx = len(df) - 3
    sh = sl = None
    for i in range(max(0, idx-10), idx+1):
        if i-left < 0 or i+right >= len(df): continue
        if highs[i] == max(highs[i-left:i+right+1]): sh = float(highs[i])
        if lows[i]  == min(lows[i-left:i+right+1]):  sl = float(lows[i])
    return sh, sl

# ================== Exhaustion filter ==================
def _is_exhausted(closed, atr_val=None):
    """
    [FIX v4.2] فلتر الاستنزاف يعمل فقط عند الصعود (السعر فوق EMA50).
    إذا السعر تحت EMA50 → السوق هابط وليس مستنزفاً صعوداً → لا نرفضه هنا.
    """
    try:
        rsi_v = float(closed.get("rsi", 50))
        close_v = float(closed["close"])
        ema50_v = closed.get("ema50")

        # تحقق من الاتجاه أولاً
        above_ema50 = (
            ema50_v is not None and
            math.isfinite(float(ema50_v)) and
            float(ema50_v) > 0 and
            close_v > float(ema50_v)
        )

        # RSI مرتفع جداً → استنزاف صعودي (يُطبَّق دائماً)
        if rsi_v > EXH_RSI_MAX and above_ema50:
            return True, "rsi_high"

        # فلاتر المسافة → فقط إذا كان السعر فوق EMA50 (صعود مفرط)
        if above_ema50 and ema50_v is not None:
            dist_pct = (close_v - float(ema50_v)) / close_v  # موجب فقط
            if dist_pct > EXH_EMA50_DIST_PCT:
                return True, "ema50_far"
            if atr_val is not None and atr_val > 0:
                dist_atr = (close_v - float(ema50_v)) / atr_val  # موجب فقط
                if dist_atr > EXH_EMA50_DIST_ATR:
                    return True, "atr_far"

    except:
        pass
    return False, None

# ================== Opportunity Score ==================
def _opportunity_score(df, prev, closed):
    score, why, pattern = 0, [], ""
    try:
        close_v = float(closed["close"])
        open_v  = float(closed["open"])
        rvol    = float(closed.get("rvol", 0) or 0)

        ema50_v  = closed.get("ema50")
        ema100_v = closed.get("ema100")
        ema200_v = closed.get("ema200")

        above_50  = ema50_v  is not None and close_v > float(ema50_v)
        above_100 = ema100_v is not None and close_v > float(ema100_v)
        above_200 = ema200_v is not None and close_v > float(ema200_v)

        if above_50 and above_100:
            score += 12; why.append("AboveEMA50+100")
        elif above_50:
            score += 6;  why.append("AboveEMA50")
        if above_200:
            score += 8;  why.append("AboveEMA200")

        if rvol >= 2.0:
            score += 25; why.append("StrongRVOL≥2.0")
        elif rvol >= 1.5:
            score += 15; why.append("HighRVOL≥1.5")
        elif rvol >= 1.2:
            score += 8;  why.append("RVOL≥1.2")

        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_range  = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        is_engulf = _bullish_engulf(prev, closed)
        is_nr_bo  = nr_recent and (close_v > hi_range)

        if is_nr_bo:
            score += 25; why.append("NR_Breakout"); pattern = "NR_Breakout"
        elif is_engulf:
            score += 20; why.append("BullishEngulf"); pattern = pattern or "BullishEngulf"

        try:
            rng  = max(float(closed["high"]) - float(closed["low"]), 1e-9)
            body = abs(close_v - open_v)
            if body / rng >= 0.65 and close_v > open_v:
                score += 8; why.append("StrongBody≥65%")
        except: pass

        try:
            atr_avg = float(
                (df["high"]-df["low"]).rolling(14, min_periods=5).mean().iloc[-2])
            for ref_key in ("ema50","vwap","ema21"):
                ref = closed.get(ref_key)
                if ref is not None and math.isfinite(float(ref)):
                    if abs(close_v - float(ref)) <= 0.3 * atr_avg:
                        score += 10; why.append(f"NearValue({ref_key})"); break
        except: pass

        try:
            rsi_v = float(closed.get("rsi", 50))
            if 50 <= rsi_v <= 68:
                score += 10; why.append(f"RSI_Healthy({rsi_v:.0f})")
            elif 68 < rsi_v <= 75:
                score += 4;  why.append(f"RSI_Warm({rsi_v:.0f})")
        except: pass

    except: pass

    return score, ", ".join(why), (pattern or "Generic")


# ================== SR / HTF helpers ==================
def get_sr_on_closed(df, window=SR_WINDOW):
    if len(df) < window + 3: return None, None
    dp = df.iloc[:-1]; w = min(window, len(dp))
    res = dp["high"].rolling(w, min_periods=max(5,w//3)).max().iloc[-1]
    sup = dp["low"].rolling(w,  min_periods=max(5,w//3)).min().iloc[-1]
    if pd.isna(res) or pd.isna(sup): return None, None
    return float(sup), float(res)

def recent_swing(df, lookback=60):
    if len(df) < lookback+5: return None, None
    seg = df.iloc[-(lookback+1):-1]
    hhv, llv = seg["high"].max(), seg["low"].min()
    if pd.isna(hhv) or pd.isna(llv) or hhv <= llv: return None, None
    return float(hhv), float(llv)

def _rolling_sr(symbol, tf, window, bars=300):
    data = get_ohlcv_cached(symbol, tf, bars)
    if not data: return None, None
    df = _df(data)
    if len(df) < window+3: return None, None
    dp = df.iloc[:-1]; w = min(window, len(dp))
    res = dp["high"].rolling(w, min_periods=max(5,w//3)).max().iloc[-1]
    sup = dp["low"].rolling(w,  min_periods=max(5,w//3)).min().iloc[-1]
    if pd.isna(res) or pd.isna(sup): return None, None
    return float(sup), float(res)

SR_LEVELS_CFG = [
    ("LTF_H1","1h",50,0.8),
    ("HTF_H4","4h",50,1.2),
    ("HTF_D1","1d",30,1.5),
]
def get_sr_multi(symbol):
    levels = {}
    base = symbol.split("#")[0]
    for name, tf, window, nm in SR_LEVELS_CFG:
        try:
            sup, res = _rolling_sr(base, tf, window)
            if sup or res:
                levels[name] = {"tf":tf,"support":sup,"resistance":res,"near_mult":nm}
        except: pass
    return levels

def macd_rsi_gate(prev_row, closed_row, policy):
    if not policy: return True
    rsi_now = float(closed_row.get("rsi", 50.0))
    rsi_up  = rsi_now > float(prev_row.get("rsi", rsi_now))
    mh_now  = float(closed_row.get("macd_hist", 0.0))
    mh_prev = float(prev_row.get("macd_hist", 0.0))
    macd_pos= mh_now > 0; macd_up = mh_now > mh_prev
    flags   = []
    if rsi_now > 50:  flags.append("RSI>50")
    if rsi_up:        flags.append("RSI↑")
    if macd_pos:      flags.append("MACD_hist>0")
    if macd_up:       flags.append("MACD_hist↑")
    k = len(flags)
    if policy == "lenient": return k >= 1
    if policy == "strict":  return ("RSI>50" in flags and "MACD_hist>0" in flags and "MACD_hist↑" in flags)
    return k >= 2

# ================== OHLCV cache ==================
def reset_cycle_cache():
    _OHLCV_CACHE.clear()
    for k in _METRICS: _METRICS[k] = 0
    for k in _REJ_COUNTS: _REJ_COUNTS[k] = 0

def metrics_snapshot(): return dict(_METRICS)
def metrics_format():
    m = _METRICS
    return (
        "📈 <b>Metrics (this round)</b>\n"
        f"- OHLCV cache hits/misses: <b>{m['ohlcv_cache_hits']}/{m['ohlcv_cache_misses']}</b>\n"
        f"- OHLCV API calls: <b>{m['ohlcv_api_calls']}</b>\n"
        f"- HTF hits/misses: <b>{m['htf_cache_hits']}/{m['htf_cache_misses']}</b>"
    )

def _retry_fetch_ohlcv(symbol, tf, bars, attempts=5, base_wait=1.5, max_wait=12.0):
    last_exc = None
    for i in range(attempts):
        try:
            return fetch_ohlcv(symbol, tf, bars)
        except Exception as e:
            last_exc = e
            msg  = str(e)
            boost= 2.5 if ("50011" in msg or "Too Many" in msg) else 1.0
            wait = min(max_wait, base_wait*(2**i)*boost) * (0.9+0.3*np.random.rand())
            time.sleep(wait)
    if last_exc: raise last_exc
    return None

def api_fetch_ohlcv(symbol, tf, bars):
    _METRICS["ohlcv_api_calls"] += 1
    return _retry_fetch_ohlcv(symbol, tf, bars)

def get_ohlcv_cached(symbol, tf, bars):
    key = (symbol, tf, bars)
    if key in _OHLCV_CACHE:
        _METRICS["ohlcv_cache_hits"] += 1
        return _OHLCV_CACHE[key]
    _METRICS["ohlcv_cache_misses"] += 1
    data = api_fetch_ohlcv(symbol, tf, bars)
    if data: _OHLCV_CACHE[key] = data
    return data

_TF_MIN = {"5m":5,"15m":15,"30m":30,"45m":45,"1h":60,"2h":120,"4h":240,"1d":1440}
def _tf_minutes(tf): return _TF_MIN.get(tf.lower(), 60)

def _row_is_recent_enough(df, tf, bars_back=2):
    try:
        last_ts = int(df["timestamp"].iloc[-bars_back])
        if last_ts < 10**12: last_ts *= 1000
        return (int(time.time()*1000) - last_ts) <= (2*_tf_minutes(tf)*60*1000)
    except: return False

def _get_ltf_df_with_fallback(symbol, tf=None):
    tf = tf or STRAT_LTF_TIMEFRAME
    for bars in (140, 120, 100, 80):
        try:
            data = get_ohlcv_cached(symbol, tf, bars)
            if not data or len(data) < 60: continue
            df = _df(data)
            if not _row_is_recent_enough(df, tf, bars_back=2): continue
            df = _ensure_ltf_indicators(df)
            if len(df) >= 60: return df
        except: continue
    return None

# ================== HTF context ==================
def _ensure_htf_indicators(df):
    try:
        if "ema21" not in df: df["ema21"] = df["close"].ewm(span=21,adjust=False).mean()
        if "ema50" not in df: df["ema50"] = df["close"].ewm(span=50,adjust=False).mean()
        if "rsi14" not in df:
            delta = df["close"].diff()
            g = delta.where(delta>0,0.0).ewm(alpha=1/14,adjust=False).mean()
            l = (-delta.where(delta<0,0.0)).ewm(alpha=1/14,adjust=False).mean()
            df["rsi14"] = 100-(100/(1+g/l.replace(0,1e-9)))
    except: pass
    return df

def _htf_gate(base, *args, **kwargs):
    rule = {}
    if args:
        if isinstance(args[0], dict): rule.update(args[0])
        elif isinstance(args[0], str): rule["tf"] = args[0]
    rule.update(kwargs or {})
    tf   = (rule.get("tf") or "4h").lower()
    ind  = (rule.get("ind") or "ema21").lower()
    dire = (rule.get("dir") or "above").lower()
    bars = int(rule.get("bars") or 1)
    fail_open = bool(rule.get("fail_open", True))
    tf_map = {"h1":"1h","1h":"1h","h4":"4h","4h":"4h","d1":"1d","1d":"1d"}
    try:
        raw = get_ohlcv_cached(base, tf_map.get(tf,tf), 120)
        if not raw or len(raw) < max(30, bars+2):
            return True if fail_open else False
        df = _df(raw); df = _ensure_htf_indicators(df)
        closes = df["close"]
        if ind in ("ema21","ema50","ema200"):
            ema_col = ind if ind in df.columns else None
            if not ema_col and ind == "ema200":
                df["ema200"] = df["close"].ewm(span=200,adjust=False).mean()
                ema_col = "ema200"
            ema_vals = df[ema_col]
            for k in range(2, 2+bars):
                c = float(closes.iloc[-k]); e = float(ema_vals.iloc[-k])
                if dire == "above":
                    if not (c >= e): return False
                else:
                    if not (c <= e): return False
            return True
        return True
    except:
        return True if fail_open else False

def _get_htf_context(symbol):
    base = symbol.split("#")[0]
    now  = now_riyadh()
    ent  = _HTF_CACHE.get(base)
    if ent and (now - ent["t"]).total_seconds() <= _HTF_TTL_SEC:
        _METRICS["htf_cache_hits"] += 1
        return ent["ctx"]
    _METRICS["htf_cache_misses"] += 1
    data = get_ohlcv_cached(base, STRAT_HTF_TIMEFRAME, 200)
    if not data: return None
    df = _df(data); df["ema50_htf"] = ema(df["close"], HTF_EMA_TREND_PERIOD)
    if len(df) < HTF_SR_WINDOW+3: return None
    dp = df.iloc[:-2]; w = min(HTF_SR_WINDOW, len(dp))
    res = _finite_or(None, dp["high"].rolling(w).max().iloc[-1])
    sup = _finite_or(None, dp["low"].rolling(w).min().iloc[-1])
    closed_row  = df.iloc[-2]
    ema_series  = ema(df["close"], HTF_EMA_TREND_PERIOD)
    ema_now     = _finite_or(float(closed_row["close"]),
                             ema_series.iloc[-2] if len(ema_series)>=2 else None)
    ema_prev    = _finite_or(ema_now,
                             ema_series.iloc[-7] if len(ema_series)>=7 else None)
    ctx = {"close": float(closed_row["close"]),
           "ema50_now": float(ema_now), "ema50_prev": float(ema_prev),
           "support": sup, "resistance": res, "mtf": {}}
    _HTF_CACHE[base] = {"t": now, "ctx": ctx}
    return ctx


# ================== Breadth ==================
_BREADTH_CACHE = {"t":0.0,"ratio":None}

def _breadth_refs():
    if BREADTH_SYMBOLS_ENV.strip():
        return [s.strip().replace("-","/").upper().split("#")[0]
                for s in BREADTH_SYMBOLS_ENV.split(",") if s.strip()]
    uniq, seen = [], set()
    for s in SYMBOLS:
        base = s.split("#")[0].replace("-","/").upper()
        if base not in seen:
            uniq.append(base); seen.add(base)
        if len(uniq) >= 12: break
    return uniq or ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT"]

def _compute_breadth_ratio():
    refs = _breadth_refs()
    if not refs: return None
    ok = tot = 0
    for sym in refs:
        try:
            d = get_ohlcv_cached(sym, BREADTH_TF, 140)
            if not d or len(d) < 60: continue
            df = _df(d)
            if not _row_is_recent_enough(df, BREADTH_TF, bars_back=2): continue
            df["ema50"] = ema(df["close"], 50)
            row = df.iloc[-2]
            c = float(row["close"]); e = float(row["ema50"])
            if math.isfinite(c) and math.isfinite(e):
                tot += 1
                if c > e: ok += 1
        except: continue
    if tot < 5: return None
    ratio = ok / float(tot)
    return None if ratio <= 0.05 else ratio

def _get_breadth_ratio_cached():
    now_s = time.time()
    if _BREADTH_CACHE["ratio"] is not None and (now_s-_BREADTH_CACHE["t"]) <= BREADTH_TTL_SEC:
        return _BREADTH_CACHE["ratio"]
    r = _compute_breadth_ratio()
    _BREADTH_CACHE["ratio"] = r
    _BREADTH_CACHE["t"]     = now_s
    return r

def _effective_breadth_min():
    base = BREADTH_MIN_RATIO
    try:
        d = get_ohlcv_cached("BTC/USDT","4h",220)
        if not d or len(d) < 100: return base
        df = _df(d); df["ema50"] = ema(df["close"],50)
        row    = df.iloc[-2]
        above  = float(row["close"]) > float(row["ema50"])
        rsi_btc= float(rsi(df["close"],14).iloc[-2])
        if above and rsi_btc >= 55:  return max(0.40, base-0.15)
        if (not above) or rsi_btc <= 45: return min(0.75, base+0.10)
    except: pass
    return base

def _breadth_min_auto():
    try:    return max(0.38, min(0.72, _effective_breadth_min()))
    except: return BREADTH_MIN_RATIO

def breadth_status():
    try:
        r   = _get_breadth_ratio_cached()
        eff = _breadth_min_auto()
        if r is None: return {"ok":True,"ratio":None,"min":eff}
        return {"ok":(r>=eff),"ratio":r,"min":eff}
    except:
        return {"ok":True,"ratio":None,"min":BREADTH_MIN_RATIO}

# ================== Soft schedule ==================
def _parse_soft_hours(spec):
    out = []
    try:
        for block in (spec or "").split(","):
            block = block.strip()
            if not block: continue
            a, b = block.split("-")
            ah,am = [int(x) for x in a.split(":")]
            bh,bm = [int(x) for x in b.split(":")]
            out.append((ah*60+am, bh*60+bm))
    except: pass
    return out

def _soft_scale_by_time_and_market(br, eff_min):
    if not SOFT_SCHEDULE_ENABLE: return 1.0, ""
    try:
        now = now_riyadh()
        wd  = now.weekday()
        if SOFT_SCHEDULE_WEEKDAYS:
            try:
                rng = SOFT_SCHEDULE_WEEKDAYS.split("-")
                lo  = int(rng[0]); hi = int(rng[1]) if len(rng) > 1 else lo
                if not (lo <= wd <= hi): return 1.0, ""
            except: pass
        mins = now.hour*60+now.minute
        in_window = any(s <= mins <= e for s,e in _parse_soft_hours(SOFT_SCHEDULE_HRS))
        if not in_window: return 1.0, ""
        scale = SOFT_SCALE_TIME_ONLY
        if br is not None and br < eff_min:
            scale = min(scale, SOFT_SCALE_MARKET_WEAK)
        return float(max(0.4, min(1.0, scale))), "soft_window"
    except:
        return 1.0, ""

# ================== Rejection tracking ==================
_LAST_REJECT:         Dict[str, Any] = {}
_LAST_ENTRY_BAR_TS:  Dict[str, int] = {}
_SYMBOL_LAST_TRADE_AT: Dict[str, datetime] = {}
_CURRENT_SYMKEY: Optional[str] = None

def _rej(stage, **kv):
    if stage in _REJ_COUNTS: _REJ_COUNTS[stage] += 1
    try: _REJ_SUMMARY[stage] = int(_REJ_SUMMARY.get(stage,0))+1
    except: pass
    try:
        if _CURRENT_SYMKEY:
            _LAST_REJECT[_CURRENT_SYMKEY] = {
                "stage": stage,
                "details": {k:(float(v) if isinstance(v,(int,float)) else v) for k,v in kv.items()},
                "ts": now_riyadh().isoformat(timespec="seconds")}
    except: pass
    if DEBUG_LOG_SIGNALS:
        logger.info(f"[REJECT] {stage} | {' '.join(f'{k}={v}' for k,v in kv.items())}")
    return None

def _pass(stage, **kv):
    if DEBUG_LOG_SIGNALS:
        logger.info(f"[PASS]   {stage} | {' '.join(f'{k}={v}' for k,v in kv.items())}")

def _round_relax_factors():
    f_atr = f_rvol = 1.0
    notional_min = MIN_BAR_NOTIONAL_USD
    c = _REJ_COUNTS
    if c["atr_low"]      >= 10: f_atr  = 0.92
    if c["atr_low"]      >= 30: f_atr  = 0.85
    if c["rvol_low"]     >= 10: f_rvol = 0.96
    if c["rvol_low"]     >= 30: f_rvol = 0.92
    if c["notional_low"] >= 10: notional_min *= 0.85
    return f_atr, f_rvol, notional_min

# ================== _mark_signal_now ==================
def _mark_signal_now(base=None):
    try:
        ts_ms = int(time.time() * 1000)
        key   = base or _CURRENT_SYMKEY or "_last"
        _LAST_ENTRY_BAR_TS[key] = ts_ms
    except: pass

# ================== Cooldowns ==================
_COOLDOWNS: Dict[str, Dict[str, Any]] = {}

def _cooldown_minutes_for_variant(variant):
    return {"new":8,"srr":10,"srr_plus":10,"brt":10,"vbr":8,"alpha":15}.get(str(variant).lower(),8)

def _cooldown_reason(base):
    return str((_COOLDOWNS.get(base) or {}).get("reason","cooldown"))

def _cooldown_set(base, minutes, reason="cooldown"):
    _COOLDOWNS[base] = {"until": now_riyadh()+timedelta(minutes=int(minutes)), "reason":str(reason)}

def _cooldown_left_min(base):
    ent = _COOLDOWNS.get(base)
    if not ent: return 0.0
    left = (ent["until"] - now_riyadh()).total_seconds()/60.0
    if left <= 0:
        _COOLDOWNS.pop(base, None); return 0.0
    return float(max(0.0, left))


# ================== Strategy configs ==================
BASE_CFG = {
    "ENTRY_MODE":"hybrid","HYBRID_ORDER":["pullback","breakout"],
    "PULLBACK_VALUE_REF":"ema50","PULLBACK_CONFIRM":"bullish_engulf",
    "RVOL_MIN":1.2,"ATR_MIN_FOR_TREND":ATR_MIN_BASE,"USE_FIB":False,
    "SWING_LOOKBACK":60,"FIB_TOL":0.004,"BREAKOUT_BUFFER_LTF":0.0015,
    "RSI_GATE_POLICY":None,"USE_ATR_SL_TP":False,
    "STOP_LOSS_PCT":0.02,"TP1_PCT":0.03,"TP2_PCT":0.06,
    "TRAIL_AFTER_TP1":True,"TRAIL_ATR_MULT":1.0,
    "LOCK_MIN_PROFIT_PCT":0.01,"MAX_HOLD_HOURS":12,"SYMBOL_COOLDOWN_MIN":30,
}
NEW_SCALP_OVERRIDES = {
    "HYBRID_ORDER":["breakout","pullback"],"PULLBACK_VALUE_REF":"vwap",
    "PULLBACK_CONFIRM":"bos","RVOL_MIN":RVOL_MIN_NEW,"ATR_MIN_FOR_TREND":ATR_MIN_NEW,
    "USE_FIB":True,"BREAKOUT_BUFFER_LTF":0.0018,"RSI_GATE_POLICY":"lenient",
    "USE_ATR_SL_TP":True,"SL_ATR_MULT":0.9,"TP1_ATR_MULT":1.2,"TP2_ATR_MULT":2.2,
    "TRAIL_AFTER_TP1":True,"TRAIL_ATR_MULT":1.0,"LOCK_MIN_PROFIT_PCT":0.003,
    "MAX_HOLD_HOURS":6,"SYMBOL_COOLDOWN_MIN":8,
}
BRT_OVERRIDES = {
    "ENTRY_MODE":"breakout","RVOL_MIN":RVOL_MIN_BRT,"ATR_MIN_FOR_TREND":ATR_MIN_BRT,
    "RSI_GATE_POLICY":"balanced","USE_ATR_SL_TP":True,
    "SL_ATR_MULT":0.9,"TP1_ATR_MULT":1.4,"TP2_ATR_MULT":2.4,
    "TRAIL_AFTER_TP1":True,"TRAIL_ATR_MULT":1.1,
    "LOCK_MIN_PROFIT_PCT":0.004,"MAX_HOLD_HOURS":8,"SYMBOL_COOLDOWN_MIN":10,
}
SRR_OVERRIDES = {
    "ENTRY_MODE":"pullback","PULLBACK_VALUE_REF":"ema21",
    "PULLBACK_CONFIRM":"bullish_engulf","RVOL_MIN":1.20,"ATR_MIN_FOR_TREND":0.0018,
    "RSI_GATE_POLICY":"balanced","USE_ATR_SL_TP":True,
    "SL_ATR_MULT":0.9,"TP1_ATR_MULT":1.2,"TP2_ATR_MULT":2.2,
    "TRAIL_AFTER_TP1":True,"TRAIL_ATR_MULT":1.0,
    "LOCK_MIN_PROFIT_PCT":0.004,"MAX_HOLD_HOURS":8,"SYMBOL_COOLDOWN_MIN":10,
}
VBR_OVERRIDES = {
    "ENTRY_MODE":"pullback","RVOL_MIN":1.2,"ATR_MIN_FOR_TREND":0.0018,
    "RSI_GATE_POLICY":"balanced","USE_ATR_SL_TP":True,
    "SL_ATR_MULT":0.8,"TP1_ATR_MULT":1.2,"TP2_ATR_MULT":2.0,
    "TRAIL_AFTER_TP1":True,"TRAIL_ATR_MULT":1.0,
    "LOCK_MIN_PROFIT_PCT":0.003,"MAX_HOLD_HOURS":6,"SYMBOL_COOLDOWN_MIN":8,
}

PER_STRAT_MGMT = {
    "new":  {"SL":"atr","SL_MULT":0.9,"TP1":"sr_or_atr","TP1_ATR":1.2,"TP2_ATR":2.2,
             "TRAIL_AFTER_TP1":True,"TRAIL_ATR":1.0,"TIME_HRS":6},
    "old":  {"SL":"pct","SL_PCT":0.02,"TP1_PCT":0.03,"TP2_PCT":0.06,
             "TRAIL_AFTER_TP1":False,"TIME_HRS":12},
    "brt":  {"SL":"atr_below_retest","SL_MULT":1.0,"TP1":"range_or_atr",
             "TP1_ATR":1.5,"TP2_ATR":2.5,"TRAIL_AFTER_TP1":True,"TRAIL_ATR":0.9,"TIME_HRS":8},
    "srr":  {"SL":"atr","SL_MULT":0.9,"TP1":"sr_or_atr","TP1_ATR":1.2,"TP2_ATR":2.2,
             "TRAIL_AFTER_TP1":True,"TRAIL_ATR":1.0,"TIME_HRS":8},
    "vbr":  {"SL":"atr","SL_MULT":1.0,"TP1":"vwap_or_sr","TP2_ATR":1.8,
             "TRAIL_AFTER_TP1":True,"TRAIL_ATR":0.8,"TIME_HRS":3},
}

def _mgmt(variant): return PER_STRAT_MGMT.get(variant, PER_STRAT_MGMT["new"])

def get_cfg(variant):
    cfg = dict(BASE_CFG)
    v = (variant or "new").lower()
    if v == "new":    cfg.update(NEW_SCALP_OVERRIDES)
    elif v == "brt":  cfg.update(BRT_OVERRIDES)
    elif v == "srr":  cfg.update(SRR_OVERRIDES)
    elif v == "vbr":  cfg.update(VBR_OVERRIDES)
    return cfg

# ================== Entry logic ==================
def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    try:
        close_v = _finite_or(None, closed.get("close"))
        low_v   = _finite_or(None, closed.get("low"))
        if close_v is None or low_v is None: return False

        ref_cfg = (cfg.get("PULLBACK_VALUE_REF") or "").lower()
        ref_val = _finite_or(
            close_v,
            closed.get("ema50") if ref_cfg in ("ema50","") else None,
            closed.get("vwap")  if ref_cfg == "vwap"        else None,
            closed.get("ema21") if ref_cfg == "ema21"       else None,
            closed.get("ema50"), closed.get("vwap"), closed.get("ema21"),
        )
        if ref_val is None or ref_val <= 0: return False

        base_tol = float(cfg.get("PULLBACK_TOL_PCT", 0.003))
        try:    atrp = float(atr_ltf) / max(close_v, 1e-9)
        except: atrp = 0.0
        tol = base_tol + min(0.35*atrp, 0.006)

        zone_low  = ref_val*(1.0-tol)
        zone_high = ref_val*(1.0+tol)
        near_val  = (low_v <= zone_high) and (close_v >= zone_low)
        if not near_val: return False

        if not macd_rsi_gate(prev, closed, cfg.get("RSI_GATE_POLICY")): return False

        confirm = (cfg.get("PULLBACK_CONFIRM") or "").lower()
        if confirm == "bullish_engulf":
            if _bullish_engulf(prev, closed): return True
        elif confirm == "bos":
            sh, _ = _swing_points(df)
            if sh is not None and close_v > sh: return True
        elif confirm == "sweep_reclaim":
            if _sweep_then_reclaim(df, prev, closed, ref_val, lookback=20, tol=tol*1.2):
                return True
        else:
            return True

        try:
            htf_ok = float(htf_ctx.get("close",0)) > float(htf_ctx.get("ema50_now",0))
        except: htf_ok = True
        ema100_v = _finite_or(None, closed.get("ema100"))
        mid_zone = (zone_low+zone_high)*0.5
        if htf_ok and (ema100_v is None or close_v > ema100_v) and close_v >= mid_zone:
            return True
        return False
    except: return False

def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    try:
        n = int(cfg.get("SWING_LOOKBACK",60))
        if len(df) < max(40, n+3): return False
        close_v = _finite_or(None, closed.get("close"))
        open_v  = _finite_or(None, closed.get("open"))
        if close_v is None or close_v <= 0: return False

        hi_slice = df["high"].iloc[-n-2:-2]
        if hi_slice.isna().all(): return False
        hi = float(hi_slice.max())
        if not math.isfinite(hi) or hi <= 0: return False

        try:    atrp = float(atr_ltf)/max(close_v,1e-9)
        except: atrp = 0.0
        buf = float(cfg.get("BREAKOUT_BUFFER_LTF",0.0015)) + min(0.5*atrp,0.008)
        breakout_level = hi*(1.0+buf)

        rvol_v    = float(_finite_or(1.0, closed.get("rvol"), 1.0))
        need_rvol = float(cfg.get("RVOL_MIN",1.2))
        ema100_v  = _finite_or(None, closed.get("ema100"))

        try:
            htf_ok = float(htf_ctx.get("close",0)) > float(htf_ctx.get("ema50_now",0))
        except: htf_ok = True

        if (close_v > breakout_level and close_v > open_v and
                rvol_v >= max(need_rvol,1.15) and
                (ema100_v is None or close_v > ema100_v) and htf_ok):
            return True

        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        if (nr_recent and close_v > hi and rvol_v >= max(1.0,need_rvol-0.15) and
                (ema100_v is None or close_v >= ema100_v*0.995) and htf_ok):
            return True

        if (close_v > hi and close_v <= breakout_level and
                close_v > open_v and rvol_v >= max(need_rvol,1.25) and htf_ok and
                (ema100_v is None or close_v > ema100_v)):
            try:
                rng  = max(float(closed.get("high",close_v))-float(closed.get("low",close_v)),1e-9)
                body = abs(close_v-open_v)
                if body/rng >= 0.55: return True
            except: return True

        return False
    except: return False

# ================== Thresholds ==================
def regime_thresholds(breadth_ratio, atrp_now):
    try:    br = max(0.0, min(1.0, float(breadth_ratio) if breadth_ratio is not None else 0.5))
    except: br = 0.5

    if br >= 0.60:
        thr = {"ATRP_MIN_MAJ":0.0015,"ATRP_MIN_ALT":0.0018,"ATRP_MIN_MICRO":0.0022,
               "RVOL_NEED_BASE":1.10,"NOTIONAL_AVG_MIN":18000,
               "NOTIONAL_MINBAR":max(14000.0,MIN_BAR_NOTIONAL_USD*0.6),"NEUTRAL_HTF_PASS":True}
    elif br >= 0.50:
        thr = {"ATRP_MIN_MAJ":0.0018,"ATRP_MIN_ALT":0.0022,"ATRP_MIN_MICRO":0.0026,
               "RVOL_NEED_BASE":1.20,"NOTIONAL_AVG_MIN":23000,
               "NOTIONAL_MINBAR":max(19000.0,MIN_BAR_NOTIONAL_USD*0.9),"NEUTRAL_HTF_PASS":True}
    else:
        thr = {"ATRP_MIN_MAJ":0.0022,"ATRP_MIN_ALT":0.0026,"ATRP_MIN_MICRO":0.0030,
               "RVOL_NEED_BASE":1.28,"NOTIONAL_AVG_MIN":28000,
               "NOTIONAL_MINBAR":max(24000.0,MIN_BAR_NOTIONAL_USD),"NEUTRAL_HTF_PASS":False}

    try:
        if float(atrp_now) >= 0.01:
            thr["RVOL_NEED_BASE"] = max(1.05, thr["RVOL_NEED_BASE"]-0.05)
    except: pass

    f_atr, f_rvol, notional_min = _round_relax_factors()
    thr["RVOL_NEED_BASE"]  = float(thr["RVOL_NEED_BASE"]) * float(f_rvol)
    thr["ATRP_MIN_ALT"]    = float(thr["ATRP_MIN_ALT"])   * float(f_atr)
    thr["ATRP_MIN_MAJ"]    = float(thr["ATRP_MIN_MAJ"])   * float(f_atr)
    thr["ATRP_MIN_MICRO"]  = float(thr["ATRP_MIN_MICRO"]) * float(f_atr)
    thr["NOTIONAL_MINBAR"] = max(float(thr["NOTIONAL_MINBAR"])*0.95, float(notional_min)*0.95)

    try:
        eff_min = _breadth_min_auto()
        scale, _ = _soft_scale_by_time_and_market(breadth_ratio, eff_min)
        if scale < 1.0:
            ease = 1.0-(1.0-float(scale))*0.30
            thr["RVOL_NEED_BASE"] = max(1.05, float(thr["RVOL_NEED_BASE"])*ease)
            thr["ATRP_MIN_ALT"]   = float(thr["ATRP_MIN_ALT"])  *ease
            thr["ATRP_MIN_MAJ"]   = float(thr["ATRP_MIN_MAJ"])  *ease
            thr["ATRP_MIN_MICRO"] = float(thr["ATRP_MIN_MICRO"])*ease
    except: pass

    return thr

def _atrp_min_for_symbol(sym_ctx, thr):
    bucket = str(sym_ctx.get("bucket","alt")).lower()
    q35    = float(sym_ctx.get("atrp_q35_lookback",0) or 0.0)
    base   = {"maj":float(thr.get("ATRP_MIN_MAJ",0.0018)),
              "alt":float(thr.get("ATRP_MIN_ALT",0.0022)),
              "micro":float(thr.get("ATRP_MIN_MICRO",0.0026))}.get(
              bucket, float(thr.get("ATRP_MIN_ALT",0.0022)))
    if not math.isfinite(q35): q35 = 0.0
    return float(max(base, q35*0.9 if q35 > 0 else base))

def _notional_ok(sym_ctx, thr):
    avg = float(sym_ctx.get("notional_avg_30",0.0))
    mn  = float(sym_ctx.get("notional_min_30",0.0))
    return (avg >= float(thr.get("NOTIONAL_AVG_MIN",20000.0)) and
            mn  >= float(thr.get("NOTIONAL_MINBAR",15000.0))), avg, mn

def _partials_for(score, tp_count, atrp):
    try:    tp_count = max(1, min(int(tp_count), MAX_TP_COUNT))
    except: tp_count = 1
    if score >= 55 and tp_count >= 3:
        base = [0.40,0.30,0.30,0.0,0.0][:tp_count]
    elif score >= 45 and tp_count >= 3:
        base = [0.45,0.30,0.25,0.0,0.0][:tp_count]
    else:
        base = [1.0] if tp_count==1 else [0.50,0.30,0.20,0.0,0.0][:tp_count]
    try:
        if float(atrp) >= 0.008 and tp_count >= 3:
            base = [0.40,0.30,0.30,0.0,0.0][:tp_count]
    except: pass
    s = sum(base) or 1.0
    out = [round(x/s,6) for x in base]
    diff = round(1.0-sum(out),6)
    if diff and out: out[0] = round(out[0]+diff,6)
    return out


# ================== RS / Leader ==================
_RS_LAST_CHOICE = {"hours": None, "ts": 0.0}

def _rs_window_decider(br, is_breakout):
    base_hours = int(_env_float("RS_LOOKBACK_BASE_HOURS", 48))
    up_thr     = _env_float("RS_BREADTH_UP",   0.58)
    down_thr   = _env_float("RS_BREADTH_DOWN", 0.52)
    hold_min   = int(_env_float("RS_MIN_HOLD_MINUTES", 120))
    allow_72   = _env_bool("RS_ALLOW_72H", True)
    allow_24bo = _env_bool("RS_ALLOW_24H_ON_BO", True)
    now_ts     = time.time()
    last_h     = _RS_LAST_CHOICE.get("hours")
    last_ts    = float(_RS_LAST_CHOICE.get("ts") or 0.0)
    hold_ok    = (now_ts-last_ts) >= (hold_min*60)
    if br is None:
        choice = base_hours
    elif br >= up_thr:
        choice = 24 if (is_breakout and allow_24bo) else 48
    elif br <= down_thr:
        choice = 72 if allow_72 else 48
    else:
        choice = 48
    if last_h is not None and not hold_ok:
        return int(last_h)
    _RS_LAST_CHOICE["hours"] = int(choice)
    _RS_LAST_CHOICE["ts"]    = now_ts
    return int(choice)

def _is_relative_leader_vs_btc(base, lookback=None, tf="1h"):
    try:
        is_bo = False
        br    = _get_breadth_ratio_cached()
        hours = int(lookback) if lookback is not None else _rs_window_decider(br, is_bo)
        bars  = int(max(10, min(500, hours)))
        if base in ("BTC/USDT","BTC/USDC"): return True
        d_base = _df(get_ohlcv_cached(base,    tf, bars+2))
        d_btc  = _df(get_ohlcv_cached("BTC/USDT",tf, bars+2))
        if len(d_base) < bars or len(d_btc) < bars: return False
        rb = float(d_base["close"].iloc[-2]/d_base["close"].iloc[-(bars+2)]-1.0)
        rt = float(d_btc["close"].iloc[-2] /d_btc["close"].iloc[-(bars+2)] -1.0)
        return rb >= rt
    except: return False

# ================== check_signal ==================
def check_signal(symbol: str):
    global _CURRENT_SYMKEY
    base, variant = _split_symbol_variant(symbol)
    _CURRENT_SYMKEY = base

    # [NEW v4.2] تنظيف دوري
    _maybe_cleanup_dust()

    # [NEW v4.2] قراءة توجيهات العقل المدبر
    _brain = _get_brain_directives() if BRAIN_AVAILABLE else {}
    _brain_entry_ok = bool(_brain.get("entry_allowed", True))
    _brain_blocked  = list(_brain.get("blocked_patterns", []))
    _brain_score    = _brain.get("score_threshold_override")
    _effective_score_thr = int(_brain_score) if _brain_score is not None else SCORE_THRESHOLD

    # إذا العقل قال لا دخول
    if not _brain_entry_ok:
        return _rej("brain_no_entry", regime=_brain.get("regime","?"))

    left = _cooldown_left_min(base)
    if left > 0.0:
        return _rej("cooldown", left_min=round(left,1), reason=_cooldown_reason(base))

    try:
        def _lerp(x,x0,x1,y0,y1):
            if x1==x0: return (y0+y1)/2.0
            t = max(0.0,min(1.0,(float(x)-float(x0))/(float(x1)-float(x0))))
            return y0+t*(y1-y0)

        def _is_leader(base_symbol):
            coin = (base_symbol or "").split("/")[0].strip().upper()
            return coin in LEADERS_SET or bool(_is_relative_leader_vs_btc(base_symbol))

        def _atr_need_dynamic(br, eff_min):
            if br is None: return ATR_NEED_BASE
            br_weak=min(ATR_BR_WEAK, float(eff_min or ATR_BR_WEAK))
            br_base=0.50; br_strong=ATR_BR_STRONG
            if br <= br_weak:   return ATR_NEED_WEAK
            if br >= br_strong: return ATR_NEED_STRONG
            if br <= br_base:   return _lerp(br,br_weak,br_base,ATR_NEED_WEAK,ATR_NEED_BASE)
            return                     _lerp(br,br_base,br_strong,ATR_NEED_BASE,ATR_NEED_STRONG)

        def _rvol_need_dynamic(br, base_symbol):
            need = RVOL_BASE
            try:
                if br is not None and br >= RVOL_RELAX_BR_MIN and _is_leader(base_symbol):
                    need *= RVOL_RELAX_FOR_LEADERS
            except: pass
            return need

        def _safe_div(a,b,eps=1e-9):
            try:
                b=eps if(b is None or b==0 or not math.isfinite(float(b))) else float(b)
                return float(a)/float(b)
            except: return 1.0

        htf_ctx = _get_htf_context(symbol)
        if not htf_ctx: return _rej("data_unavailable")

        df = _get_ltf_df_with_fallback(symbol, STRAT_LTF_TIMEFRAME)
        if df is None or len(df) < 60:
            _cooldown_set(base, max(5,min(_cooldown_minutes_for_variant(variant),20)), reason="no_ltf")
            return _rej("no_ltf")

        if "ema100" not in df: df["ema100"] = ema(df["close"],100)
        closed = df.iloc[-2]
        prev   = df.iloc[-3]

        atr_val = _finite_or(None, _atr_from_df(df))
        price   = _finite_or(None, closed.get("close"))
        if atr_val is None or price is None or price <= 0:
            return _rej("atr_calc")
        atrp = float(atr_val)/float(price)

        # [NEW v4.2] فلتر RSI منخفض جداً: سوق في هبوط حر → لا دخول
        RSI_OVERSOLD_BLOCK = _env_float("RSI_OVERSOLD_BLOCK", 28.0)
        try:
            rsi_now = float(closed.get("rsi", 50))
            if rsi_now < RSI_OVERSOLD_BLOCK:
                return _rej("rsi_oversold_block", rsi=round(rsi_now, 1), threshold=RSI_OVERSOLD_BLOCK)
        except: pass

        _exh, _exh_reason = _is_exhausted(closed, atr_val)
        if _exh:
            return _rej(
                f"exhaustion_{_exh_reason}",
                rsi=float(closed.get("rsi", 50)),
                ema50_dist_pct=round(
                    abs(float(price) - float(closed.get("ema50", price))) / float(price), 4
                )
            )

        bucket = "maj" if base.split("/")[0] in ("BTC","ETH","BNB","SOL") else "alt"
        sym_ctx = {
            "bucket": bucket,
            "atrp_q35_lookback": float(df["close"].pct_change().rolling(35).std().iloc[-1] or 0.0),
            "price": float(price),
            "notional_avg_30": float(max(0.0, df["volume"].iloc[-30:].mean()*float(price))),
            "notional_min_30": float(max(0.0, df["volume"].iloc[-30:].min() *float(price))),
            "is_meme": False,
        }
        try:
            d1h = get_ohlcv_cached(base,"1h",80)
            if d1h and len(d1h) >= 35:
                df1h = _df(d1h); px1h = float(df1h["close"].iloc[-2])
                vol30= float(df1h["volume"].iloc[-30:].mean())
                mn30 = float(df1h["volume"].iloc[-30:].min())
                sym_ctx["notional_avg_30"] = float(min(sym_ctx["notional_avg_30"], vol30*px1h))
                sym_ctx["notional_min_30"] = float(min(sym_ctx["notional_min_30"], mn30*px1h))
        except: pass

        RVOL_BREAKOUT_BOOST = _env_float("RVOL_BREAKOUT_BOOST",0.05)
        RVOL_NR_GAIN        = _env_float("RVOL_NR_GAIN",1.03)
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_slice  = df["high"].iloc[-NR_WINDOW-2:-2]
        if len(hi_slice) < 3 or hi_slice.isna().all(): return _rej("no_ltf")
        hi_range  = float(hi_slice.max())
        if not math.isfinite(hi_range): return _rej("no_ltf")

        is_breakout = bool(
            float(closed["close"]) > hi_range and
            (nr_recent or float(closed["close"]) > _finite_or(float(closed["close"]),
             closed.get("vwap"), closed.get("ema50"))))

        try:
            vol_fast_ma = float(df["volume"].rolling(RVOL_WINDOW_FAST,
                               min_periods=max(3,RVOL_WINDOW_FAST//3)).mean().iloc[-2])
            vol_slow_ma = float(df["volume"].rolling(RVOL_WINDOW_SLOW,
                               min_periods=max(5,RVOL_WINDOW_SLOW//3)).mean().iloc[-2])
        except:
            vol_fast_ma = float(df["volume"].rolling(24,min_periods=6).mean().iloc[-2])
            vol_slow_ma = float(df["volume"].rolling(30,min_periods=8).mean().iloc[-2])

        vol_now    = float(_finite_or(df["volume"].iloc[-2], closed.get("volume"), df["volume"].iloc[-2]))
        rvol_fast  = _safe_div(vol_now, vol_fast_ma)
        rvol_slow  = _safe_div(vol_now, vol_slow_ma)
        rvol_blend = float(0.60*rvol_fast + 0.40*rvol_slow)
        rvol_eff   = (max(rvol_fast,rvol_blend)+RVOL_BREAKOUT_BOOST if is_breakout
                      else rvol_blend*(RVOL_NR_GAIN if nr_recent else 1.0))
        rvol_mode  = ("fast+boost" if is_breakout else ("blend+nr" if nr_recent else "blend"))

        try:
            ema100_val  = float(closed.get("ema100"))
            ema100_trend= "up" if float(price) > ema100_val else "down"
        except:
            ema100_trend= "flat_up"

        try:
            ema50_v = _finite_or(None, closed.get("ema50"))
            vwap_v  = _finite_or(None, closed.get("vwap"))
            close_v = float(closed["close"]); low_v = float(closed["low"])
            pb_ok = any(
                (ref is not None and close_v >= ref and low_v <= ref)
                for ref in (ema50_v, vwap_v))
        except:
            pb_ok = False

        ltf_ctx = {
            "rvol":float(rvol_eff),"rvol_fast":float(rvol_fast),
            "rvol_slow":float(rvol_slow),"rvol_mode":rvol_mode,
            "is_breakout":bool(is_breakout),"ema100_trend":ema100_trend,
            "pullback_ok":bool(pb_ok),
        }

        br      = _get_breadth_ratio_cached()
        thr     = regime_thresholds(br, atrp)
        eff_min = _breadth_min_auto()

        trend = "neutral"
        try:
            if   float(htf_ctx["close"]) > float(htf_ctx["ema50_now"]): trend = "up"
            elif float(htf_ctx["close"]) < float(htf_ctx["ema50_now"]): trend = "down"
        except: pass

        neutral_ok   = bool(thr.get("NEUTRAL_HTF_PASS",True))
        weak_market  = (br is not None) and (br < eff_min)
        need_rvol_base = float(thr.get("RVOL_NEED_BASE",1.15))
        strong_breakout= bool(
            is_breakout and ema100_trend=="up" and float(rvol_eff)>=need_rvol_base*1.10)

        if trend == "down":
            if not ((br is not None and br >= max(0.58,eff_min+0.04)) or strong_breakout):
                return _rej("htf_trend", trend=trend)
        elif trend == "neutral" and not neutral_ok:
            if not (weak_market or strong_breakout):
                return _rej("htf_trend", trend=trend)

        if ema100_trend == "down" and not strong_breakout:
            return _rej("ema100_trend",
                        price=float(price),
                        ema100=float(closed.get("ema100") or 0.0))

        need_atrp = max(_atr_need_dynamic(br, eff_min), _atrp_min_for_symbol(sym_ctx, thr))
        if float(atrp) < float(need_atrp):
            return _rej("atr_low", atrp=atrp, need=need_atrp)

        rvol_need = _rvol_need_dynamic(br, base)
        if float(sym_ctx.get("price",1.0)) < 0.1 or bool(sym_ctx.get("is_meme")):
            rvol_need -= 0.08
        if is_breakout:
            rvol_need -= 0.05
        rvol_val = float(ltf_ctx["rvol"])
        if rvol_val < rvol_need:
            return _rej("rvol_low", rvol=rvol_val, need=rvol_need)

        n_ok, avg_not, minbar = _notional_ok(sym_ctx, thr)
        if not n_ok:
            return _rej("notional_low", avg=avg_not, minbar=minbar)

        score, why_str, pattern = _opportunity_score(df, prev, closed)

        cfg = get_cfg(variant)
        chosen_mode = None
        order_pref  = cfg.get("HYBRID_ORDER",["pullback","breakout"]) if cfg.get("ENTRY_MODE")=="hybrid" else [cfg.get("ENTRY_MODE","pullback")]
        for m in (order_pref + [x for x in ["pullback","breakout"] if x not in order_pref]):
            if m == "pullback" and _entry_pullback_logic(df,closed,prev,atr_val,htf_ctx,cfg):
                chosen_mode = "pullback"; break
            if m == "breakout" and _entry_breakout_logic(df,closed,prev,atr_val,htf_ctx,cfg):
                chosen_mode = "breakout"; break

        if chosen_mode:
            # [NEW v4.2] استخدام عتبة السكور من العقل المدبر
            if int(score) < _effective_score_thr:
                return _rej("score_low", score=score, need=_effective_score_thr)
            # [NEW v4.2] فلتر الأنماط المحظورة من العقل
            if pattern in _brain_blocked:
                return _rej("brain_blocked_pattern", pattern=pattern)
            _pass("buy", mode=chosen_mode, score=int(score))
            _mark_signal_now(base)
            return {
                "decision":"buy","mode":chosen_mode,"score":int(score),
                "reasons":why_str,"pattern":pattern,"is_early_scout":False,
                "features":{
                    "atrp":float(atrp),"rvol":float(rvol_val),
                    "rvol_fast":float(ltf_ctx["rvol_fast"]),
                    "rvol_slow":float(ltf_ctx["rvol_slow"]),
                    "rvol_mode":str(ltf_ctx["rvol_mode"]),
                    "breadth_ratio":(None if br is None else float(br)),
                    "htf_ok":bool(trend in ("up","neutral")),
                    "ema100_trend":ema100_trend,
                    "notional_avg_30":float(avg_not),
                    "notional_min_30":float(minbar),
                }
            }

        if EARLY_SCOUT_ENABLE:
            try:
                ema50_val = _finite_or(None, closed.get("ema50"))
                if ema50_val and atr_val and atr_val > 0:
                    dist_atr = abs(float(price)-float(ema50_val))/float(atr_val)
                    EARLY_RVOL_MULT = _env_float("EARLY_SCOUT_RVOL_MULT", 1.02)
                    EARLY_RVOL_MIN  = _env_float("EARLY_SCOUT_RVOL_MIN",  0.95)
                    early_rvol_ok   = (float(rvol_val) >= max(EARLY_RVOL_MULT*need_rvol_base, EARLY_RVOL_MIN))

                    early_ok = (
                        ema100_trend == "up" and
                        trend == "up" and
                        br is not None and br >= EARLY_SCOUT_BR_MIN and
                        bool(ltf_ctx["pullback_ok"]) and
                        dist_atr <= float(EARLY_SCOUT_MAX_ATR_DIST) and
                        int(score) >= int(EARLY_SCOUT_SCORE_MIN) and
                        early_rvol_ok and n_ok
                    )

                    if early_ok:
                        _pass("buy", mode="early_scout", score=int(score))
                        _mark_signal_now(base)
                        return {
                            "decision":"buy","mode":"early_scout","score":int(score),
                            "reasons":(why_str+" | early_scout"),
                            "pattern":pattern,"is_early_scout":True,
                            "features":{
                                "atrp":float(atrp),"rvol":float(rvol_val),
                                "rvol_fast":float(ltf_ctx["rvol_fast"]),
                                "rvol_slow":float(ltf_ctx["rvol_slow"]),
                                "rvol_mode":str(ltf_ctx["rvol_mode"]),
                                "breadth_ratio":(None if br is None else float(br)),
                                "htf_ok":bool(trend=="up"),
                                "ema100_trend":ema100_trend,
                                "ema50":float(ema50_val),
                                "dist_atr_ema50":float(dist_atr),
                                "notional_avg_30":float(avg_not),
                                "notional_min_30":float(minbar),
                            }
                        }
            except: pass

        return _rej("entry_mode", mode=get_cfg(variant).get("ENTRY_MODE","hybrid"))

    except Exception as e:
        return _rej("error", err=str(e))
    finally:
        _CURRENT_SYMKEY = None


# ================== Entry plan builder ==================
def _atr_latest(symbol_base, tf, bars=180):
    data = get_ohlcv_cached(symbol_base, tf, bars)
    if not data: raise RuntimeError("no LTF data")
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 50: raise RuntimeError("ltf too short")
    closed = df.iloc[-2]
    px     = float(closed["close"])
    atr_abs= _atr_from_df(df)
    if atr_abs is None or not math.isfinite(float(atr_abs)) or float(atr_abs)<=0:
        raise RuntimeError("atr invalid")
    return float(px), float(atr_abs), float(atr_abs)/max(px,1e-9)

def _build_entry_plan(symbol, sig=None):
    base, variant = _split_symbol_variant(symbol)
    if sig is None:
        r = check_signal(symbol)
        if not (isinstance(r,dict) and r.get("decision")=="buy"):
            raise RuntimeError("no buy signal")
        sig = r

    price, atr_abs, atrp = _atr_latest(base, LTF_TIMEFRAME)
    mgmt = _mgmt(variant)

    if mgmt.get("SL") in ("atr","atr_below_sweep","atr_below_retest"):
        sl_mult = float(mgmt.get("SL_MULT",1.0))
        majors  = {s.strip().upper() for s in os.getenv("MAJORS","BTC,ETH,BNB,SOL").split(",") if s.strip()}
        if base.split("/")[0].upper() in majors:
            sl_mult *= float(os.getenv("SL_MULT_MAJORS","1.25"))
        sl = float(price - sl_mult*atr_abs)
    elif mgmt.get("SL") == "pct":
        sl = float(price*(1.0-float(mgmt.get("SL_PCT",0.02))))
    else:
        sl = float(price - 1.0*atr_abs)

    tps = []
    if ENABLE_MULTI_TARGETS:
        mults = [float(mgmt[k]) for k in ("TP1_ATR","TP2_ATR","TP3_ATR") if k in mgmt]
        if not mults: mults = list(TP_ATR_MULTS_TREND)[:3]
        for m in mults[:MAX_TP_COUNT]:
            tps.append(float(price+float(m)*atr_abs))
    else:
        tps.append(float(price+float(mgmt.get("TP1_ATR",1.2))*atr_abs))
        tps.append(float(price+float(mgmt.get("TP2_ATR",2.2))*atr_abs))

    tps = sorted(tps)
    score_for_partials = int(sig.get("score",SCORE_THRESHOLD))
    partials = _partials_for(score_for_partials, len(tps), atrp)
    if len(partials) != len(tps):
        partials = (partials[:len(tps)] if len(partials)>len(tps)
                    else partials+[round(max(0.0,1.0-sum(partials)),6)])

    if USE_DYNAMIC_MAX_BARS:
        max_bars = (MAX_BARS_BASE+6 if atrp>=0.01 else MAX_BARS_BASE+3 if atrp>=0.006 else MAX_BARS_BASE)
    else:
        max_bars = None

    sig = dict(sig)
    sig["sl"]             = float(sl)
    sig["targets"]        = [float(x) for x in tps]
    sig["partials"]       = partials
    sig["atrp"]           = float(atrp)
    sig["max_bars_to_tp1"]= max_bars
    sig.setdefault("messages",{})
    return sig


# ================== [NEW v4.2] تنظيف الغبار التلقائي ==================
def _is_dust_position(symbol: str, pos: dict):
    """يتحقق إذا كان المركز غباراً لا يمكن بيعه"""
    try:
        base = symbol.split("#")[0]
        amount = float(pos.get("amount", pos.get("qty", 0)) or 0)
        if amount <= 0:
            return True, "amount=0"
        try:
            f = fetch_symbol_filters(base) or {}
            min_qty = float(f.get("minQty", 0) or 0)
            min_notional = float(f.get("minNotional", MIN_NOTIONAL_USDT) or MIN_NOTIONAL_USDT)
        except:
            return False, ""
        if min_qty > 0 and amount < min_qty * 0.5:
            return True, f"amount={amount:.8f} < min_qty/2={min_qty/2:.8f}"
        try:
            price = float(fetch_price(base) or 0)
            if price > 0:
                value = amount * price
                if value < float(os.getenv("SELL_DUST_MIN_USDT", "0.5")):
                    return True, f"value={value:.4f}$ < dust_min"
                if value < min_notional * 0.5:
                    return True, f"value={value:.4f}$ < min_notional/2"
        except:
            pass
    except Exception as e:
        logger.warning(f"[dust_check] {symbol}: error={e}")
    return False, ""


def cleanup_dust_positions(force: bool = False, notify: bool = True) -> int:
    """[NEW v4.2] تنظيف تلقائي لمراكز الغبار"""
    cleaned = 0
    dust_log = _read_json(DUST_LOG_FILE, [])
    try:
        files = [f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")]
    except:
        return 0
    for fname in files:
        try:
            pos = _read_json(f"{POSITIONS_DIR}/{fname}", None)
            if not pos:
                try: os.remove(f"{POSITIONS_DIR}/{fname}")
                except: pass
                cleaned += 1
                continue
            symbol = pos.get("symbol", fname.replace(".json","").replace("_","/",1))
            is_dust, reason = _is_dust_position(symbol, pos)
            if is_dust:
                logger.info(f"[cleanup_dust] 🧹 حذف غبار: {symbol} | {reason}")
                dust_log.append({"symbol": symbol, "reason": reason,
                                 "amount": pos.get("amount", 0),
                                 "cleaned_at": now_riyadh().isoformat(timespec="seconds")})
                clear_position(symbol)
                cleaned += 1
                if notify and STRAT_TG_SEND:
                    _tg_once(f"dust_cleaned_{symbol}",
                             f"🧹 تنظيف غبار: {symbol}\nالسبب: {reason}",
                             ttl_sec=3600)
        except Exception as e:
            logger.warning(f"[cleanup_dust] {fname}: error={e}")
    if dust_log:
        try: _atomic_write(DUST_LOG_FILE, dust_log[-200:])
        except: pass
    if cleaned > 0:
        logger.info(f"[cleanup_dust] ✅ تم تنظيف {cleaned} مركز/مراكز")
    return cleaned


_LAST_DUST_CLEANUP = 0.0
def _maybe_cleanup_dust():
    global _LAST_DUST_CLEANUP
    now_s = time.time()
    interval = float(os.getenv("DUST_CLEANUP_INTERVAL_SEC", "3600"))
    if (now_s - _LAST_DUST_CLEANUP) >= interval:
        cleanup_dust_positions(notify=True)
        _LAST_DUST_CLEANUP = now_s


def on_startup():
    """يُستدعى عند بدء التشغيل — ينظف الغبار تلقائياً"""
    global _LAST_DUST_CLEANUP
    logger.info("[startup] 🚀 strategy v4.2")
    cleaned = cleanup_dust_positions(force=True, notify=True)
    _LAST_DUST_CLEANUP = time.time()
    if cleaned > 0:
        logger.info(f"[startup] ✅ تم تنظيف {cleaned} مركز غبار")
    return cleaned


# ================== [FIX] fetch_balance مع retry ==================
def _fetch_balance_safe(coin: str, retries: int = 3, delay: float = 1.0) -> float:
    """
    جلب الرصيد المتاح بشكل آمن مع إعادة المحاولة.
    يرجع 0.0 في حالة الفشل مع تسجيل السبب.
    """
    last_err = None
    for attempt in range(retries):
        try:
            val = fetch_balance(coin)
            if val is None:
                logger.warning(f"[balance] {coin}: fetch_balance returned None (attempt {attempt+1})")
                time.sleep(delay)
                continue
            result = float(val)
            if result < 0:
                logger.warning(f"[balance] {coin}: negative balance={result}, treating as 0")
                return 0.0
            return result
        except Exception as e:
            last_err = e
            logger.warning(f"[balance] {coin}: attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    logger.error(f"[balance] {coin}: all {retries} attempts failed. last_err={last_err}")
    return 0.0

# ================== [FIX] _safe_sell المُصلَحة ==================
def _safe_sell(base_symbol: str, want_qty: float):
    """
    بيع آمن مع:
    - لوج تشخيصي كامل
    - تحقق دقيق من الرصيد المتاح
    - معالجة stepSize بشكل صحيح (منع التقريب لصفر)
    - retry عند خطأ رصيد غير كافٍ
    - fallback للسعر من OHLCV إذا فشل fetch_price
    """
    SELL_DUST_MIN  = float(os.getenv("SELL_DUST_MIN_USDT", "5.0"))
    SELL_WARN_TTL  = int(os.getenv("SELL_WARN_TTL", "1800"))
    SELL_RETRIES   = int(os.getenv("SELL_RETRIES", "2"))

    coin = base_symbol.split("/")[0]

    # ── [FIX] جلب الرصيد مع retry ──
    avail = _fetch_balance_safe(coin, retries=3, delay=0.8)

    logger.info(
        f"[_safe_sell] {base_symbol} | want={float(want_qty or 0):.8f}"
        f" | avail={avail:.8f} | dry={DRY_RUN}"
    )

    if avail <= 0.0:
        msg = f"⚠️ لا كمية متاحة لـ {base_symbol}. (avail={avail:.8f})"
        logger.warning(f"[_safe_sell] {msg}")
        _tg_once(f"warn_insuff_{base_symbol}", msg, ttl_sec=SELL_WARN_TTL)
        return None, None, 0.0

    # ── جلب فلاتر الرمز ──
    try:
        f = fetch_symbol_filters(base_symbol) or {}
    except Exception as e:
        logger.warning(f"[_safe_sell] fetch_symbol_filters failed for {base_symbol}: {e}")
        f = {}

    step    = float(f.get("stepSize", 0.000001) or 0.000001)
    min_qty = float(f.get("minQty", 0.0) or 0.0)

    # ── [FIX] جلب السعر مع fallback ──
    price_now = _price_now_safe(base_symbol)
    if price_now <= 0.0:
        logger.warning(f"[_safe_sell] {base_symbol}: price unavailable, skipping sell")
        _tg_once(
            f"sell_skip_noprice_{base_symbol}",
            f"⚠️ تخطّي البيع {base_symbol}: السعر غير متاح.",
            ttl_sec=SELL_WARN_TTL
        )
        return None, None, 0.0

    min_notional = _min_notional_for(base_symbol, price_now)

    # ── [FIX] حساب الكمية بشكل صحيح ──
    raw = max(0.0, min(float(want_qty or 0.0), avail))

    # تقريب للأسفل حسب stepSize
    if step > 0:
        qty = math.floor(raw / step) * step
        # [FIX] إذا التقريب أعطى صفر لكن raw > 0، استخدم step واحد كحد أدنى
        if qty <= 0.0 and raw > 0.0:
            qty = step
            logger.info(f"[_safe_sell] {base_symbol}: raw={raw:.8f} < step={step:.8f}, using step as min qty")
    else:
        qty = raw

    # تطبيق minQty
    # [FIX-KEY v4.2] إذا avail أقل بكثير من min_qty → غبار، لا تُعيد المحاولة
    if min_qty > 0 and avail < min_qty * 0.9:
        logger.warning(
            f"[_safe_sell] {base_symbol}: DUST — avail={avail:.8f} << min_qty={min_qty:.8f} → حذف المركز"
        )
        _tg_once(
            f"dust_auto_{base_symbol}",
            f"🧹 غبار محذوف: {base_symbol} | avail={avail:.8f} < min_qty={min_qty:.8f}",
            ttl_sec=SELL_WARN_TTL
        )
        return None, None, 0.0

    if min_qty > 0 and qty < min_qty:
        adjusted = math.floor(min(avail, min_qty * 1.001) / step) * step if step > 0 else min_qty
        if adjusted <= avail:
            logger.info(
                f"[_safe_sell] {base_symbol}: qty={qty:.8f} < min_qty={min_qty:.8f}"
                f", adjusting to {adjusted:.8f}"
            )
            qty = adjusted
        else:
            logger.warning(
                f"[_safe_sell] {base_symbol}: can't meet min_qty={min_qty:.8f}"
                f" with avail={avail:.8f}"
            )
            _tg_once(
                f"sell_min_qty_{base_symbol}",
                f"⚠️ {base_symbol}: الكمية المتاحة ({avail:.8f}) أقل من الحد الأدنى ({min_qty:.8f}).",
                ttl_sec=SELL_WARN_TTL
            )
            return None, None, 0.0

    # ── تحقق من الغبار ──
    if qty <= 0.0 or (qty * price_now) < SELL_DUST_MIN:
        logger.info(
            f"[_safe_sell] {base_symbol}: dust skip | qty={qty:.8f}"
            f" | value={qty * price_now:.4f}$ < {SELL_DUST_MIN}$"
        )
        _tg_once(
            f"sell_dust_{base_symbol}",
            f"⚠️ غبار {base_symbol}: {qty * price_now:.2f}$",
            ttl_sec=SELL_WARN_TTL
        )
        return None, None, 0.0

    # ── [FIX] تحقق من min_notional مع محاولة رفع الكمية ──
    notional_value = qty * price_now
    if notional_value < min_notional:
        need_amt = math.ceil((min_notional / price_now) / step) * step if step > 0 else (min_notional / price_now)
        logger.info(
            f"[_safe_sell] {base_symbol}: notional={notional_value:.4f}$"
            f" < min_notional={min_notional:.4f}$, need_amt={need_amt:.8f}, avail={avail:.8f}"
        )
        if need_amt <= avail:
            qty = need_amt
            logger.info(f"[_safe_sell] {base_symbol}: qty adjusted to {qty:.8f} to meet min_notional")
        else:
            # بيع كل ما هو متاح كآخر محاولة
            qty_all = math.floor(avail / step) * step if step > 0 else avail
            if qty_all > 0 and (qty_all * price_now) >= SELL_DUST_MIN:
                logger.info(
                    f"[_safe_sell] {base_symbol}: selling all available qty={qty_all:.8f}"
                    f" (notional={qty_all * price_now:.4f}$)"
                )
                qty = qty_all
            else:
                logger.warning(f"[_safe_sell] {base_symbol}: can't meet min_notional, skipping")
                _tg_once(
                    f"sell_skip_small_{base_symbol}",
                    f"⚠️ قيمة صغيرة {base_symbol}: {notional_value:.2f}$ < {min_notional:.2f}$.",
                    ttl_sec=SELL_WARN_TTL
                )
                return None, None, 0.0

    # ── DRY RUN ──
    if DRY_RUN:
        logger.info(f"[_safe_sell] DRY_RUN: {base_symbol} sell {qty:.8f} @ {price_now:.6f}")
        return {"id": f"dry_sell_{int(time.time())}", "average": price_now}, price_now, qty

    # ── تنفيذ البيع الفعلي مع retry ──
    for sell_attempt in range(SELL_RETRIES + 1):
        try:
            order = place_market_order(base_symbol, "sell", qty)
            if not order:
                logger.warning(f"[_safe_sell] {base_symbol}: place_market_order returned None (attempt {sell_attempt+1})")
                time.sleep(1.0)
                continue

            exit_px = float(order.get("average") or order.get("price") or price_now)
            logger.info(
                f"[_safe_sell] ✅ {base_symbol}: sold qty={qty:.8f}"
                f" @ {exit_px:.6f} | total={qty * exit_px:.4f}$"
            )
            return order, exit_px, qty

        except Exception as e:
            msg = str(e)
            logger.warning(f"[_safe_sell] {base_symbol}: sell attempt {sell_attempt+1} error: {msg}")

            # [FIX] خطأ رصيد غير كافٍ — تحديث الرصيد وإعادة الحساب
            if "51008" in msg or "insufficient" in msg.lower() or "balance" in msg.lower():
                logger.info(f"[_safe_sell] {base_symbol}: insufficient balance error, refreshing balance...")
                fresh_avail = _fetch_balance_safe(coin, retries=2, delay=0.5)
                logger.info(f"[_safe_sell] {base_symbol}: fresh_avail={fresh_avail:.8f}")

                if fresh_avail <= 0.0:
                    logger.warning(f"[_safe_sell] {base_symbol}: no balance after refresh, aborting")
                    _tg_once(
                        f"sell_fail_{base_symbol}",
                        f"❌ فشل بيع {base_symbol}: رصيد صفر بعد التحديث.",
                        ttl_sec=SELL_WARN_TTL
                    )
                    return None, None, 0.0

                # إعادة حساب الكمية بناءً على الرصيد الجديد
                qty2 = math.floor(max(0.0, min(qty * 0.99, fresh_avail)) / step) * step if step > 0 else min(qty * 0.99, fresh_avail)
                if qty2 > 0 and (qty2 * price_now) >= max(SELL_DUST_MIN, min_notional):
                    qty = qty2
                    logger.info(f"[_safe_sell] {base_symbol}: retrying with adjusted qty={qty:.8f}")
                    continue
                else:
                    logger.warning(f"[_safe_sell] {base_symbol}: adjusted qty={qty2:.8f} too small, aborting")
                    _tg_once(
                        f"sell_fail_{base_symbol}",
                        f"❌ فشل بيع {base_symbol} (51008): الكمية المتاحة لا تكفي.",
                        ttl_sec=SELL_WARN_TTL
                    )
                    return None, None, 0.0
            else:
                # خطأ آخر — انتظر ثم أعد المحاولة
                if sell_attempt < SELL_RETRIES:
                    time.sleep(1.5 * (sell_attempt + 1))
                    continue
                else:
                    _tg_once(
                        f"sell_fail_{base_symbol}",
                        f"❌ فشل بيع {base_symbol}: {msg}",
                        ttl_sec=SELL_WARN_TTL
                    )
                    return None, None, 0.0

    # وصلنا هنا = فشلت كل المحاولات
    logger.error(f"[_safe_sell] {base_symbol}: all sell attempts failed")
    _tg_once(
        f"sell_fail_{base_symbol}",
        f"❌ فشل بيع {base_symbol} بعد {SELL_RETRIES+1} محاولات.",
        ttl_sec=SELL_WARN_TTL
    )
    return None, None, 0.0


# ================== execute_buy ==================
def execute_buy(symbol, sig=None):
    base, variant = _split_symbol_variant(symbol)
    sig = _build_entry_plan(symbol, sig)

    try:
        s = load_risk_state()
        if int(s.get("trades_today",0)) >= int(MAX_TRADES_PER_DAY):
            return None, "🚫 تم بلوغ حد الصفقات اليومية."
    except: pass

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 تم بلوغ الحد الأقصى للمراكز المفتوحة."
    if _is_blocked():
        return None, "⏸️ النظام في حالة حظر مؤقت."

    EXEC_USDT_RESERVE  = _env_float("EXEC_USDT_RESERVE",  10.0)
    EXEC_MIN_FREE_USDT = _env_float("EXEC_MIN_FREE_USDT", 15.0)
    SLIPPAGE_MAX_PCT   = _env_float("SLIPPAGE_MAX_PCT",   0.012)
    TRADE_USDT_MIN     = _env_float("TRADE_USDT_MIN",     MIN_TRADE_USDT)
    TRADE_USDT_MAX     = _env_float("MAX_TRADE_USDT",     0.0)
    LEADER_SIZE_MULT_ENV      = _env_float("LEADER_SIZE_MULT",      LEADER_SIZE_MULT)
    LEADER_DONT_DOWNSCALE_ENV = _env_bool("LEADER_DONT_DOWNSCALE", LEADER_DONT_DOWNSCALE)

    trade_usdt = float(TRADE_BASE_USDT)
    br         = _get_breadth_ratio_cached()
    eff_min    = _breadth_min_auto()
    is_leader  = _is_relative_leader_vs_btc(base)

    if br is not None:
        if br < 0.45:   trade_usdt *= 0.72
        elif br < 0.55: trade_usdt *= 0.88

    if SOFT_BREADTH_ENABLE and br is not None and br < eff_min and not is_leader:
        scale, note = _soft_scale_by_time_and_market(br, eff_min)
        trade_usdt *= scale
        if SOFT_MSG_ENABLE:
            sig["messages"]["breadth_soft"] = f"⚠️ Soft breadth: ratio={br:.2f} → size×{scale:.2f}"

    try:
        sc   = int(sig.get("score",SCORE_THRESHOLD))
        atrp_sig = float(sig.get("atrp",0.0))
        if sc >= 70:    trade_usdt *= 1.30
        elif sc >= 60:  trade_usdt *= 1.15
        elif sc >= 50:  trade_usdt *= 1.00
        if atrp_sig >= 0.008: trade_usdt *= 1.10
        elif atrp_sig <= 0.0035: trade_usdt *= 0.90
    except: pass

    if is_leader and not LEADER_DONT_DOWNSCALE_ENV:
        trade_usdt *= LEADER_SIZE_MULT_ENV

    is_early_scout = bool(sig.get("is_early_scout",False))
    if EARLY_SCOUT_ENABLE and is_early_scout:
        trade_usdt *= float(EARLY_SCOUT_SIZE_MULT)
        sig["messages"]["early_scout"] = f"🟢 Early Scout: size×{EARLY_SCOUT_SIZE_MULT:.2f}"

    if AGGR_MODE_ENABLE and not is_early_scout:
        try:
            sc_aggr = int(sig.get("score",0))
            mode    = str(sig.get("mode","")).lower()
            if sc_aggr >= AGGR_SCORE_MIN and ((not AGGR_BREAKOUT_ONLY) or mode=="breakout"):
                aggr_mult  = 1.60 if sc_aggr >= AGGR_SCORE_STRONG else 1.35
                max_allowed= TRADE_BASE_USDT*float(AGGR_MAX_RISK_MULT)
                trade_usdt = min(trade_usdt*aggr_mult, max_allowed)
                sig["messages"]["aggressive_clean"] = f"🔥 Aggressive: score={sc_aggr} → ≤x{AGGR_MAX_RISK_MULT:.2f}"
        except: pass

    # [NEW v4.2] تطبيق size_multiplier من العقل المدبر
    if BRAIN_AVAILABLE:
        _b = _get_brain_directives()
        _brain_mult = float(_b.get("size_multiplier", 1.0))
        if _brain_mult != 1.0:
            trade_usdt *= _brain_mult
            sig["messages"]["brain_size"] = f"🧠 Brain: size×{_brain_mult:.2f} ({_b.get('regime','?')})"
            logger.info(f"[execute_buy] 🧠 Brain size×{_brain_mult:.2f} → {trade_usdt:.2f}$")
        _brain_max_pos = _b.get("max_open_positions_override")
        if _brain_max_pos is not None and count_open_positions() >= int(_brain_max_pos):
            return None, f"🧠 Brain: حد المراكز ({int(_brain_max_pos)}) — {_b.get('regime','?')}"

    if TRADE_USDT_MAX > 0: trade_usdt = min(trade_usdt, TRADE_USDT_MAX)
    trade_usdt = max(trade_usdt, TRADE_USDT_MIN)

    # [FIX] استخدام _fetch_balance_safe بدل get_usdt_free المباشرة
    usdt_free = _fetch_balance_safe("USDT", retries=3, delay=0.8)
    logger.info(f"[execute_buy] {base}: usdt_free={usdt_free:.4f}$, trade_usdt={trade_usdt:.4f}$")

    if usdt_free < EXEC_MIN_FREE_USDT:
        return None, f"🚫 رصيد غير كافٍ ({usdt_free:.2f}$ < {EXEC_MIN_FREE_USDT:.2f}$)."
    max_affordable = max(0.0, usdt_free - EXEC_USDT_RESERVE)
    if max_affordable <= 0:
        return None, f"🚫 الاحتياطي محجوز ({EXEC_USDT_RESERVE:.2f}$)."
    trade_usdt = min(trade_usdt, max_affordable)

    # [FIX] جلب فلاتر الرمز مع معالجة الأخطاء
    try:
        f = fetch_symbol_filters(base) or {}
    except Exception as e:
        logger.warning(f"[execute_buy] fetch_symbol_filters failed for {base}: {e}")
        f = {}

    step    = float(f.get("stepSize", 0.000001) or 0.000001)
    min_qty = float(f.get("minQty", 0.0) or 0.0)
    min_notional = float(f.get("minNotional", MIN_NOTIONAL_USDT) or MIN_NOTIONAL_USDT)
    tick    = float(f.get("tickSize", 0.00000001) or 0.00000001)

    price = float(fetch_price(base) or 0.0)
    if not (price > 0): return None, "⚠️ لا يمكن جلب سعر صالح."

    raw_amount = trade_usdt / price
    amount     = math.floor(raw_amount / step) * step

    # [FIX] إذا التقريب أعطى صفر، استخدم step واحد
    if amount <= 0.0 and raw_amount > 0.0:
        amount = step
        trade_usdt = amount * price
        logger.info(f"[execute_buy] {base}: amount rounded to 0, using step={step:.8f} → trade_usdt={trade_usdt:.4f}$")

    if min_qty and amount < min_qty:
        amount = min_qty
        trade_usdt = amount * price
        if trade_usdt > max_affordable:
            return None, "🚫 لا يمكن بلوغ الحد الأدنى للكمية ضمن الرصيد."

    if amount * price < min_notional:
        need_amt  = math.ceil((min_notional / price) / step) * step
        need_usdt = need_amt * price
        cap       = TRADE_USDT_MAX if TRADE_USDT_MAX > 0 else float("inf")
        if need_usdt <= min(max_affordable, cap):
            amount = need_amt; trade_usdt = need_usdt
        else:
            return None, f"🚫 قيمة الصفقة ({amount * price:.2f}$) أقل من الحد الأدنى ({min_notional:.2f}$) ولا يمكن رفعها ضمن الرصيد المتاح."

    if trade_usdt < TRADE_USDT_MIN:
        return None, f"🚫 حجم الصفقة أقل من الحد الأدنى ({TRADE_USDT_MIN:.2f}$)."

    logger.info(
        f"[execute_buy] {base}: amount={amount:.8f}, trade_usdt={trade_usdt:.4f}$"
        f", price={price:.6f}, min_notional={min_notional:.4f}$"
    )

    if DRY_RUN:
        order = {"id":f"dry_{int(time.time())}", "average":price, "filled":float(amount)}
    else:
        try:
            order = place_market_order(base, "buy", amount)
        except Exception as e:
            _tg_once(f"buy_fail_{base}", f"❌ فشل شراء {base}: {e}", ttl_sec=600)
            return None, "⚠️ فشل تنفيذ الصفقة."
        if not order: return None, "⚠️ فشل تنفيذ الصفقة."

    fill_px    = float(order.get("average") or order.get("price") or price)
    filled_amt = float(order.get("filled") or amount)
    if filled_amt <= 0: return None, "⚠️ لم يتم تنفيذ أي كمية."
    amount = filled_amt
    trade_usdt_final = amount * fill_px

    slippage = abs(fill_px - price) / price
    if slippage > SLIPPAGE_MAX_PCT:
        try:
            if not DRY_RUN: place_market_order(base, "sell", amount)
        except: pass
        return None, f"🚫 انزلاق مرتفع وتم التراجع ({slippage:.2%})."

    sl_raw   = float(sig["sl"])
    if sl_raw >= fill_px: sl_raw = fill_px * 0.985
    sl_final = _round_to_tick(sl_raw, tick)

    pos = {
        "symbol":symbol,"amount":float(amount),
        "entry_price":_round_to_tick(fill_px,tick),
        "stop_loss":float(sl_final),
        "targets":[float(x) for x in sig["targets"]],
        "partials":list(sig.get("partials") or []),
        "opened_at":now_riyadh().isoformat(timespec="seconds"),
        "variant":variant,"htf_stop":sig.get("stop_rule"),
        "max_bars_to_tp1":sig.get("max_bars_to_tp1"),
        "messages":sig.get("messages",{}),
        "tp_hits":[False]*len(sig["targets"]),
        "score":sig.get("score"),"pattern":sig.get("pattern"),
        "reason":sig.get("reasons"),
        "max_hold_hours":_mgmt(variant).get("TIME_HRS"),
        "is_early_scout":bool(is_early_scout),
        "breakeven_done":False,
    }

    try:
        df_ltf = _df(get_ohlcv_cached(base, LTF_TIMEFRAME, 120))
        if len(df_ltf) >= 40:
            pos["atr_entry"] = float(_atr_from_df(df_ltf))
    except: pass

    save_position(symbol, pos)
    register_trade_opened()

    try:
        if STRAT_TG_SEND:
            _tg(
                f"✅ دخول {symbol}\n"
                f"🎯 Mode: {sig.get('mode','-')} • Score: {sig.get('score','-')} • Pattern: {sig.get('pattern','-')}\n"
                f"🟢 Entry: <code>{pos['entry_price']:.6f}</code>\n"
                f"🛡️ SL: <code>{pos['stop_loss']:.6f}</code>\n"
                f"🎯 TPs: {', '.join(str(round(t,6)) for t in pos['targets'])}\n"
                f"💰 الحجم: {trade_usdt_final:.2f}$"
                + ("\n🟢 Early Scout" if is_early_scout else "")
            )
    except: pass

    return order, (
        f"✅ شراء {symbol} | SL: {pos['stop_loss']:.6f}"
        f" | 💰 {trade_usdt_final:.2f}$"
        + (" | Early Scout" if is_early_scout else ""))


# ================== Sell helpers ==================
def _price_now_safe(sym):
    """جلب السعر الحالي مع fallback للـ OHLCV"""
    try:
        px = float(fetch_price(sym) or 0.0)
        if px > 0: return px
    except: pass
    # [FIX] fallback: جلب السعر من آخر شمعة مغلقة
    try:
        d = _df(get_ohlcv_cached(sym, LTF_TIMEFRAME, 5))
        if d is not None and len(d) >= 3:
            px = float(d["close"].iloc[-2])
            if px > 0:
                logger.info(f"[_price_now_safe] {sym}: using OHLCV fallback price={px:.6f}")
                return px
    except: pass
    logger.warning(f"[_price_now_safe] {sym}: price unavailable")
    return 0.0

def _min_notional_for(sym, price_now):
    base_default = float(MIN_NOTIONAL_USDT)
    try:
        f = fetch_symbol_filters(sym) or {}
        mn = f.get("minNotional")
        if mn is not None and float(mn) > 0: return float(mn)
        min_qty = float(f.get("minQty",0.0) or 0.0)
        est = min_qty*max(price_now,0.0)
        return max(base_default,est) if est > 0 else base_default
    except: return base_default

# ================== Risk state ==================
def _default_risk_state():
    return {"date":_today_str(),"daily_pnl":0.0,"consecutive_losses":0,
            "trades_today":0,"blocked_until":None,"hourly_pnl":{},
            "last_signal_ts":None,"relax_success_count":0}

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str():
        s = _default_risk_state(); save_risk_state(s)
    for k,d in [("hourly_pnl",{}),("last_signal_ts",None),("relax_success_count",0)]:
        if k not in s: s[k] = d
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)
def register_trade_opened():
    s = load_risk_state()
    s["trades_today"] = int(s.get("trades_today",0))+1
    save_risk_state(s)

def _set_block(minutes, reason="risk"):
    s = load_risk_state()
    until = now_riyadh()+timedelta(minutes=minutes)
    s["blocked_until"] = until.isoformat(timespec="seconds")
    save_risk_state(s)
    _tg(f"⛔️ حظر مؤقت ({reason}) حتى {until.strftime('%H:%M')}.")

def _is_blocked():
    s  = load_risk_state()
    bu = s.get("blocked_until")
    if not bu: return False
    try:
        t = datetime.fromisoformat(bu)
        return now_riyadh() < t
    except: return False

def _hours_since_last_signal():
    s  = load_risk_state()
    ts = s.get("last_signal_ts")
    if not ts: return None
    try:
        dt = datetime.fromisoformat(ts)
        return max(0.0,(now_riyadh()-dt).total_seconds()/3600.0)
    except: return None

def _relax_level_current():
    s = load_risk_state()
    if int(s.get("relax_success_count",0)) >= RELAX_RESET_SUCCESS_TRADES: return 0
    hrs = _hours_since_last_signal()
    if hrs is None: return 0
    if hrs >= AUTO_RELAX_AFTER_HRS_2: return 2
    if hrs >= AUTO_RELAX_AFTER_HRS_1: return 1
    return 0

def register_trade_result(pnl_usdt):
    try:
        from risk_and_notify import register_trade_result as _rr
        _rr(float(pnl_usdt)); return
    except: pass

    s = load_risk_state()
    s["daily_pnl"]           = float(s.get("daily_pnl",0.0))+float(pnl_usdt)
    s["consecutive_losses"]  = 0 if pnl_usdt>0 else int(s.get("consecutive_losses",0))+1
    if pnl_usdt > 0:
        s["relax_success_count"] = int(s.get("relax_success_count",0))+1
        if s["relax_success_count"] >= RELAX_RESET_SUCCESS_TRADES:
            s["relax_success_count"] = 0
            s["last_signal_ts"]      = now_riyadh().isoformat(timespec="seconds")
            try: _tg("✅ صفقتان ناجحتان — عودة للوضع الطبيعي.")
            except: pass
    else:
        s["relax_success_count"] = 0

    hk = _hour_key(now_riyadh())
    s["hourly_pnl"][hk] = float(s["hourly_pnl"].get(hk,0.0))+float(pnl_usdt)

    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s); _set_block(90,"خسائر متتالية"); return
    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_day = now_riyadh().replace(hour=23,minute=59,second=0,microsecond=0)
        mins    = max(1,int((end_day-now_riyadh()).total_seconds()//60))
        save_risk_state(s); _set_block(mins,"تجاوز الخسارة اليومية"); return

    if os.getenv("HOURLY_DD_BLOCK_ENABLE","1").lower() in ("1","true","yes"):
        try:
            equity   = _fetch_balance_safe("USDT")
            hour_pnl = float(s["hourly_pnl"].get(hk,0.0))
            HOURLY_DD_PCT = float(os.getenv("HOURLY_DD_PCT","0.05"))
            if equity > 0 and hour_pnl <= -abs(HOURLY_DD_PCT)*equity:
                save_risk_state(s); _set_block(60,f"هبوط {HOURLY_DD_PCT*100:.1f}%/ساعة"); return
        except: pass
    save_risk_state(s)


# ================== manage_position ==================
def manage_position(symbol):
    pos = load_position(symbol)
    if not pos: return False

    if "amount" not in pos and "qty" in pos:
        try:    pos["amount"] = float(pos["qty"])
        except: pos["amount"] = float(pos.get("amount",0.0))

    targets = pos.get("targets") or []
    if targets and not pos.get("tp_hits"):
        pos["tp_hits"] = [False]*len(targets)
    if "stop_loss" not in pos:
        try:
            base = pos["symbol"].split("#")[0]
            px   = float(fetch_price(base) or pos.get("entry_price",0.0))
            data = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
            if data and px > 0:
                df     = _df(data)
                atr_abs= _atr_from_df(df)
                pos["stop_loss"] = float(max(0.0, px-1.0*atr_abs) if atr_abs and atr_abs>0
                                         else pos.get("entry_price",px)*0.97)
            else:
                pos["stop_loss"] = float(pos.get("entry_price",0.0)*0.97)
        except:
            pos["stop_loss"] = float(pos.get("entry_price",0.0)*0.97)
        save_position(symbol, pos)

    # [FIX-3 v4.2] تحقق من صحة المركز أولاً
    _symbol_check = pos.get("symbol", symbol)
    _is_dust_mgmt, _dust_reason_mgmt = _is_dust_position(_symbol_check, pos)
    if _is_dust_mgmt:
        logger.info(f"[manage] {symbol}: مركز غبار ({_dust_reason_mgmt}) — حذف")
        clear_position(symbol)
        return True

    base    = pos["symbol"].split("#")[0]
    try:
        current = float(fetch_price(base))
        if not (current > 0): raise ValueError("bad price")
    except: return False

    entry   = float(pos.get("entry_price",0.0))
    amount  = float(pos.get("amount", pos.get("qty",0.0)) or 0.0)
    targets = list(pos.get("targets") or [])
    partials= list(pos.get("partials") or [])
    variant = str(pos.get("variant","new"))
    mgmt    = _mgmt(variant)

    if amount <= 0:
        clear_position(symbol); return False

    if targets:
        tp_hits = list(pos.get("tp_hits") or [])
        if len(tp_hits) != len(targets):
            pos["tp_hits"] = [False]*len(targets)
            save_position(symbol, pos)

    # [FIX] جلب فلاتر مع معالجة الأخطاء
    try:
        f = fetch_symbol_filters(base) or {}
    except:
        f = {}
    tick = float(f.get("tickSize",0.00000001) or 0.00000001)

    # ── Break-even ──
    if BREAKEVEN_ENABLE and not pos.get("breakeven_done", False):
        try:
            atr_entry = float(pos.get("atr_entry", 0.0) or 0.0)
            if atr_entry > 0 and entry > 0:
                profit_needed = BREAKEVEN_TRIGGER_ATR * atr_entry
                if (current - entry) >= profit_needed:
                    be_price = entry * (1.0 + BREAKEVEN_OFFSET_PCT)
                    be_price = _round_to_tick(be_price, tick)
                    if be_price > float(pos.get("stop_loss",0.0)) and be_price < current:
                        pos["stop_loss"]     = float(be_price)
                        pos["breakeven_done"] = True
                        save_position(symbol, pos)
                        if STRAT_TG_SEND:
                            _tg(f"🔒 <b>Break-even</b> {symbol}: SL → <code>{be_price:.6f}</code>")
        except: pass

    # ── وقف زمني ──
    max_bars = pos.get("max_bars_to_tp1")
    if isinstance(max_bars,int) and max_bars > 0:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=now_riyadh().tzinfo)
            bar_min    = _tf_minutes(LTF_TIMEFRAME)
            bars_passed= int((now_riyadh()-opened_at)//timedelta(minutes=bar_min))
        except: bars_passed = 0

        if bars_passed >= max_bars and pos.get("tp_hits") and not pos["tp_hits"][0]:
            order, exit_px, sold_qty = _safe_sell(base, amount)
            if order and sold_qty > 0:
                pnl = (exit_px-entry)*sold_qty - (entry+exit_px)*sold_qty*(FEE_BPS_ROUNDTRIP/10000.0)
                p = load_position(symbol) or {}
                p["amount"] = max(0.0,float(p.get("amount",0.0))-float(sold_qty))
                save_position(symbol,p); pos = p
                if float(pos.get("amount",0.0)) <= 0.0:
                    close_trade(symbol,exit_px,pnl,reason="TIME_EXIT")
                    if STRAT_TG_SEND: _tg("⌛ خروج زمني")
                    return True
                else:
                    register_trade_result(pnl)
                    if STRAT_TG_SEND: _tg("⌛ خروج زمني جزئي")
                    return True

    # ── أقصى مدة احتفاظ ──
    max_hold_hours = float(pos.get("max_hold_hours") or mgmt.get("TIME_HRS") or 0)
    if max_hold_hours > 0:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=now_riyadh().tzinfo)
            hold_expired = (now_riyadh()-opened_at) >= timedelta(hours=max_hold_hours)
        except: hold_expired = False
        if hold_expired:
            order, exit_px, sold_qty = _safe_sell(base, amount)
            if order and sold_qty > 0:
                pnl = (exit_px-entry)*sold_qty - (entry+exit_px)*sold_qty*(FEE_BPS_ROUNDTRIP/10000.0)
                p = load_position(symbol) or {}
                p["amount"] = max(0.0,float(p.get("amount",0.0))-float(sold_qty))
                save_position(symbol,p); pos = p
                if float(pos.get("amount",0.0)) <= 0.0:
                    close_trade(symbol,exit_px,pnl,reason="TIME_HOLD_MAX")
                    if STRAT_TG_SEND: _tg("⌛ انتهاء مدة الاحتفاظ")
                    return True
                else:
                    register_trade_result(pnl); return True

    # ── أهداف + Partials + Trailing ──
    if targets and partials and len(targets)==len(partials):
        current = float(fetch_price(base))
        for i, tp in enumerate(targets):
            if not pos["tp_hits"][i] and current >= float(tp) and pos["amount"] > 0:
                part_qty = float(pos["amount"])*float(partials[i])
                if part_qty*current < MIN_NOTIONAL_USDT:
                    part_qty = float(pos["amount"])
                order, exit_px, sold_qty = _safe_sell(base, part_qty)
                if order and sold_qty > 0:
                    pnl = (exit_px-entry)*sold_qty - (entry+exit_px)*sold_qty*(FEE_BPS_ROUNDTRIP/10000.0)
                    p = load_position(symbol) or {}
                    p["amount"] = max(0.0,float(p.get("amount",0.0))-float(sold_qty))
                    if p.get("tp_hits") and i < len(p["tp_hits"]):
                        p["tp_hits"][i] = True
                    save_position(symbol,p); pos = p
                    register_trade_result(pnl)
                    if STRAT_TG_SEND: _tg(pos.get("messages",{}).get(f"tp{i+1}",f"🎯 TP{i+1} تحقق"))

                    if i == 0 and pos["amount"] > 0:
                        lock_raw = entry*(1.0+float(get_cfg(variant).get("LOCK_MIN_PROFIT_PCT",0.0)))
                        lock_sl  = _round_to_tick(lock_raw, tick)
                        if lock_sl > float(pos.get("stop_loss",0.0)):
                            pos["stop_loss"] = float(lock_sl)
                            save_position(symbol,pos)
                            if STRAT_TG_SEND:
                                _tg(f"🔒 وقف حماية ربح: <code>{lock_sl:.6f}</code>")

                    if i >= 0 and pos["amount"] > 0:
                        data_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
                        if data_atr:
                            df_atr  = _df(data_atr)
                            atr_val2= _atr_from_df(df_atr)
                            if atr_val2 and atr_val2 > 0:
                                trail_mult = float(mgmt.get("TRAIL_ATR",1.0))
                                if i >= 1: trail_mult = max(0.7, trail_mult*0.85)
                                new_sl = _round_to_tick(float(current)-trail_mult*float(atr_val2), tick)
                                if new_sl > float(pos.get("stop_loss",0.0))*(1+TRAIL_MIN_STEP_RATIO):
                                    pos["stop_loss"] = float(new_sl)
                                    save_position(symbol,pos)
                                    if STRAT_TG_SEND:
                                        _tg(f"🧭 Trailing SL {symbol} → <code>{new_sl:.6f}</code>")
                    return True  # [FIX-2 v4.2] early return بعد أول TP

    # ── Trailing عام ──
    if mgmt.get("TRAIL_AFTER_TP1") and pos["amount"] > 0 and any(pos.get("tp_hits",[])):
        data_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
        if data_atr:
            df_atr  = _df(data_atr)
            atr_val3= _atr_from_df(df_atr)
            if atr_val3 and atr_val3 > 0:
                current  = float(fetch_price(base))
                new_sl   = _round_to_tick(float(current)-float(mgmt.get("TRAIL_ATR",1.0))*float(atr_val3), tick)
                if new_sl > float(pos.get("stop_loss",0.0))*(1+TRAIL_MIN_STEP_RATIO):
                    pos["stop_loss"] = float(new_sl)
                    save_position(symbol,pos)

    # ── وقف نهائي ──
    current = float(fetch_price(base))
    if current <= float(pos.get("stop_loss",0.0)) and pos["amount"] > 0:
        order, exit_px, sold_qty = _safe_sell(base, float(pos["amount"]))
        if order and sold_qty > 0:
            pnl = (exit_px-entry)*sold_qty - (entry+exit_px)*sold_qty*(FEE_BPS_ROUNDTRIP/10000.0)
            p = load_position(symbol) or {}
            p["amount"] = max(0.0,float(p.get("amount",0.0))-float(sold_qty))
            save_position(symbol,p); pos = p
            if float(pos.get("amount",0.0)) <= 0.0:
                close_trade(symbol,exit_px,pnl,reason="SL")
                if STRAT_TG_SEND: _tg(pos.get("messages",{}).get("sl","🛑 SL"))
                return True
            else:
                register_trade_result(pnl)
                if STRAT_TG_SEND:
                    _tg(f"🛑 SL جزئي {symbol} @ <code>{exit_px:.6f}</code>")
                return True  # [FIX-2 v4.2] early return

    return False

# ================== close_trade ==================
def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos: return
    closed = load_closed_positions()
    try:    entry  = float(pos.get("entry_price",0.0))
    except: entry  = 0.0
    try:    amount = float(pos.get("amount",0.0))
    except: amount = 0.0
    pnl_pct = ((float(exit_price)/entry)-1.0) if entry else 0.0
    tp_hits = {}
    try:
        if "targets" in pos and "tp_hits" in pos and isinstance(pos["tp_hits"],list):
            for i,hit in enumerate(pos["tp_hits"],start=1):
                tp_hits[f"tp{i}_hit"] = bool(hit)
    except: pass
    closed.append({
        "symbol":pos.get("symbol",symbol),
        "entry_price":float(entry),"exit_price":float(exit_price),
        "amount":float(amount),"profit":float(pnl_net),
        "pnl_pct":round(float(pnl_pct),6),"reason":reason,
        "opened_at":pos.get("opened_at"),
        "closed_at":now_riyadh().isoformat(timespec="seconds"),
        "variant":pos.get("variant"),"score":pos.get("score"),
        "pattern":pos.get("pattern"),"entry_reason":pos.get("reason"),
        **tp_hits
    })
    save_closed_positions(closed)
    register_trade_result(float(pnl_net))
    try:
        if STRAT_TG_SEND:
            _tg(
                f"🔻 خروج كامل {symbol}\n"
                f"🚪 السبب: <code>{reason}</code>\n"
                f"💵 P&L: <code>{float(pnl_net):+.2f} USDT</code>\n"
                f"🎯 دخول: <code>{float(entry):.6f}</code> • خروج: <code>{float(exit_price):.6f}</code>"
            )
    except: pass
    clear_position(symbol)

def close_trend(symbol):
    pos = load_position(symbol)
    if not pos: return False, "no_pos"
    base = pos.get("symbol",symbol).split("#")[0]
    data = get_ohlcv_cached(base, STRAT_HTF_TIMEFRAME, 200)
    if not data or len(data) < 60: return False, "no_htf"
    try:
        df = _df(data); df["ema50_htf"] = ema(df["close"],HTF_EMA_TREND_PERIOD)
        row_close = float(df["close"].iloc[-2])
        row_ema   = float(df["ema50_htf"].iloc[-2])
    except: return False, "no_htf"
    br = _get_breadth_ratio_cached(); eff_min = _breadth_min_auto()
    market_weak = (br is None) or (br < max(0.58,eff_min))
    if not (row_close < row_ema and market_weak): return False, "hold"
    entry  = float(pos.get("entry_price",0.0))
    amount = float(pos.get("amount",pos.get("qty",0.0)) or 0.0)
    if amount <= 0.0:
        clear_position(symbol); return False,"hold"
    order, exit_px, sold_qty = _safe_sell(base, amount)
    if not order or not exit_px or sold_qty <= 0.0: return False,"hold"
    fees    = (entry+float(exit_px))*float(sold_qty)*(FEE_BPS_ROUNDTRIP/10000.0)
    pnl_net = (float(exit_px)-entry)*float(sold_qty)-fees
    p = load_position(symbol) or {}
    p["amount"] = max(0.0,float(p.get("amount",0.0))-float(sold_qty))
    save_position(symbol,p)
    if float(p.get("amount",0.0)) <= 0.0:
        close_trade(symbol,float(exit_px),pnl_net,reason="CLOSE_TREND_EMA50")
        if STRAT_TG_SEND: _tg(f"🧭 CloseTrend {symbol} @ <code>{float(exit_px):.6f}</code>")
        return True,"closed"
    else:
        register_trade_result(pnl_net); return True,"partial"

# ================== Reports ==================
def _fmt_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i,c in enumerate(r): widths[i]=max(widths[i],len(str(c)))
    def fmt(r): return "  ".join(str(c).ljust(widths[i]) for i,c in enumerate(r))
    return "<pre>"+fmt(headers)+"\n"+"\n".join(fmt(r) for r in rows)+"</pre>"

def build_daily_report_text():
    closed = load_closed_positions(); today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at","")).startswith(today)]
    s      = load_risk_state()
    def f6(x):
        try: return "{:,.6f}".format(float(x))
        except: return str(x)
    def f2(x):
        try: return "{:,.2f}".format(float(x))
        except: return str(x)
    if not todays:
        return (f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم.\n"
                f"PnL: {float(s.get('daily_pnl',0.0)):.2f}$ • صفقات: {int(s.get('trades_today',0))}")
    total_pnl = sum(float(t.get("profit",0.0)) for t in todays)
    wins      = [t for t in todays if float(t.get("profit",0.0))>0]
    win_rate  = round(100*len(wins)/max(1,len(todays)),2)
    headers   = ["الرمز","الكمية","دخول","خروج","P/L$","P/L%","Score","نمط","Exit"]
    rows = []
    for t in todays:
        rows.append([
            t.get("symbol","-"), f6(t.get("amount",0)),
            f6(t.get("entry_price",0)), f6(t.get("exit_price",0)),
            f2(t.get("profit",0)),
            f"{round(float(t.get('pnl_pct',0))*100,2)}%",
            str(t.get("score","-")), t.get("pattern","-"), t.get("reason","-")
        ])
    table = _fmt_table(rows, headers)
    bu_txt = "سماح"
    try:
        bu = s.get("blocked_until")
        if bu:
            dt = datetime.fromisoformat(bu)
            bu_txt = f"محظور حتى {dt.strftime('%H:%M')}"
    except: pass
    return (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • P&L: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b> • الحالة: {bu_txt}\n"
        f"الخسارة اليومية: <b>{float(s.get('daily_pnl',0.0)):.2f}$</b>\n"
    ) + table

def maybe_emit_reject_summary():
    if not _REJ_SUMMARY: return
    try:
        top = sorted(_REJ_SUMMARY.items(),key=lambda kv:kv[1],reverse=True)[:5]
        print(f"[summary] rejects_top5: {', '.join(f'{k}:{v}' for k,v in top)}")
    except: pass
    finally: _REJ_SUMMARY.clear()

def get_last_reject(symbol):
    if symbol in _LAST_REJECT: return _LAST_REJECT[symbol]
    base, variant = _split_symbol_variant(symbol)
    for key in (f"{base}|{variant}", base):
        if key in _LAST_REJECT: return _LAST_REJECT[key]
    return None

def check_signal_debug(symbol):
    r = check_signal(symbol)
    if isinstance(r,dict) and r.get("decision")=="buy": return r,["buy_ok"]
    last = get_last_reject(symbol)
    if last:
        stg = last.get("stage","no_buy"); det = last.get("details",{})
        return None,[f"{stg}:{det}"]
    return None,["no_buy"]
