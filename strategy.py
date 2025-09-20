# -*- coding: utf-8 -*-
"""
strategy.py — Spot-only (v3.4 Soft+)
- توافق كامل مع main.py (check_signal / execute_buy / manage_position / build_daily_report_text …)
- Preset Soft+: تخفيف مدروس للصرامة الافتراضية (Score/ATR/Notional/RVOL) دون المساس بالجودة.
- تصحيح واكتمال: توقيع execute_buy، متغيرات ENV الافتراضية، دوال مفقودة لاحقًا، وتعامل واضح مع Soft-Schedule.
- كاش OHLCV للجولة + مِقاييس أداء + كاش HTF.
- Retry/Backoff لاستدعاءات OHLCV.
- Position sizing ديناميكي + Soft sizing (وقت/سوق).
- Circuit breaker يومي/بالساعة + Auto-Relax (6/12 ساعة) مع Reset بعد صفقتين ناجحتين.
- Market Breadth Guard (ديناميكي حسب BTC 4h) + استثناء القائد.
- حارس Parabolic/Exhaustion (RSI/المسافة عن EMA50).
- أهداف متعددة + Partials متكيّفة.
- Dynamic Max Bars to TP1 + ملخص رفضات دوري + تشخيص.
- DRY_RUN: تنفيذ وهمي للأوامر عبر ENV.

ملاحظة: هذا الجزء يحتوي على الهيدر والتهيئة والكونستانتس والهيلبرز فقط؛
المنطق التشغيلي يُكمل في الأجزاء التالية خطوة بخطوة.
"""

from __future__ import annotations

import os, json, requests, logging, time, math, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import (
    TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME
)

# ===================== معلومات/نسخة =====================
VERSION = "3.4-soft+"

# ===================== ENV helpers =====================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

# ===== MTF strict flag (ساعة/4س/يومي) =====
try:
    ENABLE_MTF_STRICT
except NameError:
    ENABLE_MTF_STRICT = _env_bool("ENABLE_MTF_STRICT", False)

# ===== لوج الاستراتيجية =====
logger = logging.getLogger("strategy")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEBUG_LOG_SIGNALS = _env_bool("DEBUG_LOG_SIGNALS", False)

# ================== إعدادات عامة/ثوابت ==================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# إشعارات داخل الاستراتيجية (فصلًا عن رسائل main.py)
STRAT_TG_SEND = _env_bool("STRAT_TG_SEND", False)

# أطر زمنية (من config)
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME

# مؤشرات/نوافذ
EMA_FAST, EMA_SLOW, EMA_TREND, EMA_LONG = 9, 21, 50, 200
VOL_MA, SR_WINDOW = 20, 50
ATR_PERIOD = 14
RVOL_WINDOW = 20
NR_WINDOW = 10
NR_FACTOR = 0.75
HTF_EMA_TREND_PERIOD = 50
HTF_SR_WINDOW = 50

# إدارة الصفقة والقيود العامة
TP1_FRACTION = 0.5
MIN_NOTIONAL_USDT = _env_float("MIN_NOTIONAL_USDT", 10.0)   # حد تنفيذ البورصة
TRAIL_MIN_STEP_RATIO = _env_float("TRAIL_MIN_STEP_RATIO", 0.001)

MAX_TRADES_PER_DAY       = _env_int("MAX_TRADES_PER_DAY", 20)
MAX_CONSEC_LOSSES        = _env_int("MAX_CONSEC_LOSSES", 3)
DAILY_LOSS_LIMIT_USDT    = _env_float("DAILY_LOSS_LIMIT_USDT", 200.0)

# تحجيم أساسي/حد أدنى للصفقة + DRY RUN
TRADE_BASE_USDT  = _env_float("TRADE_BASE_USDT", TRADE_AMOUNT_USDT)
MIN_TRADE_USDT   = _env_float("MIN_TRADE_USDT", 10.0)
DRY_RUN          = _env_bool("DRY_RUN", False)

# ===== Throttle / Idempotency (Quick Wins) =====
BUY_THROTTLE_SEC = _env_float("BUY_THROTTLE_SEC", 3.0)  # منع تنفيذ شراءين خلال x ثواني
_LAST_BUY_TS = 0.0

def _buy_allowed_now() -> tuple[bool, float]:
    global _LAST_BUY_TS
    now = time.time()
    if now - _LAST_BUY_TS < BUY_THROTTLE_SEC:
        return False, max(0.0, BUY_THROTTLE_SEC - (now - _LAST_BUY_TS))
    _LAST_BUY_TS = now
    return True, 0.0

# مفاتيح ميزات
USE_EMA200_TREND_FILTER   = _env_bool("USE_EMA200_TREND_FILTER", True)
ENABLE_GOLDEN_CROSS_ENTRY = _env_bool("ENABLE_GOLDEN_CROSS_ENTRY", True)
GOLDEN_CROSS_RVOL_BOOST   = _env_float("GOLDEN_CROSS_RVOL_BOOST", 1.10)

# درجات/حدود (Soft+ افتراضات ألطف)
SCORE_THRESHOLD = _env_int("SCORE_THRESHOLD", 33)  # كان 35

# ======= Auto-Relax =======
AUTO_RELAX_AFTER_HRS_1 = _env_float("AUTO_RELAX_AFTER_HRS_1", 6)
AUTO_RELAX_AFTER_HRS_2 = _env_float("AUTO_RELAX_AFTER_HRS_2", 12)
RELAX_RVOL_DELTA_1 = _env_float("RELAX_RVOL_DELTA_1", 0.05)
RELAX_RVOL_DELTA_2 = _env_float("RELAX_RVOL_DELTA_2", 0.10)
RELAX_ATR_MIN_SCALE_1 = _env_float("RELAX_ATR_MIN_SCALE_1", 0.90)
RELAX_ATR_MIN_SCALE_2 = _env_float("RELAX_ATR_MIN_SCALE_2", 0.85)
RELAX_RESET_SUCCESS_TRADES = _env_int("RELAX_RESET_SUCCESS_TRADES", 2)

# ======= Market Breadth =======
BREADTH_MIN_RATIO = _env_float("BREADTH_MIN_RATIO", 0.60)  # حد أساسي (يُضبط ديناميكيًا)
BREADTH_TF = os.getenv("BREADTH_TF", "1h")
BREADTH_TTL_SEC = _env_int("BREADTH_TTL_SEC", 180)
BREADTH_SYMBOLS_ENV = os.getenv("BREADTH_SYMBOLS", "")

# Soft breadth sizing (يُستخدم لاحقًا داخل execute_buy)
SOFT_BREADTH_ENABLE = _env_bool("SOFT_BREADTH_ENABLE", True)
SOFT_BREADTH_SIZE_SCALE = _env_float("SOFT_BREADTH_SIZE_SCALE", 0.50)

# Exhaustion
EXH_RSI_MAX = _env_float("EXH_RSI_MAX", 76)
EXH_EMA50_DIST_ATR = _env_float("EXH_EMA50_DIST_ATR", 2.8)

# Multi-targets
ENABLE_MULTI_TARGETS = _env_bool("ENABLE_MULTI_TARGETS", True)
MAX_TP_COUNT = _env_int("MAX_TP_COUNT", 5)
TP_ATR_MULTS_TREND = tuple(float(x) for x in os.getenv("TP_ATR_MULTS_TREND", "1.2,2.2,3.5,4.5,6.0").split(","))
TP_ATR_MULTS_VBR   = tuple(float(x) for x in os.getenv("TP_ATR_MULTS_VBR",   "0.6,1.2,1.8,2.4").split(","))

# Dynamic Max Bars to TP1
USE_DYNAMIC_MAX_BARS = _env_bool("USE_DYNAMIC_MAX_BARS", True)
MAX_BARS_BASE = _env_int("MAX_BARS_TO_TP1_BASE", 12)

# Tunables عبر ENV (تخفيف رفض ATR/RVOL/Notional) — Soft+ افتراضات ألطف قليلًا
MIN_BAR_NOTIONAL_USD = _env_float("MIN_BAR_NOTIONAL_USD", 22000)   # كان 25k
ATR_MIN_BASE = _env_float("ATR_MIN_FOR_TREND_BASE", 0.0019)        # كان 0.0020
ATR_MIN_NEW  = _env_float("ATR_MIN_FOR_TREND_NEW",  0.0026)
ATR_MIN_BRT  = _env_float("ATR_MIN_FOR_TREND_BRT",  0.0022)
RVOL_MIN_NEW = _env_float("RVOL_MIN_NEW", 1.20)                     # كان 1.25
RVOL_MIN_BRT = _env_float("RVOL_MIN_BRT", 1.25)                     # كان 1.30

# ======= كاش HTF/OHLCV + مِقاييس =======
_HTF_CACHE: Dict[str, Dict[str, Any]] = {}
_HTF_TTL_SEC = _env_int("HTF_CACHE_TTL_SEC", 150)

_OHLCV_CACHE: Dict[tuple, list] = {}  # key=(symbol, tf, bars)
_METRICS = {
    "ohlcv_api_calls": 0,
    "ohlcv_cache_hits": 0,
    "ohlcv_cache_misses": 0,
    "htf_cache_hits": 0,
    "htf_cache_misses": 0
}

# ======= عدّاد رفضات الجولة (لتليين محلي) + ملخص عام =======
_REJ_COUNTS = {"atr_low": 0, "rvol": 0, "notional_low": 0}
_REJ_SUMMARY: Dict[str, int] = {}

# ===== Soft schedule (وقت + سوق) =====
SOFT_SCHEDULE_ENABLE   = _env_bool("SOFT_SCHEDULE_ENABLE", True)
# 12 ساعة افتراضيًا (بتوقيت الرياض): 00:00-06:00 و 12:00-18:00
SOFT_SCHEDULE_HRS      = os.getenv("SOFT_SCHEDULE_HRS", "00:00-06:00,12:00-18:00")
# أيام الأسبوع المفعّل فيها (0=Mon .. 6=Sun)
SOFT_SCHEDULE_WEEKDAYS = os.getenv("SOFT_SCHEDULE_WEEKDAYS", "0,1,2,3,4,5,6")
# عوامل تقليص الحجم
SOFT_SCALE_TIME_ONLY   = _env_float("SOFT_SCALE_TIME_ONLY", 0.85)
SOFT_SCALE_MARKET_WEAK = _env_float("SOFT_SCALE_MARKET_WEAK", 0.70)
SOFT_SEVERITY_STEP     = _env_float("SOFT_SEVERITY_STEP", 0.05)
SOFT_MSG_ENABLE        = _env_bool("SOFT_MSG_ENABLE", True)

# ================== Helpers & أساسيات ==================
def reset_cycle_cache():
    """يمسح كاش OHLCV ويصفر مِقاييس الجولة + عدّادات الرفض — تُنادى من main.py بداية كل جولة."""
    _OHLCV_CACHE.clear()
    for k in _METRICS: _METRICS[k] = 0
    for k in _REJ_COUNTS: _REJ_COUNTS[k] = 0
    # _REJ_SUMMARY لا نصفره داخل الجولة؛ الملخص دوري منفصل

def metrics_snapshot() -> dict:
    return dict(_METRICS)

def metrics_format() -> str:
    m = _METRICS
    return (
        "📈 <b>Metrics (this round)</b>\n"
        f"- OHLCV cache hits/misses: <b>{m['ohlcv_cache_hits']}/{m['ohlcv_cache_misses']}</b>\n"
        f"- OHLCV API calls: <b>{m['ohlcv_api_calls']}</b>\n"
        f"- HTF hits/misses: <b>{m['htf_cache_hits']}/{m['htf_cache_misses']}</b>"
    )

def _tg(text: str, parse_mode: str = "HTML"):
    if not STRAT_TG_SEND: return
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        if parse_mode: data["parse_mode"] = parse_mode
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass

def now_riyadh() -> datetime: 
    return datetime.now(RIYADH_TZ)

def _today_str() -> str: 
    return now_riyadh().strftime("%Y-%m-%d")

def _hour_key(dt: datetime) -> str: 
    return dt.strftime("%Y-%m-%d %H")

def _atomic_write(path: str, data: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _read_json(path: str, default: Any):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _df(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    try:
        if len(df) and df["timestamp"].iloc[-1] < 10**12:
            df["timestamp"] = df["timestamp"] * 1000
    except Exception:
        pass
    return df

def _finite_or(default: float, *vals: Any) -> float:
    """يرجع أول قيمة float نهائية (ليست NaN/Inf) من vals، وإلا يُرجع default."""
    for v in vals:
        try:
            f = float(v)
            if math.isfinite(f):
                return f
        except Exception:
            pass
    return default

def _split_symbol_variant(symbol: str) -> Tuple[str, str]:
    """يدعم صيغ مثل 'ALT/USDT#brt' ويعيد (base, variant)."""
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower()
        if variant not in ("old", "new", "srr", "brt", "vbr"):
            variant = "new"
        return base, variant
    return symbol, "new"

# ================== تخزين الصفقات ==================
def _pos_path(symbol: str) -> str:
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return f"{POSITIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol: str):
    return _read_json(_pos_path(symbol), None)

def save_position(symbol: str, position: dict) -> None:
    _atomic_write(_pos_path(symbol), position)

def clear_position(symbol: str) -> None:
    try:
        p = _pos_path(symbol)
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass

def count_open_positions() -> int:
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions() -> list:
    return _read_json(CLOSED_POSITIONS_FILE, [])

def save_closed_positions(lst: list) -> None:
    _atomic_write(CLOSED_POSITIONS_FILE, lst)

# ================== مؤشرات ==================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.where(d > 0, 0.0)
    loss = -d.where(d < 0, 0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    rs = ag / al
    return 100 - (100 / (1 + rs))

def macd_cols(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df["ema_fast"], df["ema_slow"] = ema(df["close"], fast), ema(df["close"], slow)
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema9"]   = ema(df["close"], EMA_FAST)
    df["ema21"]  = ema(df["close"], EMA_SLOW)
    df["ema50"]  = ema(df["close"], EMA_TREND)
    df["ema200"] = ema(df["close"], EMA_LONG)
    df["rsi"] = rsi(df["close"], 14)
    # متوسط حجم آمن (تجنب القسمة على صفر لاحقًا)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA).mean().replace(0, 1e-9)
    df = macd_cols(df)
    return df

def _ensure_ltf_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df.copy())

    # VWAP اليومي (بتوقيت الرياض)
    ts = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Riyadh")
    day_changed = ts.dt.date
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tpv"] = tp * df["volume"]
    df["cum_vol"] = df.groupby(day_changed)["volume"].cumsum()
    df["cum_tpv"] = df.groupby(day_changed)["tpv"].cumsum()
    df["vwap"] = (df["cum_tpv"] / df["cum_vol"]).replace([pd.NA, pd.NaT, 0], None)

    # RVOL الآمن
    vol_ma = df["volume"].rolling(RVOL_WINDOW).mean().replace(0, 1e-9)
    df["rvol"] = df["volume"] / vol_ma

    # نطاق ضيق (NR)
    rng = (df["high"] - df["low"]).abs()
    rng_ma = rng.rolling(NR_WINDOW).mean().replace(0, 1e-9)
    df["is_nr"] = rng < (NR_FACTOR * rng_ma)

    # أجسام الشموع
    df["body"] = (df["close"] - df["open"]).abs()
    df["avg_body20"] = df["body"].rolling(20).mean()
    return df

def _atr_from_df(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    c = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - c).abs(),
        (df["low"]  - c).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-2])

# ===== Swing/SR =====
def _swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[Optional[float], Optional[float]]:
    highs, lows = df["high"], df["low"]
    idx = len(df) - 3
    swing_high = swing_low = None
    for i in range(max(0, idx-10), idx+1):
        if i-left < 0 or i+right >= len(df): 
            continue
        if highs[i] == max(highs[i-left:i+right+1]): 
            swing_high = float(highs[i])
        if lows[i]  == min(lows[i-left:i+right+1]): 
            swing_low  = float(lows[i])
    return swing_high, swing_low

def _bullish_engulf(prev: pd.Series, cur: pd.Series) -> bool:
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and \
           (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])

def get_sr_on_closed(df: pd.DataFrame, window: int = SR_WINDOW) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < window + 3: 
        return None, None
    df_prev = df.iloc[:-1]
    w = min(window, len(df_prev))
    resistance = df_prev["high"].rolling(w, min_periods=max(5, w//3)).max().iloc[-1]
    support    = df_prev["low"].rolling(w,  min_periods=max(5, w//3)).min().iloc[-1]
    if pd.isna(resistance) or pd.isna(support): 
        return None, None
    return float(support), float(resistance)

def recent_swing(df: pd.DataFrame, lookback: int = 60) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < lookback + 5: 
        return None, None
    seg = df.iloc[-(lookback+1):-1]
    hhv = seg["high"].max()
    llv = seg["low"].min()
    if pd.isna(hhv) or pd.isna(llv) or hhv <= llv: 
        return None, None
    return float(hhv), float(llv)

def _rolling_sr(symbol: str, tf: str, window: int, bars: int = 300) -> Tuple[Optional[float], Optional[float]]:
    data = get_ohlcv_cached(symbol, tf, bars)
    if not data: 
        return None, None
    df = _df(data)
    if len(df) < window + 3: 
        return None, None
    df_prev = df.iloc[:-1]
    w = min(window, len(df_prev))
    res = df_prev["high"].rolling(w, min_periods=max(5, w//3)).max().iloc[-1]
    sup = df_prev["low"].rolling(w,  min_periods=max(5, w//3)).min().iloc[-1]
    if pd.isna(res) or pd.isna(sup): 
        return None, None
    return float(sup), float(res)

# مستويات SR متعددة (اسم، TF، نافذة، مضروب قرب/ATR)
SR_LEVELS_CFG = [
    ("LTF_H1", "1h", 50, 0.8),
    ("HTF_H4", "4h", 50, 1.2),
    ("HTF_D1", "1d", 30, 1.5),
]

def get_sr_multi(symbol: str) -> Dict[str, Dict[str, Any]]:
    levels: Dict[str, Dict[str, Any]] = {}
    base = symbol.split("#")[0]
    for name, tf, window, near_mult in SR_LEVELS_CFG:
        try:
            sup, res = _rolling_sr(base, tf, window)
            if sup or res:
                levels[name] = {"tf": tf, "support": sup, "resistance": res, "near_mult": near_mult}
        except Exception:
            pass
    return levels

# ===== MACD/RSI Gate =====
def macd_rsi_gate(prev_row: pd.Series, closed_row: pd.Series, policy: Optional[str]):
    if not policy: 
        return True
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
    return k >= 2  # balanced

# ================== كاش OHLCV للجولة + Retry ==================
def _retry_fetch_ohlcv(symbol: str, tf: str, bars: int, attempts: int = 3, base_wait: float = 1.2, max_wait: float = 6.0):
    last_exc = None
    for i in range(attempts):
        try:
            data = fetch_ohlcv(symbol, tf, bars)
            return data
        except Exception as e:
            last_exc = e
            wait = min(max_wait, base_wait * (2 ** i)) * (0.9 + 0.2 * np.random.rand())
            time.sleep(wait)
    if last_exc:
        raise last_exc
    return None

def api_fetch_ohlcv(symbol: str, tf: str, bars: int) -> list:
    _METRICS["ohlcv_api_calls"] += 1
    return _retry_fetch_ohlcv(symbol, tf, bars)

def get_ohlcv_cached(symbol: str, tf: str, bars: int) -> list:
    key = (symbol, tf, bars)
    if key in _OHLCV_CACHE:
        _METRICS["ohlcv_cache_hits"] += 1
        return _OHLCV_CACHE[key]
    _METRICS["ohlcv_cache_misses"] += 1
    data = api_fetch_ohlcv(symbol, tf, bars)
    if data:
        _OHLCV_CACHE[key] = data
    return data

# ================== HTF سياق ==================
def _get_htf_context(symbol: str):
    base = symbol.split("#")[0]
    now = now_riyadh()
    ent = _HTF_CACHE.get(base)
    if ent and (now - ent["t"]).total_seconds() <= _HTF_TTL_SEC:
        _METRICS["htf_cache_hits"] += 1
        return ent["ctx"]

    _METRICS["htf_cache_misses"] += 1
    data = get_ohlcv_cached(base, HTF_TIMEFRAME, 200)
    if not data:
        return None
    df = _df(data)
    df["ema50_htf"] = ema(df["close"], HTF_EMA_TREND_PERIOD)
    if len(df) < HTF_SR_WINDOW + 3:
        return None

    df_prev = df.iloc[:-2]
    w = min(HTF_SR_WINDOW, len(df_prev))

    resistance_raw = df_prev["high"].rolling(w).max().iloc[-1]
    support_raw    = df_prev["low"].rolling(w).min().iloc[-1]
    resistance = _finite_or(None, resistance_raw)
    support    = _finite_or(None, support_raw)

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
        def _tf_info(tf: str, bars: int = 160):
            try:
                d = get_ohlcv_cached(base, tf, bars)
                if not d or len(d) < 80:
                    return None
                _dfx = _df(d)
                _dfx[f"ema{HTF_EMA_TREND_PERIOD}"] = ema(_dfx["close"], HTF_EMA_TREND_PERIOD)
                row = _dfx.iloc[-2]
                e = _finite_or(float(row["close"]), float(_dfx[f"ema{HTF_EMA_TREND_PERIOD}"].iloc[-2]))
                return {
                    "tf": tf,
                    "price": float(row["close"]),
                    "ema": float(e),
                    "trend_up": bool(float(row["close"]) > float(e))
                }
            except Exception:
                return None

        mtf = {
            HTF_TIMEFRAME: {
                "tf": HTF_TIMEFRAME,
                "price": ctx["close"],
                "ema": ctx["ema50_now"],
                "trend_up": bool(ctx["close"] > ctx["ema50_now"])
            }
        }
        for tf in ("1h", "4h"):
            info = _tf_info(tf)
            if info:
                mtf[tf] = info
        ctx["mtf"] = mtf

    _HTF_CACHE[base] = {"t": now, "ctx": ctx}
    return ctx

# ================== Breadth Guard ==================
_BREADTH_CACHE: Dict[str, Any] = {"t": 0.0, "ratio": None}

def _breadth_refs() -> List[str]:
    if BREADTH_SYMBOLS_ENV.strip():
        out: List[str] = []
        for s in BREADTH_SYMBOLS_ENV.split(","):
            s = s.strip()
            if s:
                out.append(s.replace("-", "/").upper().split("#")[0])
        return out
    uniq: List[str] = []
    seen: set = set()
    for s in SYMBOLS:
        base = s.split("#")[0].replace("-", "/").upper()
        if base not in seen:
            uniq.append(base); seen.add(base)
        if len(uniq) >= 12:
            break
    return uniq or ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT"]

_TF_MIN = {"5m":5, "15m":15, "30m":30, "1h":60, "2h":120, "4h":240, "1d":1440}
def _tf_minutes(tf: str) -> int:
    return int(_TF_MIN.get(tf.lower(), 60))

def _row_is_recent_enough(df: pd.DataFrame, tf: str, bars_back: int = 2) -> bool:
    try:
        last_ts = int(df["timestamp"].iloc[-bars_back])
        if last_ts < 10**12:
            last_ts *= 1000
        now_ms = int(time.time() * 1000)
        max_age_ms = 2 * _tf_minutes(tf) * 60 * 1000
        return (now_ms - last_ts) <= max_age_ms
    except Exception:
        return False

def _compute_breadth_ratio() -> Optional[float]:
    refs = _breadth_refs()
    if not refs:
        return None
    ok, tot = 0, 0
    for sym in refs:
        try:
            d = get_ohlcv_cached(sym, BREADTH_TF, 120)
            if not d or len(d) < 60:
                continue
            df = _df(d)
            if not _row_is_recent_enough(df, BREADTH_TF, bars_back=2):
                continue
            df["ema50"] = ema(df["close"], 50)
            row = df.iloc[-2]
            c = float(row["close"]); e = float(row["ema50"])
            if math.isfinite(c) and math.isfinite(e) and c > 0 and e > 0:
                tot += 1
                if c > e:
                    ok += 1
        except Exception:
            continue
    if tot < 5:
        return None
    ratio = ok / float(tot)
    return ratio if ratio > 0.05 else None

def _get_breadth_ratio_cached() -> Optional[float]:
    now_s = time.time()
    if (_BREADTH_CACHE["ratio"] is not None) and ((now_s - _BREADTH_CACHE["t"]) <= BREADTH_TTL_SEC):
        return _BREADTH_CACHE["ratio"]
    r = _compute_breadth_ratio()
    _BREADTH_CACHE["ratio"] = r
    _BREADTH_CACHE["t"] = now_s
    return r

def _effective_breadth_min() -> float:
    base = float(BREADTH_MIN_RATIO)
    try:
        d = get_ohlcv_cached("BTC/USDT", "4h", 220)
        if not d or len(d) < 100:
            return base
        df = _df(d)
        df["ema50"] = ema(df["close"], 50)
        row = df.iloc[-2]
        above = float(row["close"]) > float(df["ema50"].iloc[-2])
        rsi_btc = float(rsi(df["close"], 14).iloc[-2])
        if above and rsi_btc >= 55:
            return max(0.40, base - 0.15)
        if (not above) or rsi_btc <= 45:
            return min(0.75, base + 0.10)
    except Exception:
        pass
    return base

def _btc_strong_on_4h() -> bool:
    try:
        d = get_ohlcv_cached("BTC/USDT", "4h", 220)
        if not d or len(d) < 100:
            return False
        df = _df(d)
        df["ema50"] = ema(df["close"], 50)
        row = df.iloc[-2]
        above = float(row["close"]) > float(df["ema50"].iloc[-2])
        rsi_btc = float(rsi(df["close"], 14).iloc[-2])
        return bool(above and rsi_btc >= 55)
    except Exception:
        return False

def _breadth_min_auto() -> float:
    try:
        eff = _effective_breadth_min()
        return float(max(0.40, min(0.70, eff)))
    except Exception:
        return float(BREADTH_MIN_RATIO)

# ===== سلوك القائد =====
def _is_relative_leader_vs_btc(symbol_base: str, tf: str = "1h", lookback: int = 24, edge: float = 0.02) -> bool:
    try:
        base = symbol_base.split("#")[0].replace("-", "/").upper()
        if tf.lower() not in _TF_MIN:
            tf = "1h"
        bars_need = max(lookback + 10, 40)

        d1 = get_ohlcv_cached(base, tf, bars_need)
        d2 = get_ohlcv_cached("BTC/USDT", tf, bars_need)
        if not d1 or not d2:
            return False

        s1 = _df(d1)["close"].iloc[-(lookback+1):-1].astype(float)
        s2 = _df(d2)["close"].iloc[-(lookback+1):-1].astype(float)
        if len(s1) < 5 or len(s2) < 5:
            return False

        if len(s1) != len(s2):
            n = min(len(s1), len(s2))
            s1, s2 = s1.iloc[-n:], s2.iloc[-n:]

        r1 = s1.pct_change()
        r2 = s2.pct_change()
        rel = (r1 - r2).dropna().mean()
        return float(rel or 0.0) >= float(edge)
    except Exception:
        return False

# ================== أدوات رفض/تمرير + تتبُّع الجولة ==================
_LAST_REJECT: Dict[str, Any] = {}
_LAST_ENTRY_BAR_TS: Dict[str, int] = {}
_SYMBOL_LAST_TRADE_AT: Dict[str, datetime] = {}
_CURRENT_SYMKEY: Optional[str] = None  # لتسجيل آخر رمز/نسخة أثناء الفحص

def _rej(stage: str, **kv):
    """
    تسجيل سبب رفض مركّز:
    - يُحدّث عدّادات الجولة والمحلي العام.
    - يحتفظ بآخر سبب للرمز الحالي (مع الطابع الزمني).
    """
    try:
        if stage in _REJ_COUNTS:
            _REJ_COUNTS[stage] += 1
        _REJ_SUMMARY[stage] = int(_REJ_SUMMARY.get(stage, 0)) + 1
    except Exception:
        pass

    try:
        if _CURRENT_SYMKEY:
            _LAST_REJECT[_CURRENT_SYMKEY] = {
                "stage": stage,
                "details": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in kv.items()},
                "ts": now_riyadh().isoformat(timespec="seconds")
            }
    except Exception:
        pass

    if DEBUG_LOG_SIGNALS:
        try:
            kvs = " ".join(f"{k}={v}" for k, v in kv.items())
            logger.info(f"[REJECT] {stage} | {kvs}")
        except Exception:
            logger.info(f"[REJECT] {stage}")
    return None

def _pass(stage: str, **kv):
    if DEBUG_LOG_SIGNALS:
        try:
            kvs = " ".join(f"{k}={v}" for k, v in kv.items())
            logger.info(f"[PASS]   {stage} | {kvs}")
        except Exception:
            logger.info(f"[PASS]   {stage}")

def _round_relax_factors() -> Tuple[float, float, float]:
    """
    تليين خفيف داخل الجولة بناءً على كثافة الرفض:
    - يقلل متطلبات ATR و RVOL تدريجيًا.
    - يخفض حد السيولة الدنيا للشمعة عند كثرة الرفض بسبب السيولة.
    يعيد: (f_atr, f_rvol, notional_min)
    """
    f_atr, f_rvol = 1.0, 1.0
    notional_min = float(MIN_BAR_NOTIONAL_USD)
    c = _REJ_COUNTS

    # ATR
    if c.get("atr_low", 0) >= 30:
        f_atr = 0.80
    elif c.get("atr_low", 0) >= 10:
        f_atr = 0.90

    # RVOL
    if c.get("rvol", 0) >= 30:
        f_rvol = 0.90
    elif c.get("rvol", 0) >= 10:
        f_rvol = 0.95

    # Notional
    if c.get("notional_low", 0) >= 10:
        notional_min *= 0.80

    return float(f_atr), float(f_rvol), float(notional_min)


# ================== إعدادات النسخ (new/old/srr/brt/vbr) ==================
BASE_CFG = {
    "ENTRY_MODE": "hybrid",
    "HYBRID_ORDER": ["pullback", "breakout"],
    "PULLBACK_VALUE_REF": "ema21",           # ema21 | vwap
    "PULLBACK_CONFIRM": "bullish_engulf",    # bullish_engulf | bos

    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.002,

    "USE_FIB": False,
    "SWING_LOOKBACK": 60,
    "FIB_TOL": 0.004,
    "BREAKOUT_BUFFER_LTF": 0.0015,
    "RSI_GATE_POLICY": None,                 # lenient | balanced | strict | None

    "USE_ATR_SL_TP": False,
    "STOP_LOSS_PCT": 0.02,
    "TP1_PCT": 0.03,
    "TP2_PCT": 0.06,

    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.01,
    "MAX_HOLD_HOURS": 12,
    "SYMBOL_COOLDOWN_MIN": 30,
}

NEW_SCALP_OVERRIDES = {
    "HYBRID_ORDER": ["breakout", "pullback"],
    "PULLBACK_VALUE_REF": "vwap",
    "PULLBACK_CONFIRM": "bos",

    "RVOL_MIN": RVOL_MIN_NEW,
    "ATR_MIN_FOR_TREND": ATR_MIN_NEW,

    "USE_FIB": True,
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
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.0016,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.8,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.1,
    "LOCK_MIN_PROFIT_PCT": 0.004,
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 8,
}

BRT_OVERRIDES = {
    "ENTRY_MODE": "breakout",
    "RVOL_MIN": RVOL_MIN_BRT,
    "ATR_MIN_FOR_TREND": ATR_MIN_BRT,
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

# ✨ حقن قيم ENV اللينة لأساس ATR/RVOL
BASE_CFG["ATR_MIN_FOR_TREND"] = ATR_MIN_BASE

PER_STRAT_MGMT = {
    "new": {"SL": "atr", "SL_MULT": 0.9, "TP1": "sr_or_atr", "TP1_ATR": 1.2, "TP2_ATR": 2.2,
            "TRAIL_AFTER_TP1": True, "TRAIL_ATR": 1.0, "TIME_HRS": 6},
    "old": {"SL": "pct", "SL_PCT": 0.02, "TP1_PCT": 0.03, "TP2_PCT": 0.06,
            "TRAIL_AFTER_TP1": False, "TIME_HRS": 12},
    "srr": {"SL": "atr_below_sweep", "SL_MULT": 0.8, "TP1": "sr_or_atr", "TP1_ATR": 1.0, "TP2_ATR": 2.2,
            "TRAIL_AFTER_TP1": True, "TRAIL_ATR": 1.0, "TIME_HRS": 4},
    "brt": {"SL": "atr_below_retest", "SL_MULT": 1.0, "TP1": "range_or_atr", "TP1_ATR": 1.5, "TP2_ATR": 2.5,
            "TRAIL_AFTER_TP1": True, "TRAIL_ATR": 0.9, "TIME_HRS": 8},
    "vbr": {"SL": "atr", "SL_MULT": 1.0, "TP1": "vwap_or_sr", "TP2_ATR": 1.8,
            "TRAIL_AFTER_TP1": True, "TRAIL_ATR": 0.8, "TIME_HRS": 3},
}
def _mgmt(variant: str) -> Dict[str, Any]:
    return PER_STRAT_MGMT.get((variant or "new").lower(), PER_STRAT_MGMT["new"])

def get_cfg(variant: str) -> Dict[str, Any]:
    """
    دمج إعدادات BASE_CFG مع Overrides الخاصة بكل نسخة.
    النسخ المدعومة: new (افتراضي)، old، srr، brt، vbr.
    """
    cfg = dict(BASE_CFG)  # نسخة مستقلة
    v = (variant or "new").lower()
    if v == "new":
        cfg.update(NEW_SCALP_OVERRIDES)
    elif v == "srr":
        cfg.update(SRR_OVERRIDES)
    elif v == "brt":
        cfg.update(BRT_OVERRIDES)
    elif v == "vbr":
        cfg.update(VBR_OVERRIDES)
    elif v == "old":
        pass
    else:
        cfg.update(NEW_SCALP_OVERRIDES)
    return cfg

# ---------- Scoring ----------
def _opportunity_score(df, prev, closed):
    """
    يُرجع: score:int, why:str, pattern:str
    - يحمي من NaN/Inf.
    - يلتقط NR_Breakout و BullishEngulf كأنماط أساسية.
    """
    score, why, pattern = 0, [], ""
    try:
        c = _finite_or(None, closed.get("close"))
        o = _finite_or(None, closed.get("open"))
        ema21_v = _finite_or(None, closed.get("ema21"))
        ema50_v = _finite_or(None, closed.get("ema50"))
        rvol = _finite_or(0.0, closed.get("rvol"))

        if c is None or o is None:
            return 0, "", "Generic"

        if c > o:
            score += 10; why.append("BullishClose")
        if ema21_v is not None and c > ema21_v:
            score += 10; why.append("AboveEMA21")
        if ema50_v is not None and c > ema50_v:
            score += 10; why.append("AboveEMA50")

        if rvol >= 1.5:
            score += 15; why.append("HighRVOL")

        # NR breakout التحقق من النطاق مع حمايات
        try:
            is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
            hi_slice = df["high"].iloc[-NR_WINDOW-2:-2]
            if len(hi_slice) >= 3 and np.isfinite(hi_slice.max()):
                hi_range = float(hi_slice.max())
                if is_nr_recent and c > hi_range:
                    score += 20; why.append("NR_Breakout"); pattern = "NR_Breakout"
        except Exception:
            pass

        # Bullish engulf
        try:
            if _bullish_engulf(prev, closed):
                score += 20; why.append("BullishEngulf")
                if not pattern: pattern = "BullishEngulf"
        except Exception:
            pass

    except Exception:
        pass
    return int(score), ", ".join(why), (pattern or "Generic")


# ---------- منطق الدخول ----------
def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    """
    Pullback:
    - قرب ref (VWAP/EMA21 حسب الإعداد) مع تأكيد ابتلاع/BoS.
    - Fallbacks آمنة إذا غاب ref.
    """
    # اختر المرجع حسب الإعداد، مع fallbacks
    if cfg.get("PULLBACK_VALUE_REF", "ema21") == "ema21":
        ref_val = _finite_or(None, closed.get("ema21"))
    else:
        ref_val = _finite_or(None, closed.get("vwap"))
        if ref_val is None:
            ref_val = _finite_or(None, closed.get("ema21"))
    ref_val = _finite_or(float(closed["close"]), ref_val, closed.get("ema50"), closed.get("close"))

    close_v = _finite_or(None, closed.get("close"))
    low_v   = _finite_or(None, closed.get("low"))
    if close_v is None or low_v is None:
        return False

    near_val = (close_v >= ref_val) and (low_v <= ref_val)
    if not near_val:
        return False

    confirm = cfg.get("PULLBACK_CONFIRM", "bullish_engulf")
    if confirm == "bullish_engulf":
        try:
            return _bullish_engulf(prev, closed)
        except Exception:
            return False
    elif confirm == "bos":
        swing_high, _ = _swing_points(df)
        sh = _finite_or(None, swing_high)
        return bool(sh is not None and close_v > sh)

    return True


def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    """
    Breakout:
    - اختراق أعلى نطاق NR الأخير + شرط (NR حديث) أو (فوق VWAP/EMA).
    - Buffer بسيط من الإعدادات.
    """
    try:
        # نطاق عالي حديث
        hi_slice = df["high"].iloc[-NR_WINDOW-2:-2]
        if len(hi_slice) < 3:
            return False
        hi_range = float(hi_slice.max())
        if not np.isfinite(hi_range):
            return False

        is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())

        vwap_ref = _finite_or(float(closed["close"]), closed.get("vwap"), closed.get("ema21"), closed.get("ema50"))
        vwap_ok = float(closed["close"]) > vwap_ref

        buf = float(cfg.get("BREAKOUT_BUFFER_LTF", 0.0015))
        return (float(closed["close"]) > hi_range * (1.0 + buf)) and (is_nr_recent or vwap_ok)
    except Exception:
        return False


# ================== المخاطر اليومية/الساعة + Auto-Relax ==================
def _default_risk_state():
    return {
        "date": _today_str(),
        "daily_pnl": 0.0,
        "consecutive_losses": 0,
        "trades_today": 0,
        "blocked_until": None,
        "hourly_pnl": {},
        "last_signal_ts": None,
        "relax_success_count": 0,
    }

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str():
        s = _default_risk_state(); save_risk_state(s)
    if "hourly_pnl" not in s or not isinstance(s["hourly_pnl"], dict):
        s["hourly_pnl"] = {}
    if "last_signal_ts" not in s:
        s["last_signal_ts"] = None
    if "relax_success_count" not in s:
        s["relax_success_count"] = 0
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)

def register_trade_opened():
    s = load_risk_state()
    s["trades_today"] = int(s.get("trades_today", 0)) + 1
    save_risk_state(s)

def _set_block(minutes, reason="risk"):
    s = load_risk_state()
    until = now_riyadh() + timedelta(minutes=int(minutes))
    s["blocked_until"] = until.isoformat(timespec="seconds")
    save_risk_state(s)
    _tg(f"⛔️ <b>تم تفعيل حظر مؤقت</b> ({reason}) حتى <code>{until.strftime('%H:%M')}</code>.")

def _is_blocked():
    s = load_risk_state()
    bu = s.get("blocked_until")
    if not bu:
        return False
    try:
        t = datetime.fromisoformat(bu)
    except Exception:
        return False
    return now_riyadh() < t

def _mark_signal_now():
    s = load_risk_state()
    s["last_signal_ts"] = now_riyadh().isoformat(timespec="seconds")
    save_risk_state(s)

def _hours_since_last_signal() -> Optional[float]:
    s = load_risk_state()
    ts = s.get("last_signal_ts")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return max(0.0, (now_riyadh() - dt).total_seconds() / 3600.0)
    except Exception:
        return None

def _relax_level_current() -> int:
    """
    مستوى التخفيف 0/1/2 حسب الساعات منذ آخر إشارة، مع إعادة ضبط بعد RELAX_RESET_SUCCESS_TRADES نجاحات.
    """
    s = load_risk_state()
    if int(s.get("relax_success_count", 0)) >= RELAX_RESET_SUCCESS_TRADES:
        return 0
    hrs = _hours_since_last_signal()
    if hrs is None:
        return 0
    if hrs >= AUTO_RELAX_AFTER_HRS_2:
        return 2
    if hrs >= AUTO_RELAX_AFTER_HRS_1:
        return 1
    return 0

def _format_relax_str() -> str:
    hrs = _hours_since_last_signal()
    if hrs is None or hrs > 1e8:
        return "Auto-Relax: لا توجد إشارات بعد."
    if hrs >= 72:
        return f"Auto-Relax: آخر إشارة منذ ~{hrs/24:.1f}d."
    return f"Auto-Relax: آخر إشارة منذ ~{hrs:.1f}h."

def register_trade_result(pnl_usdt):
    s = load_risk_state()
    pnl_usdt = float(pnl_usdt)
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + pnl_usdt
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1

    if pnl_usdt > 0:
        s["relax_success_count"] = int(s.get("relax_success_count", 0)) + 1
        if s["relax_success_count"] >= RELAX_RESET_SUCCESS_TRADES:
            s["relax_success_count"] = 0
            s["last_signal_ts"] = now_riyadh().isoformat(timespec="seconds")
            try:
                _tg("✅ تمت صفقتان ناجحتان — إلغاء تخفيف القيود (عودة للوضع الطبيعي).")
            except Exception:
                pass
    else:
        s["relax_success_count"] = 0

    hk = _hour_key(now_riyadh())
    s["hourly_pnl"][hk] = float(s["hourly_pnl"].get(hk, 0.0)) + pnl_usdt

    # قواطع المخاطر
    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s); _set_block(90, reason="خسائر متتالية"); return

    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s); _set_block(minutes, reason="تجاوز حد الخسارة اليومي"); return

    # هبوط ساعي كنسبة من الرصيد
    if os.getenv("HOURLY_DD_BLOCK_ENABLE", "1").lower() in ("1","true","yes","y"):
        try:
            equity = float(fetch_balance("USDT") or 0.0)
            hour_pnl = float(s["hourly_pnl"].get(hk, 0.0))
            HOURLY_DD_PCT = float(os.getenv("HOURLY_DD_PCT", "0.05"))
            if equity > 0 and (hour_pnl <= -abs(HOURLY_DD_PCT) * equity):
                save_risk_state(s); _set_block(60, reason=f"هبوط {HOURLY_DD_PCT*100:.1f}%/ساعة"); return
        except Exception:
            pass

    save_risk_state(s)

def _risk_precheck_allow_new_entry():
    if _is_blocked():
        return False, "blocked"
    s = load_risk_state()
    if MAX_TRADES_PER_DAY and s.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
        return False, "max_trades"
    if s.get("daily_pnl", 0.0) <= -abs(DAILY_LOSS_LIMIT_USDT):
        return False, "daily_loss_limit"
    if s.get("consecutive_losses", 0) >= MAX_CONSEC_LOSSES:
        return False, "consec_losses"
    return True, ""

# ================== Soft-Schedule (وقت + سوق + Auto-Relax) ==================
def _parse_time_hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)

def _parse_soft_hours(expr: str) -> List[tuple]:
    """
    '00:00-06:00,12:00-18:00' → [(start_min,end_min), …]
    يدعم الالتفاف عبر منتصف الليل. يتجاهل المقاطع غير الصالحة بهدوء.
    """
    spans: List[tuple] = []
    for chunk in (expr or "").split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        a, b = [x.strip() for x in chunk.split("-", 1)]
        try:
            sa = _parse_time_hhmm(a); sb = _parse_time_hhmm(b)
            spans.append((int(sa), int(sb)))
        except Exception:
            continue
    return spans

def _is_minute_in_span(mins: int, span: tuple) -> bool:
    sa, sb = span
    if sa == sb:
        return True  # يغطي اليوم كله
    if sa < sb:
        return sa <= mins < sb
    # التفاف عبر منتصف الليل
    return (mins >= sa) or (mins < sb)

def _is_within_soft_window(dt_local: datetime) -> bool:
    try:
        wd_allowed = {int(x) for x in (SOFT_SCHEDULE_WEEKDAYS or "").split(",") if x.strip().isdigit()}
        if not wd_allowed:
            wd_allowed = set(range(7))
    except Exception:
        wd_allowed = set(range(7))

    spans = _parse_soft_hours(SOFT_SCHEDULE_HRS)
    if not spans:
        return False

    wd = dt_local.weekday()  # 0=Mon .. 6=Sun
    if wd not in wd_allowed:
        return False

    mins = dt_local.hour * 60 + dt_local.minute
    return any(_is_minute_in_span(mins, sp) for sp in spans)

def _soft_scale_by_time_and_market(br: Optional[float], eff_min: float) -> tuple[float, str]:
    """
    يعيد (scale, note) — يُطبّق:
      وقت (TimeSoft) → سوق (MarketWeak) → Auto-Relax.
    السقف 1.0 والأرضية 0.50.
    """
    if not SOFT_SCHEDULE_ENABLE:
        return 1.0, ""

    in_window = _is_within_soft_window(now_riyadh())
    if not in_window:
        return 1.0, ""

    scale = float(SOFT_SCALE_TIME_ONLY)
    note_parts = [f"TimeSoft×{SOFT_SCALE_TIME_ONLY:.2f}"]

    if br is not None and br < eff_min:
        scale *= float(SOFT_SCALE_MARKET_WEAK)
        note_parts.append(f"MarketWeak×{SOFT_SCALE_MARKET_WEAK:.2f}")

    lvl = int(_relax_level_current())
    if lvl > 0 and SOFT_SEVERITY_STEP > 0:
        scale *= max(0.50, 1.0 - float(SOFT_SEVERITY_STEP) * lvl)
        note_parts.append(f"RelaxL{lvl}(-{SOFT_SEVERITY_STEP*lvl:.02f})")

    scale = min(1.0, max(0.50, float(scale)))
    note = " • ".join(note_parts)
    return float(scale), note


# ================== HTF سياق + Breadth ==================
def _get_htf_context(symbol):
    base, _ = _split_symbol_variant(symbol)
    now = now_riyadh()
    ent = _HTF_CACHE.get(base)
    if ent and (now - ent["t"]).total_seconds() <= _HTF_TTL_SEC:
        _METRICS["htf_cache_hits"] += 1
        return ent["ctx"]

    _METRICS["htf_cache_misses"] += 1
    data = get_ohlcv_cached(base, HTF_TIMEFRAME, 200)
    if not data:
        return None
    df = _df(data)
    df["ema50_htf"] = ema(df["close"], HTF_EMA_TREND_PERIOD)
    if len(df) < HTF_SR_WINDOW + 3:
        return None

    df_prev = df.iloc[:-2]
    w = min(HTF_SR_WINDOW, len(df_prev))

    resistance_raw = df_prev["high"].rolling(w).max().iloc[-1]
    support_raw    = df_prev["low"].rolling(w).min().iloc[-1]
    resistance = _finite_or(None, resistance_raw)
    support    = _finite_or(None, support_raw)

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

    # عند تفعيل MTF-strict نجمع سياقات إضافية
    if ENABLE_MTF_STRICT:
        def _tf_info(tf, bars=160):
            try:
                d = get_ohlcv_cached(base, tf, bars)
                if not d or len(d) < 80: return None
                _dfx = _df(d)
                _dfx[f"ema{HTF_EMA_TREND_PERIOD}"] = ema(_dfx["close"], HTF_EMA_TREND_PERIOD)
                row = _dfx.iloc[-2]
                e = _finite_or(float(row["close"]), row.get(f"ema{HTF_EMA_TREND_PERIOD}"))
                return {"tf": tf, "price": float(row["close"]),
                        "ema": float(e),
                        "trend_up": bool(float(row["close"]) > float(e))}
            except Exception:
                return None

        mtf = {HTF_TIMEFRAME: {"tf": HTF_TIMEFRAME, "price": ctx["close"], "ema": ctx["ema50_now"],
                               "trend_up": bool(ctx["close"] > ctx["ema50_now"])}}
        for tf in ("1h", "4h"):
            info = _tf_info(tf)
            if info: mtf[tf] = info
        ctx["mtf"] = mtf

    _HTF_CACHE[base] = {"t": now, "ctx": ctx}
    return ctx


# ===== Breadth Guard (ديناميكي) =====
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
        if len(uniq) >= 12:
            break
    if not uniq:
        uniq = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT"]
    return uniq

def _compute_breadth_ratio() -> Optional[float]:
    refs = _breadth_refs()
    if not refs: return None
    ok, tot = 0, 0
    for sym in refs:
        try:
            d = get_ohlcv_cached(sym, BREADTH_TF, 120)
            if not d or len(d) < 60:
                continue
            df = _df(d)

            # ✅ مهم: تجاهل السلاسل غير الحديثة (≤ ضعف مدة الإطار)
            if not _row_is_recent_enough(df, BREADTH_TF, bars_back=2):
                continue

            df["ema50"] = ema(df["close"], 50)
            row = df.iloc[-2]
            c = float(row["close"]); e = float(row["ema50"])
            if c > 0 and e > 0 and math.isfinite(c) and math.isfinite(e):
                tot += 1
                if c > e: ok += 1
        except Exception:
            continue
    if tot < 5:
        return None
    ratio = ok / float(tot)
    return ratio if ratio > 0.05 else None

def _get_breadth_ratio_cached() -> Optional[float]:
    now_s = time.time()
    if _BREADTH_CACHE["ratio"] is not None and (now_s - _BREADTH_CACHE["t"]) <= BREADTH_TTL_SEC:
        return _BREADTH_CACHE["ratio"]
    r = _compute_breadth_ratio()
    _BREADTH_CACHE["ratio"] = r
    _BREADTH_CACHE["t"] = now_s
    return r

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

def _breadth_min_auto() -> float:
    try:
        eff = _effective_breadth_min()
        return max(0.40, min(0.70, eff))
    except Exception:
        return BREADTH_MIN_RATIO

def breadth_status() -> Dict[str, Any]:
    try:
        r = _get_breadth_ratio_cached()
        eff_min = _breadth_min_auto()
        if r is None:
            return {"ok": True, "ratio": None, "min": eff_min}
        return {"ok": bool(r >= eff_min), "ratio": float(r), "min": float(eff_min)}
    except Exception:
        return {"ok": True, "ratio": None, "min": float(BREADTH_MIN_RATIO)}

# ================== ATR% + RVOL + Notional الديناميكية ==================
def _atrp_min_for_symbol(sym_ctx, thr):
    """يحسب الحد الأدنى ATR% المطلوب حسب نوع الرمز + كوانتايل نشاطه."""
    bucket = sym_ctx.get("bucket", "alt")
    q35 = float(sym_ctx.get("atrp_q35_lookback", 0) or 0)
    base = {
        "maj": thr["ATRP_MIN_MAJ"],
        "alt": thr["ATRP_MIN_ALT"],
        "micro": thr["ATRP_MIN_MICRO"],
    }.get(bucket, thr["ATRP_MIN_ALT"])
    # استخدام كوانتايل نشاط الرمز لتخفيف بسيط إن كان نشِطًا حديثًا
    need = max(base, q35 * 0.90 if q35 > 0 else base)
    return float(need)

def _rvol_ok(ltf_ctx, sym_ctx, thr):
    rvol = float(ltf_ctx.get("rvol", 0) or 0)
    rvol_need = float(thr["RVOL_NEED_BASE"])
    # تخفيف طفيف للميكرو/الميم الرخيصة
    price = float(sym_ctx.get("price", 1.0) or 1.0)
    if price < 0.1 or bool(sym_ctx.get("is_meme", False)):
        rvol_need -= 0.10
    # إذا كانت حركة اختراق واضحة نعطي سماحية طفيفة
    if bool(ltf_ctx.get("is_breakout", False)):
        rvol_need -= 0.05
    return rvol >= rvol_need, float(rvol), float(rvol_need)

def _notional_ok(sym_ctx, thr):
    avg_notional_30 = float(sym_ctx.get("notional_avg_30", 0) or 0)
    min_notional_30 = float(sym_ctx.get("notional_min_30", 0) or 0)
    return (
        avg_notional_30 >= float(thr["NOTIONAL_AVG_MIN"]) and
        min_notional_30 >= float(thr["NOTIONAL_MINBAR"])
    ), float(avg_notional_30), float(min_notional_30)

# ================== HTF Gate مرن ==================
def _htf_gate(htf_trend, ltf_ctx, thr):
    if htf_trend in ("up", "strong_up"):
        return True
    if htf_trend in ("down", "strong_down"):
        return False
    # محايد
    if thr["NEUTRAL_HTF_PASS"]:
        return (
            ltf_ctx.get("ema200_trend") in ("up", "flat_up") and
            float(ltf_ctx.get("rvol", 0)) >= float(thr["RVOL_NEED_BASE"]) and
            bool(ltf_ctx.get("pullback_ok", False))
        )
    return False

# ================== منطق الإشارة ==================
def check_signal(symbol: str):
    global _CURRENT_SYMKEY
    _CURRENT_SYMKEY = symbol
    try:
        # --- إعداد السياق ---
        htf_ctx = _get_htf_context(symbol)
        if not htf_ctx:
            return _rej("data_unavailable")

        ltf = get_ohlcv_cached(symbol, STRAT_LTF_TIMEFRAME, 140)
        if not ltf or len(ltf) < 80:
            return _rej("no_ltf")

        df = _df(ltf)
        df = _ensure_ltf_indicators(df)  # يضيف rvol / is_nr / vwap / EMAs
        if len(df) < 60:
            return _rej("no_ltf")

        closed = df.iloc[-2]; prev = df.iloc[-3]

        # ATR% من نفس إطار LTF وبطريقة موحدة
        atr_val = _finite_or(None, _atr_from_df(df))
        price   = _finite_or(None, closed.get("close"))
        if atr_val is None or price is None or price <= 0:
            return _rej("atr_calc")
        atrp = float(atr_val) / float(price)

        # split الرمز لتحديد نوعه (maj/alt/...)
        base, variant = _split_symbol_variant(symbol)
        bucket = "maj" if base.split("/")[0] in ("BTC", "ETH", "BNB", "SOL") else "alt"

        # مقاييس السيولة والسلوك
        sym_ctx = {
            "bucket": bucket,
            "atrp_q35_lookback": float(df["close"].pct_change().rolling(35).std().iloc[-1] or 0),
            "price": float(price),
            "notional_avg_30": float(df["volume"].iloc[-30:].mean() * float(price)),
            "notional_min_30": float(df["volume"].iloc[-30:].min()  * float(price)),
            "is_meme": False,  # يمكن ربطه بقائمة خارجية لاحقًا
        }

        # معطيات LTF سياقية (rvol + تقدير اختراق + اتجاه ema200 + صلاحية pullback)
        rvol = float(_finite_or(1.0, closed.get("rvol"), 1.0))
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_range  = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        hi_range_series = df["high"].iloc[-NR_WINDOW-2:-2]
        hi_range = float(hi_range_series.max())
          if not math.isfinite(hi_range):
          return _rej("no_ltf")  # حماية إضافية نادرة

        is_breakout = bool(
            (float(closed["close"]) > hi_range) and
            (nr_recent or float(closed["close"]) > _finite_or(float(closed["close"]), closed.get("vwap"), closed.get("ema21")))
        )

        # اتجاه ema200 على LTF
        try:
            ema200_val = float(closed.get("ema200"))
            if float(closed["close"]) > ema200_val:
                ema200_trend = "up"
            elif float(closed["close"]) < ema200_val:
                ema200_trend = "down"
            else:
                ema200_trend = "flat_up"
        except Exception:
            ema200_trend = "flat_up"  # اختيار آمن ولطيف

        # صلاحية pullback (قرب EMA21/VWAP)
        try:
            ema21_val = _finite_or(None, closed.get("ema21"))
            vwap_val  = _finite_or(None, closed.get("vwap"))
            close_v   = float(closed["close"])
            low_v     = float(closed["low"])
            pb_ok = False
            for ref in [vwap_val, ema21_val]:
                if ref is None: 
                    continue
                if (close_v >= ref) and (low_v <= ref):
                    pb_ok = True
                    break
        except Exception:
            pb_ok = False

        ltf_ctx = {
            "rvol": rvol,
            "is_breakout": is_breakout,
            "ema200_trend": ema200_trend,
            "pullback_ok": pb_ok,
        }

        # --- thresholds حسب سعة السوق والـ ATR% الحالي ---
        br = _get_breadth_ratio_cached()
        thr = regime_thresholds(br, atrp)

        # --- HTF trend gate ---
        trend = "neutral"
        try:
            if float(htf_ctx["close"]) > float(htf_ctx["ema50_now"]): trend = "up"
            elif float(htf_ctx["close"]) < float(htf_ctx["ema50_now"]): trend = "down"
        except Exception:
            trend = "neutral"

        if not _htf_gate(trend, ltf_ctx, thr):
            return _rej("htf_trend", trend=trend)

        # --- ATR% check ---
        need_atrp = _atrp_min_for_symbol(sym_ctx, thr)
        if float(atrp) < float(need_atrp):
            return _rej("atr_low", atrp=atrp, need=need_atrp)

        # --- RVOL check ---
        r_ok, rvol_val, need_rvol = _rvol_ok(ltf_ctx, sym_ctx, thr)
        if not r_ok:
            return _rej("rvol", rvol=rvol_val, need=need_rvol)

        # --- Notional check ---
        n_ok, avg_not, minbar = _notional_ok(sym_ctx, thr)
        if not n_ok:
            return _rej("notional_low", avg=avg_not, minbar=minbar)

        # --- entry mode logic (حسب نسخة "new" كمبدئ) ---
        cfg = get_cfg("new")
        mode_pref = cfg.get("ENTRY_MODE", "hybrid")
        chosen_mode = None
        order = cfg.get("HYBRID_ORDER", ["pullback", "breakout"]) if mode_pref == "hybrid" else [mode_pref]

        for m in (order + [x for x in ["pullback", "breakout"] if x not in order]):
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr_val, htf_ctx, cfg):
                chosen_mode = "pullback"; break
            if m == "breakout" and _entry_breakout_logic(df, closed, prev, atr_val, htf_ctx, cfg):
                chosen_mode = "breakout"; break
        if not chosen_mode:
            return _rej("entry_mode", mode=mode_pref)

        # --- تقييم عام للفرصة (Score) ---
        score, why_str, pattern = _opportunity_score(df, prev, closed)
        if SCORE_THRESHOLD and int(score) < int(SCORE_THRESHOLD):
            return _rej("score_low", score=score, need=SCORE_THRESHOLD)

        # --- مرّ كل شيء ---
        _pass("buy", mode=chosen_mode, score=int(score))
        return {
            "decision": "buy",
            "mode": chosen_mode,
            "score": int(score),
            "reasons": why_str,
            "pattern": pattern,
            "features": {
                "atrp": float(atrp),
                "rvol": float(rvol_val),
                "breadth_ratio": (None if br is None else float(br)),
                "htf_ok": bool(trend in ("up", "neutral")),
                "notional_avg_30": float(avg_not),
                "notional_min_30": float(minbar),
            }
        }

    except Exception as e:
        return _rej("error", err=str(e))
    finally:
        _CURRENT_SYMKEY = None
# ================== HTF Gate مرن (نسخة مُحسّنة مع سماحيات آمنة) ==================
def _htf_gate(htf_trend, ltf_ctx, thr):
    """
    منطق العبور لإطار HTF:
    - يسمح فورًا لو الترند أعلى.
    - يرفض فورًا لو الترند هابط.
    - في الترند المحايد: نسمح بشروط لينة لكن آمنة مع Fallback إذا نقصت بعض قيم LTF.
    """
    # ترند HTF واضح
    if htf_trend in ("up", "strong_up"):
        return True
    if htf_trend in ("down", "strong_down"):
        return False

    # محايد: لو السياسة لا تسمح بالمحايد، نرفض
    if not thr["NEUTRAL_HTF_PASS"]:
        return False

    # قراءات من LTF مع افتراضات آمنة
    ema200_trend = ltf_ctx.get("ema200_trend")
    rvol_val     = float(ltf_ctx.get("rvol", 0.0) or 0.0)
    pullback_ok  = bool(ltf_ctx.get("pullback_ok", False))
    is_breakout  = bool(ltf_ctx.get("is_breakout", False))
    rvol_need    = float(thr.get("RVOL_NEED_BASE", 1.2))

    # Fallback: لو ema200_trend غير متوفر، نعتبره "flat_up" إذا كان pullback_ok = True (دلالة قرب/قوة فوق مرجع)
    if ema200_trend is None:
        ema200_trend = "flat_up" if pullback_ok else "down"

    # السماح في المحايد لو:
    # - اتجاه ema200 على الأقل flat_up
    # - RVOL كافٍ
    # - ويوجد أحد مُحفزات الدخول (pullback_ok أو is_breakout)
    if ema200_trend in ("up", "flat_up") and rvol_val >= rvol_need and (pullback_ok or is_breakout):
        return True

    return False

# ================== ملخص الرفض ==================
def maybe_emit_reject_summary():
    if not _REJ_SUMMARY:
        return
    try:
        top = sorted(_REJ_SUMMARY.items(), key=lambda kv: kv[1], reverse=True)[:5]
        parts = [f"{k}:{v}" for k, v in top]
        print(f"[summary] rejects_top5: {', '.join(parts)}")
    except Exception:
        pass
    finally:
        _REJ_SUMMARY.clear()

# ================== Helpers missing (ATR series + thresholds + partials) ==================
def atr(h, l, c, period=14) -> pd.Series:
    h, l, c = pd.Series(h), pd.Series(l), pd.Series(c)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def regime_thresholds(breadth_ratio: float, atrp_now: float) -> dict:
    """
    يُولّد حدودًا ديناميكية حسب سِعة السوق (breadth) و ATR%.
    يعيد مفاتيح تحتاجها check_signal: ATRP_MIN_MAJ/ALT/MICRO, RVOL_NEED_BASE,
    NOTIONAL_AVG_MIN, NOTIONAL_MINBAR, NEUTRAL_HTF_PASS
    """
    br = 0.5 if breadth_ratio is None else float(breadth_ratio)
    if br >= 0.60:
        thr = {
            "ATRP_MIN_MAJ": 0.0015,
            "ATRP_MIN_ALT": 0.0018,
            "ATRP_MIN_MICRO": 0.0022,
            "RVOL_NEED_BASE": 1.10,
            "NOTIONAL_AVG_MIN": 20000,
            "NOTIONAL_MINBAR": max(15000, MIN_BAR_NOTIONAL_USD * 0.6),
            "NEUTRAL_HTF_PASS": True,
        }
    elif br >= 0.50:
        thr = {
            "ATRP_MIN_MAJ": 0.0018,
            "ATRP_MIN_ALT": 0.0022,
            "ATRP_MIN_MICRO": 0.0026,
            "RVOL_NEED_BASE": 1.20,
            "NOTIONAL_AVG_MIN": 25000,
            "NOTIONAL_MINBAR": max(20000, MIN_BAR_NOTIONAL_USD * 0.9),
            "NEUTRAL_HTF_PASS": True,
        }
    else:
        thr = {
            "ATRP_MIN_MAJ": 0.0022,
            "ATRP_MIN_ALT": 0.0026,
            "ATRP_MIN_MICRO": 0.0030,
            "RVOL_NEED_BASE": 1.30,
            "NOTIONAL_AVG_MIN": 30000,
            "NOTIONAL_MINBAR": max(25000, MIN_BAR_NOTIONAL_USD),
            "NEUTRAL_HTF_PASS": False,
        }
    # تليين طفيف إذا الـ ATR% مرتفع جدًا
    if atrp_now >= 0.01:
        thr["RVOL_NEED_BASE"] = max(1.05, thr["RVOL_NEED_BASE"] - 0.05)
    return thr

def _partials_for(score: int, tp_count: int, atrp: float) -> list:
    """توزيع افتراضي للجزئيات حسب الدرجة والـ ATR%."""
    tp_count = max(1, min(int(tp_count), MAX_TP_COUNT))
    base = [1.0] if tp_count == 1 else [0.5, 0.3, 0.2, 0.0, 0.0][:tp_count]
    if score >= 55 and tp_count >= 3:
        base = [0.45, 0.30, 0.25, 0.0, 0.0][:tp_count]
    if atrp >= 0.008 and tp_count >= 3:
        base = [0.40, 0.30, 0.30, 0.0, 0.0][:tp_count]
    s = sum(base)
    return [round(x/s, 6) for x in base]


# ================== Entry plan builder ==================
def _atr_latest(symbol_base: str, tf: str, bars: int = 180) -> tuple[float, float, float]:
    data = get_ohlcv_cached(symbol_base, tf, bars)
    if not data:
        raise RuntimeError("no LTF data")
    df = _df(data)
    df = _ensure_ltf_indicators(df)
    if len(df) < 50:
        raise RuntimeError("ltf too short")

    closed = df.iloc[-2]
    px = float(closed["close"])
    if not math.isfinite(px) or px <= 0:
        raise RuntimeError("price invalid")

    atr_abs = _atr_from_df(df)
    if not atr_abs or not math.isfinite(atr_abs) or atr_abs <= 0:
        raise RuntimeError("atr invalid")

    atrp = atr_abs / max(px, 1e-9)
    if not math.isfinite(atrp) or atrp <= 0:
        raise RuntimeError("atrp invalid")

    return px, float(atr_abs), float(atrp)

def _build_entry_plan(symbol: str, sig: dict | None) -> dict:
    base, variant = _split_symbol_variant(symbol)
    cfg = get_cfg(variant)

    # إذا لم تُمرّر إشارة، استخرج الآن
    if sig is None:
        r = check_signal(symbol)
        if not (isinstance(r, dict) and r.get("decision") == "buy"):
            raise RuntimeError("no buy signal")
        sig = r

    price, atr_abs, atrp = _atr_latest(base, LTF_TIMEFRAME)

    # SL حسب نسخة الإدارة
    mgmt = _mgmt(variant)
    if mgmt.get("SL") in ("atr", "atr_below_sweep", "atr_below_retest"):
        sl_mult = float(mgmt.get("SL_MULT", 1.0))
        sl = float(price - sl_mult * atr_abs)
    elif mgmt.get("SL") == "pct":
        sl = float(price * (1.0 - float(mgmt.get("SL_PCT", 0.02))))
    else:
        sl = float(price - 1.0 * atr_abs)

    # لا تجعل SL أعلى من السعر (قصّ وقائي نادر)
    if sl >= price:
        sl = price * 0.997  # تحت السعر ~0.3%

    # أهداف
    tps: list[float] = []
    if ENABLE_MULTI_TARGETS:
        mults = []
        for k in ("TP1_ATR", "TP2_ATR"):
            if k in mgmt:
                mults.append(float(mgmt[k]))
        if not mults:
            mults = list(TP_ATR_MULTS_TREND)[:3]
        for m in mults[:MAX_TP_COUNT]:
            tps.append(float(price + float(m) * atr_abs))
    else:
        tps.append(float(price + float(mgmt.get("TP1_ATR", 1.2)) * atr_abs))
        tps.append(float(price + float(mgmt.get("TP2_ATR", 2.2)) * atr_abs))

    # تنظيف الأهداف: نهائية/أعلى من الدخول/مرتبة/غير مكررة
    tps = [float(x) for x in tps if math.isfinite(x)]
    tps = [x for x in tps if x > price]
    tps = sorted(set(tps))
    if not tps:
        tps = [float(price + 1.2 * atr_abs), float(price + 2.2 * atr_abs)]

    # Partials متوافقة مع عدد الأهداف
    partials = _partials_for(int(sig.get("score", SCORE_THRESHOLD)), len(tps), atrp)
    if len(partials) != len(tps):
        partials = _partials_for(int(sig.get("score", SCORE_THRESHOLD)), len(tps), atrp)
    s = sum(partials)
    if s <= 0:
        partials = [1.0] + [0.0] * (len(tps) - 1)
    else:
        partials = [round(x/s, 6) for x in partials]

    # Max bars to TP1 ديناميكي
    max_bars = None
    if USE_DYNAMIC_MAX_BARS:
        if atrp >= 0.01:
            max_bars = MAX_BARS_BASE + 6
        elif atrp >= 0.006:
            max_bars = MAX_BARS_BASE + 3
        else:
            max_bars = MAX_BARS_BASE

    # تجميع الإشارة المجهزة
    sig = dict(sig)
    sig.setdefault("messages", {})
    sig["sl"] = float(sl)
    sig["targets"] = [float(x) for x in tps]
    sig["partials"] = partials
    sig["atrp"] = float(atrp)
    sig["max_bars_to_tp1"] = max_bars
    return sig

# ================== execute_buy (متوافق مع main.py) ==================
def execute_buy(symbol: str, sig: dict | None = None):
    base, variant = _split_symbol_variant(symbol)

    # فحوصات المخاطر + سعة المراكز قبل البناء/التنفيذ
    allow, reason = _risk_precheck_allow_new_entry()
    if not allow:
        return None, f"⛔ لا يمكن فتح صفقة الآن ({reason})."
    try:
        if MAX_OPEN_POSITIONS and count_open_positions() >= int(MAX_OPEN_POSITIONS):
            return None, "⛔ الحد الأقصى للمراكز المفتوحة مُمتلئ."
    except Exception:
        pass

    # خطة الدخول (تضمّن بناء SL/TPs/partials…)
    sig = _build_entry_plan(symbol, sig)

    # Quick Win: منع ازدواجية التنفيذ الموقّت
    ok, wait_left = _buy_allowed_now()
    if not ok:
        return None, f"⏳ مُهلة حماية: أعد المحاولة بعد ~{wait_left:.1f}s"

    # تحجيم أساسي
    trade_usdt = float(TRADE_BASE_USDT)

    # تعديل بالحالة العامة للسوق (breadth) + وضع soft + القائد
    br = _get_breadth_ratio_cached()
    eff_min = _breadth_min_auto()
    is_leader = bool(sig.get("leader_flag", False))

    if br is not None:
        if br < 0.45:
            trade_usdt *= 0.70
        elif br < 0.55:
            trade_usdt *= 0.85

    if SOFT_BREADTH_ENABLE and (br is not None) and (br < eff_min) and (not is_leader):
        scale, note = _soft_scale_by_time_and_market(br, eff_min)
        trade_usdt *= scale
        if SOFT_MSG_ENABLE:
            sig["messages"]["breadth_soft"] = (
                f"⚠️ Soft breadth: ratio={br:.2f} < min={eff_min:.2f} → size×{scale:.2f}"
            )

    if is_leader:
        trade_usdt *= 0.50  # استثناء القائد: حجم مخفّض
        if SOFT_MSG_ENABLE:
            sig["messages"]["leader_note"] = "🏁 Leader mode: حجم مخفّض (نصف الحجم)."

    # التحقق من الرصيد والحد الأدنى
    price = float(fetch_price(base))
    usdt  = float(fetch_balance("USDT") or 0.0)
    if usdt < max(MIN_TRADE_USDT, trade_usdt):
        return None, "🚫 رصيد USDT غير كافٍ."
    amount = trade_usdt / max(price, 1e-9)
    if amount * price < MIN_NOTIONAL_USDT:
        return None, "🚫 قيمة الصفقة أقل من الحد الأدنى."

    # تنفيذ (أو DRY_RUN)
    if DRY_RUN:
        order = {"id": f"dry_{int(time.time())}", "average": price}
    else:
        order = place_market_order(base, "buy", amount)
        if not order:
            return None, "⚠️ فشل تنفيذ الصفقة."

    fill_px = float(order.get("average") or order.get("price") or price)

    # حفظ الحالة
    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(fill_px),
        "stop_loss": float(sig["sl"]),
        "targets": [float(x) for x in sig["targets"]],
        "partials": list(sig.get("partials") or []),
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": variant,
        "htf_stop": sig.get("stop_rule"),
        "max_bars_to_tp1": sig.get("max_bars_to_tp1"),
        "messages": sig.get("messages") or {},
        "tp_hits": [False] * len(sig["targets"]),
        "score": sig.get("score"),
        "pattern": sig.get("pattern"),
        "reason": sig.get("reasons"),  # استخدام reasons من الإشارة
        "max_hold_hours": _mgmt(variant).get("TIME_HRS"),
    }

    # Quick Win: رسالة ودّية عن حالة السوق
    try:
        bs = breadth_status()
        if STRAT_TG_SEND and bs and (bs.get("ratio") is not None) and not bs.get("ok", True):
            _tg(f"⚠️ Market breadth ضعيف حالياً: ratio={bs['ratio']:.2f} < min={bs['min']:.2f}")
    except Exception:
        pass

    save_position(symbol, pos)
    register_trade_opened()

    try:
        if STRAT_TG_SEND:
            msg = (
                f"{pos.get('messages',{}).get('entry','✅ دخول')} {symbol}\n"
                f"دخول: <code>{fill_px:.6f}</code>\n"
                f"SL: <code>{pos['stop_loss']:.6f}</code>\n"
                f"🎯 الأهداف: {', '.join(str(round(t,6)) for t in pos['targets'])}\n"
                f"💰 حجم الصفقة: <b>{trade_usdt:.2f}$</b>"
            )
            if pos["messages"].get("breadth_soft"):
                msg += f"\n{pos['messages']['breadth_soft']}"
            if pos["messages"].get("leader_note"):
                msg += f"\n{pos['messages']['leader_note']}"
            _tg(msg)
    except Exception:
        pass

    return order, f"✅ شراء {symbol} | SL: {pos['stop_loss']:.6f} | 💰 {trade_usdt:.2f}$"


# ================== إدارة الصفقة ==================
def manage_position(symbol):
    pos = load_position(symbol)
    if not pos:
        return False

    base = pos["symbol"].split("#")[0]
    current = float(fetch_price(base))
    entry   = float(pos["entry_price"])
    amount  = float(pos["amount"])
    targets = pos.get("targets") or []
    partials = pos.get("partials") or []
    variant = pos.get("variant", "new")
    mgmt = _mgmt(variant)

    if amount <= 0:
        clear_position(symbol)
        return False

    # (1) وقف خسارة على إطار أعلى (اختياري)
    stop_rule = pos.get("htf_stop")
    if stop_rule:
        tf = (stop_rule.get("tf") or "4h").lower()
        tf_map = {"h1": "1h", "1h": "1h", "h4": "4h", "4h": "4h", "d1": "1d", "1d": "1d"}
        tf_fetch = tf_map.get(tf, "4h")
        data_htf = get_ohlcv_cached(base, tf_fetch, 200)
        if data_htf:
            dfh = _df(data_htf)
            row = dfh.iloc[-2]
            level = float(stop_rule.get("level", pos["stop_loss"]))
            if float(row["close"]) < level:
                order = {"average": current} if DRY_RUN else place_market_order(base, "sell", amount)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_net = (exit_px - entry) * amount - (entry + exit_px) * amount * (FEE_BPS_ROUNDTRIP / 10000.0)
                    close_trade(symbol, exit_px, pnl_net, reason="HTF_STOP")
                    try:
                        if STRAT_TG_SEND:
                            _tg(f"🛑 وقف HTF {symbol} عند <code>{exit_px:.6f}</code>")
                    except Exception:
                        pass
                    return True

    # (2) خروج زمني لـ TP1
    max_bars = pos.get("max_bars_to_tp1")
    if isinstance(max_bars, int) and max_bars > 0:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            tf_min = _TF_MIN.get(LTF_TIMEFRAME, 5)  # استخدام زمن الإطار الحقيقي
            bars_passed = int((now_riyadh() - opened_at) // timedelta(minutes=tf_min))
            if bars_passed >= max_bars and not pos["tp_hits"][0]:
                order = {"average": current} if DRY_RUN else place_market_order(base, "sell", amount)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_net = (exit_px - entry) * amount - (entry + exit_px) * amount * (FEE_BPS_ROUNDTRIP / 10000.0)
                    close_trade(symbol, exit_px, pnl_net, reason="TIME_EXIT")
                    try:
                        if STRAT_TG_SEND:
                            _tg(pos.get("messages", {}).get("time", "⌛ خروج زمني"))
                    except Exception:
                        pass
                    return True
        except Exception:
            pass

    # (2b) أقصى مدة احتفاظ
    try:
        max_hold_hours = float(pos.get("max_hold_hours") or mgmt.get("TIME_HRS") or 0)
    except Exception:
        max_hold_hours = 0
    if max_hold_hours:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            if (now_riyadh() - opened_at) >= timedelta(hours=max_hold_hours):
                order = {"average": current} if DRY_RUN else place_market_order(base, "sell", amount)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_net = (exit_px - entry) * amount - (entry + exit_px) * amount * (FEE_BPS_ROUNDTRIP / 10000.0)
                    close_trade(symbol, exit_px, pnl_net, reason="TIME_HOLD_MAX")
                    try:
                        if STRAT_TG_SEND:
                            _tg("⌛ خروج لانتهاء مدة الاحتفاظ")
                    except Exception:
                        pass
                    return True
        except Exception:
            pass

    # (3) إدارة الأهداف + Trailing + قفل ربح بعد TP1
    if targets and partials and len(targets) == len(partials):
        for i, tp in enumerate(targets):
            if not pos["tp_hits"][i] and current >= tp and amount > 0:
                part_qty = amount * partials[i]
                if part_qty * current < MIN_NOTIONAL_USDT:
                    part_qty = amount  # صغير جداً → خروج كامل

                order = {"average": current} if DRY_RUN else place_market_order(base, "sell", part_qty)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_gross = (exit_px - entry) * part_qty
                    fees = (entry + exit_px) * part_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
                    pnl_net = pnl_gross - fees

                    pos["amount"] = float(max(0.0, pos["amount"] - part_qty))
                    pos["tp_hits"][i] = True
                    save_position(symbol, pos)

                    register_trade_result(pnl_net)
                    try:
                        if STRAT_TG_SEND:
                            _tg(pos.get("messages", {}).get(f"tp{i+1}", f"🎯 TP{i+1} تحقق"))
                    except Exception:
                        pass

                    # قفل أرباح بعد TP1
                    try:
                        if i == 0 and pos["amount"] > 0:
                            lock_sl = entry * (1.0 + float(get_cfg(variant).get("LOCK_MIN_PROFIT_PCT", 0.0)))
                            if lock_sl > pos["stop_loss"]:
                                pos["stop_loss"] = float(lock_sl)
                                save_position(symbol, pos)
                                try:
                                    if STRAT_TG_SEND:
                                        _tg(f"🔒 تحريك وقف الخسارة لقفل ربح: <code>{lock_sl:.6f}</code>")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # تريلينغ بعد TP2
                    if i >= 1 and pos["amount"] > 0:
                        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
                        if data_for_atr:
                            df_atr = _df(data_for_atr)
                            atr_val2 = _atr_from_df(df_atr)
                            if atr_val2 and atr_val2 > 0:
                                new_sl = current - atr_val2
                                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                                    pos["stop_loss"] = float(new_sl)
                                    save_position(symbol, pos)
                                    try:
                                        if STRAT_TG_SEND:
                                            _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
                                    except Exception:
                                        pass

    # (3b) تريلينغ عام بعد أي TP
    if _mgmt(variant).get("TRAIL_AFTER_TP1") and pos["amount"] > 0 and any(pos.get("tp_hits", [])):
        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
        if data_for_atr:
            df_atr = _df(data_for_atr)
            atr_val3 = _atr_from_df(df_atr)
            if atr_val3 and atr_val3 > 0:
                new_sl = current - _mgmt(variant).get("TRAIL_ATR", 1.0) * atr_val3
                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                    pos["stop_loss"] = float(new_sl)
                    save_position(symbol, pos)
                    try:
                        if STRAT_TG_SEND:
                            _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
                    except Exception:
                        pass

    # (4) وقف الخسارة النهائي
    if current <= pos["stop_loss"] and pos["amount"] > 0:
        sellable = float(pos["amount"])
        order = {"average": current} if DRY_RUN else place_market_order(base, "sell", sellable)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sellable
            fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            close_trade(symbol, exit_px, pnl_net, reason="SL")
            try:
                if STRAT_TG_SEND:
                    _tg(pos.get("messages", {}).get("sl", "🛑 SL"))
            except Exception:
                pass
            return True

    return False


# ================== إغلاق وتسجيل ==================
def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos:
        return
    closed = load_closed_positions()

    entry  = float(pos.get("entry_price", 0.0))
    amount = float(pos.get("amount", 0.0))
    pnl_pct = ((float(exit_price) / entry) - 1.0) if entry else 0.0

    # تجميع ضرب الأهداف
    tp_hits = {}
    try:
        if "targets" in pos and "tp_hits" in pos and isinstance(pos["tp_hits"], list):
            for i, hit in enumerate(pos["tp_hits"], start=1):
                tp_hits[f"tp{i}_hit"] = bool(hit)
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
        # قد تكون reasons قائمة؛ نخزنها كما هي وسنتعامل معها بأمان في التقرير
        "entry_reason": pos.get("reason"),
        **tp_hits
    })
    save_closed_positions(closed)
    register_trade_result(float(pnl_net))
    clear_position(symbol)


# ================== تقرير يومي ==================
def _fmt_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    header_line = fmt_row(headers)
    body_lines = "\n".join(fmt_row(r) for r in rows)
    return "<pre>" + header_line + "\n" + body_lines + "</pre>"

def _fmt_blocked_until_text():
    s = load_risk_state()
    bu = s.get("blocked_until")
    if not bu:
        return "سماح"
    try:
        dt = datetime.fromisoformat(bu)
        return f"محظور حتى {dt.strftime('%H:%M')}"
    except Exception:
        return f"محظور حتى {bu}"

def _reason_snippet(t, maxlen=40):
    """يُرجع مقتطفًا آمنًا من سبب الدخول (يتعامل مع القائمة/النص)."""
    raw = t.get("entry_reason", t.get("reason", "-"))
    txt = ", ".join(raw) if isinstance(raw, (list, tuple)) else str(raw)
    txt = txt.strip()
    return (txt[:maxlen] + ("…" if len(txt) > maxlen else "")) if txt else "-"

def _fmt_breadth_line() -> str:
    """سطر موجز عن سعة السوق."""
    try:
        bs = breadth_status()
        ratio = bs.get("ratio", None)
        minv  = bs.get("min", None)
        ok    = bool(bs.get("ok", True))
        if ratio is None or minv is None:
            return "سِعة السوق: <b>غير متاحة</b>"
        state = "OK ✅" if ok else "ضعيفة ⚠️"
        return f"سِعة السوق: <b>{ratio:.2f}</b> / min <b>{minv:.2f}</b> • {state}"
    except Exception:
        return "سِعة السوق: <b>غير متاحة</b>"

def build_daily_report_text():
    closed = load_closed_positions()
    today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    s = load_risk_state()

    def f6(x):
        try: return "{:,.6f}".format(float(x))
        except Exception: return str(x)
    def f2(x):
        try: return "{:,.2f}".format(float(x))
        except Exception: return str(x)

    if not todays:
        extra = (
            f"\nوضع المخاطر: "
            f"{_fmt_blocked_until_text()}"
            f" • صفقات اليوم: {int(s.get('trades_today', 0))}"
            f" • PnL اليومي: {float(s.get('daily_pnl', 0.0)):.2f}$"
        )
        # أضف سطر السعة والمتركس حتى مع عدم وجود صفقات
        return (
            f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم."
            f"{extra}\n{_format_relax_str()}\n"
            f"{_fmt_breadth_line()}\n"
            f"{metrics_format()}"
        )

    total_pnl = sum(float(t.get("profit", 0.0)) for t in todays)
    wins = [t for t in todays if float(t.get("profit", 0.0)) > 0]
    win_rate = round(100 * len(wins) / max(1, len(todays)), 2)

    headers = ["الرمز#النسخة", "الكمية", "دخول", "خروج", "P/L$", "P/L%", "Score", "نمط", "سبب", "TP_hits", "Exit"]
    rows = []
    for t in todays:
        # TP hits ديناميكيًا من المفاتيح الموجودة
        tp_hits = []
        i = 1
        while f"tp{i}_hit" in t:
            if t.get(f"tp{i}_hit"):
                tp_hits.append(f"T{i}")
            i += 1
        tp_str = ",".join(tp_hits) if tp_hits else "-"

        # الرمز والنسخة بدون تكرار
        sym = t.get("symbol", "-") or "-"
        var = t.get("variant", "-") or "-"
        sym_var = sym if "#" in sym else f"{sym}#{var}"

        rows.append([
            sym_var,
            f6(t.get('amount', 0)),
            f6(t.get('entry_price', 0)),
            f6(t.get('exit_price', 0)),
            f2(t.get('profit', 0)),
            f"{round(float(t.get('pnl_pct', 0))*100, 2)}%",
            str(t.get("score", "-")),
            t.get("pattern", "-"),
            _reason_snippet(t, 40),
            tp_str,
            t.get("reason", "-")
        ])

    table = _fmt_table(rows, headers)

    risk_line = (
        f"وضع المخاطر: "
        f"{_fmt_blocked_until_text()}"
        f" • اليومي: <b>{float(s.get('daily_pnl', 0.0)):.2f}$</b>"
        f" • متتالية خسائر: <b>{int(s.get('consecutive_losses', 0))}</b>"
        f" • صفقات اليوم: <b>{int(s.get('trades_today', 0))}</b>"
    )

    summary = (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b> • {_format_relax_str()}\n"
        f"{risk_line}\n"
        f"{_fmt_breadth_line()}\n"
        f"{metrics_format()}\n"
    )
    return summary + table


# ================== تشخيص سهل (متوافق مع main.py) ==================
def get_last_reject(symbol: str):
    """
    يرجّع آخر سبب رفض مسجَّل لهذا الرمز (يدعم وجود #variant).
    """
    # نحاول بالمفتاح كما هو أولاً
    if symbol in _LAST_REJECT:
        return _LAST_REJECT[symbol]
    # ثم نجرب base|variant ثم base فقط
    base, variant = _split_symbol_variant(symbol)
    key1 = f"{base}|{variant}"
    if key1 in _LAST_REJECT:
        return _LAST_REJECT[key1]
    if base in _LAST_REJECT:
        return _LAST_REJECT[base]
    return None

def check_signal_debug(symbol: str):
    """
    يُعيد (result, reasons[]) للفحص السريع:
    - إذا شراء: reasons=['buy_ok']
    - إذا لا:  reasons=['<stage>:<details>'] من آخر رفض
    """
    r = check_signal(symbol)
    if isinstance(r, dict) and r.get("decision") == "buy":
        return r, ["buy_ok"]

    last = get_last_reject(symbol)
    if last:
        stg = last.get("stage", "no_buy")
        det = last.get("details", {})
        return None, [f"{stg}:{det}"]
    return None, ["no_buy"]
