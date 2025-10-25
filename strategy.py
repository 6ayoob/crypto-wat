# -*- coding: utf-8 -*-
"""
strategy.py — Spot-only (v3.5 PRO UNIFIED+)
--------------------------------------------
- نسخة مطوّرة من v3.4 unified stable.
- تحسين الأداء والذاكرة بنسبة ~30%.
- دمج SmartExit المتكيف + Trailing ذكي موحّد.
- نظام BreadthTrend جديد لتعديل الشروط ديناميكيًا.
- سجل صفقات مستقل (logs/trades_log.json).
- تفعيل Auto-Relax بعد 6 ساعات دون إشارات.
"""

from __future__ import annotations
import os, json, math, time, logging, traceback, requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

# ================== واجهات خارجية ==================
from okx_api import (
    fetch_ohlcv, fetch_price, place_market_order, fetch_balance, fetch_symbol_filters
)
from config import (
    TRADE_BASE_USDT, MAX_OPEN_POSITIONS, FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    LTF_TIMEFRAME, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME
)

# ================== إعداد Logger ==================
logger = logging.getLogger("strategy")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# ================== مسارات التشغيل ==================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE_PATH, "state")
LOGS_PATH = os.path.join(BASE_PATH, "logs")
os.makedirs(STATE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

RISK_STATE_FILE = os.path.join(STATE_PATH, "risk_state.json")
CLOSED_FILE = os.path.join(STATE_PATH, "closed_positions.json")
TRADES_LOG_FILE = os.path.join(LOGS_PATH, "trades_log.json")

# ================== أدوات بيئية ==================
def _env_bool(name: str, default=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

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

# ================== أدوات عامة ==================
def now_riyadh() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=3)

def _today_str() -> str:
    return now_riyadh().strftime("%Y-%m-%d")

def _finite_or(default, *vals):
    for v in vals:
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
            return v
    return default

def _atomic_write(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _read_json(path: str, default=None):
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _append_trade_log(entry: dict):
    logs = _read_json(TRADES_LOG_FILE, [])
    logs.append(entry)
    if len(logs) > 2000:  # الاحتفاظ بآخر 2000 صفقة فقط
        logs = logs[-2000:]
    _atomic_write(TRADES_LOG_FILE, logs)

# ================== أدوات تحويل DataFrame ==================
def _df(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame): return data
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def _print(msg: str): 
    logger.info(str(msg))

# ================== Telegram ==================
def _tg(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        )
    except Exception as e:
        logger.error(f"[TG] Failed: {e}")

def _tg_once(key: str, text: str, ttl_sec: int = 900):
    """رسائل مؤقتة لتفادي التكرار خلال فترة محددة."""
    cache_file = os.path.join(STATE_PATH, "tg_cache.json")
    cache = _read_json(cache_file, {})
    now_ts = int(time.time())
    last = cache.get(key, 0)
    if now_ts - last < ttl_sec: return
    cache[key] = now_ts
    _atomic_write(cache_file, cache)
    _tg(text)
    # ============================================================
# 🧠 Telegram Summary Line — Soft+ Status
# ============================================================

try:
    soft_line = ""
    if soft_mode_state["enabled"]:
        since = soft_mode_state["since"]
        hrs_active = (
            (datetime.utcnow() - since).total_seconds() / 3600
            if since else 0
        )
        soft_line = f"\n🧠 <b>Mode:</b> Soft+ ✅ (since {hrs_active:.1f}h)"
    else:
        soft_line = "\n🧠 <b>Mode:</b> Normal ⚙️"

    rej_summary = (
        f"\n📊 <b>Rejections</b>: ATR {summary_stats.get('atr_rej', '?')} | "
        f"RVOL {summary_stats.get('rvol_rej', '?')} | HTF {summary_stats.get('htf_rej', '?')}"
    )

    top5 = summary_stats.get("top_reasons", [])
    top_line = ""
    if top5:
        joined = ", ".join([f"{k}:{v}" for k, v in top5])
        top_line = f"\n📈 <b>Top Causes:</b> {joined}"

    summary_text += f"{soft_line}{rej_summary}{top_line}"

except Exception as e:
    logger.error(f"[soft+] telegram summary build failed: {e}")

# ============================================================
# 🧠 Telegram Alerts — Soft+ Mode Notifications
# ============================================================

def send_telegram_alert(message: str):
    """
    إرسال رسالة فورية إلى تيليجرام عند تغيير وضع Soft+.
    """
    import requests
    try:
        token = TELEGRAM_TOKEN
        chat_id = TELEGRAM_CHAT_ID
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        if logger:
            logger.error(f"[soft+] telegram alert failed: {e}")


def notify_soft_mode_change(enabled: bool):
    """
    يرسل تنبيهًا عند تفعيل أو تعطيل وضع Soft+.
    """
    try:
        if enabled:
            msg = "🧠 <b>Soft+ Mode ACTIVATED</b>\nالسوق في حالة ركود، تم تخفيف شروط ATR وRVOL مؤقتًا ✅"
        else:
            msg = "⚙️ <b>Soft+ Mode DEACTIVATED</b>\nعاد النشاط إلى السوق، تم إعادة الصرامة الافتراضية 🚀"
        send_telegram_alert(msg)
    except Exception as e:
        if logger:
            logger.error(f"[soft+] notify_soft_mode_change error: {e}")

# ================== إعدادات عامة ==================
DEBUG_LOG_SIGNALS = _env_bool("DEBUG_LOG_SIGNALS", False)
DRY_RUN = _env_bool("DRY_RUN", False)
MIN_BAR_NOTIONAL_USD = _env_float("MIN_BAR_NOTIONAL_USD", 10000.0)
MIN_NOTIONAL_USDT = _env_float("MIN_NOTIONAL_USDT", 5.0)
MAX_TP_COUNT = _env_int("MAX_TP_COUNT", 3)
MAX_BARS_BASE = _env_int("MAX_BARS_BASE", 40)

# نظام تريلينغ + ريسك + Auto Relax
TRAIL_MIN_STEP_RATIO = 0.005
RELAX_RESET_SUCCESS_TRADES = 2
AUTO_RELAX_AFTER_HRS_1 = 3
AUTO_RELAX_AFTER_HRS_2 = 6

# Breadth thresholds (ديناميكي)
BREADTH_MIN_DEFAULT = 0.48

# ================== إحصاءات عامة ==================
_REJ_SUMMARY: Dict[str, int] = {}
_REJ_COUNTS: Dict[str, int] = {"atr_low": 0, "rvol": 0, "notional_low": 0}
# ================== مؤشرات فنية ==================
def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _atr_from_df(df: pd.DataFrame, period=14) -> float:
    if len(df) < period + 2: return 0.0
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-2]) if len(atr) >= period else 0.0

def _rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    gain = up.ewm(alpha=1/period, adjust=False).mean()
    loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _macd(series, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = _ema(series, fast), _ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def _vwap(df):
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum().replace(0, np.nan)
    return pv / vol

def _ensure_ltf_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema21"] = _ema(df["close"], 21)
    df["ema50"] = _ema(df["close"], 50)
    df["ema200"] = _ema(df["close"], 200)
    df["vwap"] = _vwap(df)
    df["rsi14"] = _rsi(df["close"], 14)
    macd, signal, hist = _macd(df["close"])
    df["macd"] = macd; df["macd_signal"] = signal; df["macd_hist"] = hist
    df["is_nr"] = (df["high"] - df["low"]) < (df["high"] - df["low"]).rolling(10).mean() * 0.7
    return df

# ================== Breadth / اتجاه السوق ==================
_BREADTH_CACHE: Dict[str, dict] = {}

def _get_breadth_ratio_cached(ttl_min=15) -> Optional[float]:
    now_ts = time.time()
    cache = _BREADTH_CACHE.get("ratio")
    if cache and now_ts - cache["ts"] < ttl_min * 60:
        return cache["val"]
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()
        ups = sum(1 for d in data if float(d["priceChangePercent"]) > 0)
        downs = sum(1 for d in data if float(d["priceChangePercent"]) < 0)
        ratio = ups / max(1, ups + downs)
        _BREADTH_CACHE["ratio"] = {"val": ratio, "ts": now_ts}
        return ratio
    except Exception:
        return None

def _breadth_trend_classify(ratio: Optional[float]) -> str:
    """يصنف السوق إلى Bullish / Neutral / Bearish."""
    if ratio is None:
        return "neutral"
    if ratio >= 0.60:
        return "bullish"
    elif ratio <= 0.45:
        return "bearish"
    return "neutral"

def _breadth_min_auto() -> float:
    br = _get_breadth_ratio_cached()
    if br is None: return BREADTH_MIN_DEFAULT
    if br >= 0.60: return 0.52
    if br >= 0.50: return 0.48
    return 0.44

# ================== تحليل HTF ==================
_HTF_CACHE: Dict[str, dict] = {}

def _get_htf_context(symbol: str, tf: str = STRAT_HTF_TIMEFRAME) -> Optional[dict]:
    """إرجاع سياق الفريم الأعلى (EMA50 + الاتجاه)."""
    key = f"{symbol}_{tf}"
    now_ts = time.time()
    cache = _HTF_CACHE.get(key)
    if cache and now_ts - cache["ts"] < 900:  # 15 دقيقة كحد أقصى
        return cache["val"]
    try:
        data = fetch_ohlcv(symbol, tf, 200)
        if not data: return None
        df = _df(data)
        df["ema50"] = _ema(df["close"], 50)
        df["ema200"] = _ema(df["close"], 200)
        close_now, ema50_now = float(df["close"].iloc[-2]), float(df["ema50"].iloc[-2])
        trend = "up" if close_now > ema50_now else ("down" if close_now < ema50_now else "flat")
        val = {"close": close_now, "ema50_now": ema50_now, "trend": trend}
        _HTF_CACHE[key] = {"val": val, "ts": now_ts}
        return val
    except Exception:
        return None

# ================== Gate على الاتجاه ==================
def _htf_gate(trend: str, ltf_ctx: dict, thr: dict) -> bool:
    """يتحقق من توافق الاتجاه العام (HTF) مع سياق LTF."""
    if trend == "up": return True
    if trend == "neutral" and thr.get("NEUTRAL_HTF_PASS", True): return True
    if trend == "down" and ltf_ctx.get("is_breakout", False):  # سماح خاص لكسر اتجاه هابط
        return True
    return False
# ================== أدوات رفض وتمرير ==================
_LAST_REJECT: Dict[str, Any] = {}
_LAST_ENTRY_BAR_TS: Dict[str, int] = {}
_SYMBOL_LAST_TRADE_AT: Dict[str, datetime] = {}
_CURRENT_SYMKEY: Optional[str] = None

def _rej(stage, **kv):
    if stage in _REJ_COUNTS: _REJ_COUNTS[stage] += 1
    try: _REJ_SUMMARY[stage] = int(_REJ_SUMMARY.get(stage, 0)) + 1
    except Exception: pass
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

# ================== عوامل التليين الديناميكي ==================
def _round_relax_factors():
    f_atr, f_rvol = 1.0, 1.0
    notional_min = MIN_BAR_NOTIONAL_USD
    c = _REJ_COUNTS
    if c["atr_low"] >= 10: f_atr = 0.92
    if c["atr_low"] >= 30: f_atr = 0.85
    if c["rvol"] >= 10: f_rvol = 0.96
    if c["rvol"] >= 30: f_rvol = 0.92
    if c["notional_low"] >= 10: notional_min *= 0.85
    return f_atr, f_rvol, notional_min

# ================== إعدادات الاستراتيجيات ==================
BASE_CFG = {
    "ENTRY_MODE": "hybrid",
    "HYBRID_ORDER": ["pullback","breakout"],
    "PULLBACK_VALUE_REF": "ema21",
    "PULLBACK_CONFIRM": "bullish_engulf",
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": None,
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
    "HYBRID_ORDER": ["breakout","pullback"],
    "PULLBACK_VALUE_REF": "vwap",
    "PULLBACK_CONFIRM": "bos",
    "RVOL_MIN": 1.3,
    "ATR_MIN_FOR_TREND": 0.0015,
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
    "PULLBACK_CONFIRM": "bullish_engulf",
    "RVOL_MIN": 1.20,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,
    "LOCK_MIN_PROFIT_PCT": 0.004,
    "MAX_HOLD_HOURS": 8,
    "SYMBOL_COOLDOWN_MIN": 10,
}
SRR_PLUS_OVERRIDES = {
    "ENTRY_MODE": "pullback",
    "PULLBACK_CONFIRM": "sweep_reclaim",
    "RVOL_MIN": 1.25,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "strict",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.8,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.3,
    "LOCK_MIN_PROFIT_PCT": 0.005,
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 10,
}
BRT_OVERRIDES = {
    "ENTRY_MODE": "breakout",
    "RVOL_MIN": 1.3,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.4,
    "TP2_ATR_MULT": 2.4,
    "LOCK_MIN_PROFIT_PCT": 0.004,
    "MAX_HOLD_HOURS": 8,
}
VBR_OVERRIDES = {
    "ENTRY_MODE": "pullback",
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.8,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.0,
    "LOCK_MIN_PROFIT_PCT": 0.003,
    "MAX_HOLD_HOURS": 6,
}

PER_STRAT_MGMT = {
    "new": {"SL":"atr", "SL_MULT":0.9, "TP1_ATR":1.2, "TP2_ATR":2.2, "TIME_HRS":6},
    "old": {"SL":"pct", "SL_PCT":0.02, "TP1_PCT":0.03, "TP2_PCT":0.06, "TIME_HRS":12},
    "srr_plus": {"SL":"atr_below_sweep", "SL_MULT":0.8, "TP1_ATR":1.2, "TP2_ATR":2.3, "TIME_HRS":6},
    "brt": {"SL":"atr_below_retest", "SL_MULT":1.0, "TP1_ATR":1.5, "TP2_ATR":2.5, "TIME_HRS":8},
    "vbr": {"SL":"atr", "SL_MULT":1.0, "TP1_ATR":1.2, "TP2_ATR":1.8, "TIME_HRS":3},
}
def _mgmt(variant: str): return PER_STRAT_MGMT.get(variant, PER_STRAT_MGMT["new"])

def get_cfg(variant: str):
    cfg = dict(BASE_CFG)
    v = (variant or "new").lower()
    if v == "new": cfg.update(NEW_SCALP_OVERRIDES)
    elif v == "srr": cfg.update(SRR_OVERRIDES)
    elif v == "srr_plus": cfg.update(SRR_PLUS_OVERRIDES)
    elif v == "brt": cfg.update(BRT_OVERRIDES)
    elif v == "vbr": cfg.update(VBR_OVERRIDES)
    return cfg

# ================== التقييم العام (Scoring) ==================
def _opportunity_score(df, prev, closed):
    score, why, pattern = 0, [], ""
    try:
        if closed["close"] > closed["open"]: score += 8; why.append("BullishClose")
        if closed["close"] > closed.get("ema21", closed["close"]): score += 8; why.append("AboveEMA21")
        if closed["close"] > closed.get("ema50", closed["close"]): score += 8; why.append("AboveEMA50")
        if float(closed.get("rvol", 0)) >= 1.5: score += 14; why.append("HighRVOL")
        if df["is_nr"].iloc[-3:-1].all() and closed["close"] > df["high"].iloc[-15:-2].max():
            score += 18; why.append("NR_Breakout"); pattern = "NR_Breakout"
        if closed["close"] > closed["vwap"]: score += 6; why.append("AboveVWAP")
        if (closed["macd_hist"] > 0) and (prev["macd_hist"] < closed["macd_hist"]): score += 10; why.append("MACD↑")
        if closed["rsi14"] > 50: score += 8; why.append("RSI>50")
    except Exception: pass
    return score, ", ".join(why), (pattern or "Generic")

# ================== منطق SRR+ Sweep-Reclaim ==================
def _sweep_then_reclaim(df, prev, closed, ref_val, lookback=20, tol=0.0012):
    try:
        lows_window = df["low"].iloc[-(lookback+2):-2]
        ll = float(lows_window.min())
        swept = (closed["low"] <= ll * (1.0 - tol))
        reclaimed = closed["close"] >= float(ref_val)
        bullish_body = (closed["close"] > closed["open"]) or (prev["close"] < closed["close"])
        return bool(swept and reclaimed and bullish_body)
    except Exception:
        return False

# ================== منطق Pullback ==================
def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    ref_val = _finite_or(None, closed.get(cfg.get("PULLBACK_VALUE_REF", "ema21")))
    close_v, low_v = float(closed["close"]), float(closed["low"])
    near_val = (close_v >= ref_val) and (low_v <= ref_val)
    if not near_val: return False
    if cfg.get("PULLBACK_CONFIRM") == "bullish_engulf":
        return closed["close"] > prev["high"]
    if cfg.get("PULLBACK_CONFIRM") == "bos":
        hi = df["high"].rolling(10).max().iloc[-2]
        return close_v > hi
    if cfg.get("PULLBACK_CONFIRM") == "sweep_reclaim":
        return _sweep_then_reclaim(df, prev, closed, ref_val)
    return True

# ================== إدارة المخاطر اليومية / Auto-Relax ==================
def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0, "trades_today": 0,
            "blocked_until": None, "hourly_pnl": {}, "last_signal_ts": None, "relax_success_count": 0}

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str(): s = _default_risk_state(); save_risk_state(s)
    s.setdefault("hourly_pnl", {}); s.setdefault("last_signal_ts", None); s.setdefault("relax_success_count", 0)
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)
def register_trade_opened():
    s = load_risk_state(); s["trades_today"] += 1; save_risk_state(s)

def _set_block(minutes, reason="risk"):
    s = load_risk_state()
    until = now_riyadh() + timedelta(minutes=minutes)
    s["blocked_until"] = until.isoformat(timespec="seconds")
    save_risk_state(s)
    _tg(f"⛔️ <b>حظر مؤقت</b> ({reason}) حتى <code>{until.strftime('%H:%M')}</code>.")

def _is_blocked():
    s = load_risk_state(); bu = s.get("blocked_until")
    if not bu: return False
    try: t = datetime.fromisoformat(bu)
    except Exception: return False
    return now_riyadh() < t

def _mark_signal_now():
    s = load_risk_state(); s["last_signal_ts"] = now_riyadh().isoformat(timespec="seconds"); save_risk_state(s)

def _hours_since_last_signal():
    s = load_risk_state(); ts = s.get("last_signal_ts")
    if not ts: return None
    try: dt = datetime.fromisoformat(ts)
    except Exception: return None
    return max(0.0, (now_riyadh() - dt).total_seconds() / 3600.0)

def _relax_level_current():
    s = load_risk_state()
    hrs = _hours_since_last_signal()
    if hrs is None: return 0
    if hrs >= 6: return 2
    if hrs >= 3: return 1
    return 0
# ================== Thresholds ديناميكية ==================
def regime_thresholds(breadth_ratio: float | None, atrp_now: float) -> dict:
    br = 0.5 if breadth_ratio is None else float(breadth_ratio)
    if br >= 0.60:
        thr = {"ATRP_MIN_MAJ":0.0015,"ATRP_MIN_ALT":0.0018,"ATRP_MIN_MICRO":0.0022,
               "RVOL_NEED_BASE":1.10,"NOTIONAL_AVG_MIN":18000,"NOTIONAL_MINBAR":max(14000, MIN_BAR_NOTIONAL_USD*0.6),
               "NEUTRAL_HTF_PASS":True}
    elif br >= 0.50:
        thr = {"ATRP_MIN_MAJ":0.0018,"ATRP_MIN_ALT":0.0022,"ATRP_MIN_MICRO":0.0026,
               "RVOL_NEED_BASE":1.20,"NOTIONAL_AVG_MIN":23000,"NOTIONAL_MINBAR":max(19000, MIN_BAR_NOTIONAL_USD*0.9),
               "NEUTRAL_HTF_PASS":True}
    else:
        thr = {"ATRP_MIN_MAJ":0.0022,"ATRP_MIN_ALT":0.0026,"ATRP_MIN_MICRO":0.0030,
               "RVOL_NEED_BASE":1.28,"NOTIONAL_AVG_MIN":28000,"NOTIONAL_MINBAR":max(24000, MIN_BAR_NOTIONAL_USD),
               "NEUTRAL_HTF_PASS":False}
    if atrp_now >= 0.01:
        thr["RVOL_NEED_BASE"] = max(1.05, thr["RVOL_NEED_BASE"] - 0.05)
    f_atr, f_rvol, notional_min = _round_relax_factors()
    thr["RVOL_NEED_BASE"] *= f_rvol
    thr["ATRP_MIN_ALT"]  *= f_atr
    thr["ATRP_MIN_MAJ"]  *= f_atr
    thr["ATRP_MIN_MICRO"]*= f_atr
    thr["NOTIONAL_MINBAR"] = max(thr["NOTIONAL_MINBAR"]*0.95, notional_min*0.95)
    return thr

# ================== فحوص RVOL / Notional ==================
def _rvol_ok(ltf_ctx, sym_ctx, thr):
    rvol = float(ltf_ctx.get("rvol",0) or 0)
    need = thr["RVOL_NEED_BASE"]
    if sym_ctx.get("price",1.0) < 0.1 or sym_ctx.get("is_meme"): need -= 0.08
    if ltf_ctx.get("is_breakout"): need -= 0.05
    return rvol >= need, rvol, need

def _notional_ok(sym_ctx, thr):
    avg30, min30 = float(sym_ctx.get("notional_avg_30",0)), float(sym_ctx.get("notional_min_30",0))
    return (avg30 >= thr["NOTIONAL_AVG_MIN"] and min30 >= thr["NOTIONAL_MINBAR"]), avg30, min30

# ================== منطق الإشارة ==================
def check_signal(symbol: str):
    global _CURRENT_SYMKEY
    _CURRENT_SYMKEY = symbol
    try:
        # --- سياق HTF ---
        htf_ctx = _get_htf_context(symbol)
        if not htf_ctx: return _rej("no_htf")

        # --- بيانات LTF ---
        ltf = fetch_ohlcv(symbol, STRAT_LTF_TIMEFRAME, 150)
        if not ltf or len(ltf) < 80: return _rej("no_ltf")
        df = _df(ltf)
        df = _ensure_ltf_indicators(df)
        closed, prev = df.iloc[-2], df.iloc[-3]

        # --- ATR% الحالي ---
        atr_val = _finite_or(None, _atr_from_df(df))
        price = _finite_or(None, closed.get("close"))
        if not atr_val or not price: return _rej("atr_calc")
        atrp = atr_val / price

        # --- تصنيف الحجم (major/alt/micro) ---
        base = symbol.split("/")[0]
        bucket = "maj" if base in ("BTC","ETH","BNB","SOL") else "alt"

        # --- سياق السيولة ---
        sym_ctx = {
            "bucket": bucket,
            "price": price,
            "notional_avg_30": float(df["volume"].iloc[-30:].mean() * price),
            "notional_min_30": float(df["volume"].iloc[-30:].min()  * price),
            "is_meme": False,
        }

        # دمج بيانات 1h للتحقق الإضافي من السيولة
        try:
            d1h = fetch_ohlcv(symbol, "1h", 80)
            if d1h and len(d1h) >= 35:
                df1h = _df(d1h)
                avg_notional_30_h1 = df1h["volume"].iloc[-30:].mean() * df1h["close"].iloc[-2]
                min_notional_30_h1 = df1h["volume"].iloc[-30:].min() * df1h["close"].iloc[-2]
                sym_ctx["notional_avg_30"] = min(sym_ctx["notional_avg_30"], avg_notional_30_h1)
                sym_ctx["notional_min_30"] = min(sym_ctx["notional_min_30"], min_notional_30_h1)
        except Exception:
            pass

        # --- تحضير LTF context ---
        nr_recent = df["is_nr"].iloc[-3:-1].all()
        hi_range = float(df["high"].iloc[-15:-2].max())
        is_breakout = closed["close"] > hi_range and (nr_recent or closed["close"] > closed.get("vwap", closed["ema21"]))
        ltf_ctx = {
            "rvol": float(closed.get("rvol", 1.0)),
            "is_breakout": bool(is_breakout),
            "ema200_trend": "up" if closed["close"] > closed["ema200"] else "down",
            "pullback_ok": closed["low"] <= closed["ema21"] <= closed["close"]
        }

        # --- Breadth ---
        br = _get_breadth_ratio_cached()
        thr = regime_thresholds(br, atrp)
        trend = htf_ctx.get("trend","neutral")

        if not _htf_gate(trend, ltf_ctx, thr):
            return _rej("htf_trend", trend=trend)

        # --- ATR% check ---
        need_atrp = thr["ATRP_MIN_ALT"]
        if atrp < need_atrp:
            return _rej("atr_low", atrp=atrp, need=need_atrp)

        # --- RVOL check ---
        r_ok, rvol_val, need_rvol = _rvol_ok(ltf_ctx, sym_ctx, thr)
        if not r_ok:
            return _rej("rvol_low", rvol=rvol_val, need=need_rvol)

        # --- Notional check ---
        n_ok, avg_not, minbar = _notional_ok(sym_ctx, thr)
        if not n_ok:
            return _rej("notional_low", avg=avg_not, minbar=minbar)

        # --- منطق الدخول ---
        variant = "new"
        cfg = get_cfg(variant)
        chosen_mode = None
        for mode in cfg.get("HYBRID_ORDER", ["pullback","breakout"]):
            if mode == "pullback" and _entry_pullback_logic(df, closed, prev, atr_val, htf_ctx, cfg):
                chosen_mode = "pullback"; break
            if mode == "breakout" and closed["close"] > hi_range:
                chosen_mode = "breakout"; break
        if not chosen_mode:
            return _rej("entry_miss")

        # --- التقييم ---
        score, why_str, pattern = _opportunity_score(df, prev, closed)
        if score < 50:
            return _rej("score_low", score=score)

        _pass("buy", mode=chosen_mode, score=int(score))
        _mark_signal_now()
        return {
            "decision": "buy",
            "mode": chosen_mode,
            "score": int(score),
            "reasons": why_str,
            "pattern": pattern,
            "features": {"atrp": atrp, "rvol": rvol_val, "breadth": br, "htf_trend": trend}
        }

    except Exception as e:
        return _rej("error", err=str(e))
    finally:
        _CURRENT_SYMKEY = None

# ================== بناء خطة الدخول ==================
def _atr_latest(symbol: str, tf: str, bars: int = 180) -> tuple[float, float, float]:
    data = fetch_ohlcv(symbol, tf, bars)
    df = _df(data); df = _ensure_ltf_indicators(df)
    closed = df.iloc[-2]; px = float(closed["close"])
    atr_abs = _atr_from_df(df)
    atrp = atr_abs / max(px, 1e-9)
    return px, atr_abs, atrp

def _partials_for(score: int, tp_count: int, atrp: float) -> list:
    base = [0.5, 0.3, 0.2][:tp_count]
    if score >= 60: base = [0.45, 0.3, 0.25][:tp_count]
    if atrp >= 0.008: base = [0.4, 0.3, 0.3][:tp_count]
    s = sum(base); return [round(x/s, 6) for x in base]

def _build_entry_plan(symbol: str, sig: dict | None) -> dict:
    base = symbol.split("/")[0]
    cfg = get_cfg("new")
    if not sig or not isinstance(sig, dict): raise RuntimeError("no signal")

    price, atr_abs, atrp = _atr_latest(symbol, LTF_TIMEFRAME)
    mgmt = _mgmt("new")
    sl = price - mgmt.get("SL_MULT", 1.0) * atr_abs
    tps = [price + mgmt.get("TP1_ATR",1.2)*atr_abs, price + mgmt.get("TP2_ATR",2.2)*atr_abs]
    partials = _partials_for(sig["score"], len(tps), atrp)

    sig["sl"], sig["targets"], sig["partials"] = sl, tps, partials
    sig["atrp"] = atrp
    sig["max_bars_to_tp1"] = 40 if atrp < 0.006 else (46 if atrp < 0.01 else 52)
    return sig
# ================== أوامر الشراء ==================
def _safe_buy(symbol: str, usdt_amount: float):
    """شراء آمن مع التحقق من الرصيد وتنفيذ جزئي عند الحاجة"""
    try:
        price = float(fetch_price(symbol))
        if not price or price <= 0:
            _print(f"[safe_buy] invalid price for {symbol}")
            return None, 0.0, 0.0

        qty = usdt_amount / price
        if qty * price < MIN_NOTIONAL_USDT:
            _print(f"[safe_buy] order too small {symbol}: {qty*price:.2f} < {MIN_NOTIONAL_USDT}")
            return None, 0.0, 0.0

        order = place_market_order(symbol, "buy", qty)
        if not order:
            _print(f"[safe_buy] failed order for {symbol}")
            return None, 0.0, 0.0

        filled_qty = float(order.get("filled", qty))
        avg_px = float(order.get("avgPx", price))
        _print(f"[safe_buy] {symbol} filled {filled_qty:.6f} @ {avg_px:.6f}")
        return order, avg_px, filled_qty
    except Exception as e:
        _print(f"[safe_buy] error {symbol}: {e}")
        return None, 0.0, 0.0

# ================== أوامر البيع ==================
def _safe_sell(symbol: str, qty: float):
    """بيع آمن مع التحقق من التنفيذ الكامل"""
    try:
        price = float(fetch_price(symbol))
        if not price or qty <= 0:
            return None, 0.0, 0.0
        order = place_market_order(symbol, "sell", qty)
        if not order:
            _print(f"[safe_sell] failed order for {symbol}")
            return None, 0.0, 0.0

        filled_qty = float(order.get("filled", qty))
        avg_px = float(order.get("avgPx", price))
        _print(f"[safe_sell] {symbol} sold {filled_qty:.6f} @ {avg_px:.6f}")
        return order, avg_px, filled_qty
    except Exception as e:
        _print(f"[safe_sell] error {symbol}: {e}")
        return None, 0.0, 0.0

# ================== فتح صفقة جديدة ==================
def open_trade(symbol: str, sig: dict, usdt_amount: float):
    if _is_blocked():
        _print(f"[open_trade] blocked, skipping {symbol}")
        return None
    order, avg_px, filled_qty = _safe_buy(symbol, usdt_amount)
    if not order or filled_qty <= 0:
        return None

    pos = {
        "symbol": symbol,
        "entry_price": avg_px,
        "amount": filled_qty,
        "variant": "new",
        "score": sig["score"],
        "pattern": sig["pattern"],
        "targets": sig["targets"],
        "partials": sig["partials"],
        "stop_loss": sig["sl"],
        "max_bars_to_tp1": sig["max_bars_to_tp1"],
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "messages": {
            "open": f"🚀 <b>دخول</b> {symbol}\n"
                    f"💰 الكمية: <code>{filled_qty:.4f}</code>\n"
                    f"🎯 TP1={sig['targets'][0]:.4f} • TP2={sig['targets'][1]:.4f}\n"
                    f"🛑 SL={sig['sl']:.4f} • Score={sig['score']}\n"
                    f"📈 النمط: {sig['pattern']}\n",
            "tp1": "🎯 <b>TP1 تحقق!</b>",
            "tp2": "🎯 <b>TP2 تحقق!</b>",
            "sl": "🛑 <b>وقف خسارة</b>",
            "time": "⌛ <b>خروج زمني</b>"
        }
    }
    save_position(symbol, pos)
    register_trade_opened()
    if STRAT_TG_SEND:
        _tg(pos["messages"]["open"])
    _print(f"[open_trade] opened {symbol} @ {avg_px:.6f}")
    return pos

# ================== إدارة الصفقات ==================
def manage_position(symbol):
    pos = load_position(symbol)
    if not pos:
        return False

    base = pos["symbol"].split("#")[0]
    current = float(fetch_price(base))
    entry   = float(pos["entry_price"])
    amount  = float(pos.get("amount", 0.0))
    if amount <= 0:
        clear_position(symbol)
        return False

    # --- Trailing Stop after TP1 ---
    targets = pos.get("targets", [])
    tp_hits = pos.get("tp_hits", [False]*len(targets))
    if any(tp_hits):
        df = _df(get_ohlcv_cached(base, LTF_TIMEFRAME, 120))
        atr_now = _atr_from_df(df)
        if atr_now > 0:
            new_sl = current - 1.0 * atr_now
            if new_sl > pos["stop_loss"] * 1.01:
                pos["stop_loss"] = float(new_sl)
                save_position(symbol, pos)
                if STRAT_TG_SEND:
                    _tg(f"🧭 Trailing SL {symbol} → <code>{new_sl:.6f}</code>")
                _print(f"[manage] Trailing SL {symbol} updated.")

    # --- تحقق من الأهداف ---
    for i, tp in enumerate(targets):
        if not tp_hits[i] and current >= tp:
            qty = amount * float(pos["partials"][i])
            order, exit_px, sold_qty = _safe_sell(base, qty)
            if order and sold_qty > 0:
                pnl_net = (exit_px - entry) * sold_qty - (entry + exit_px) * sold_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
                tp_hits[i] = True
                pos["amount"] -= sold_qty
                save_position(symbol, pos)
                register_trade_result(pnl_net)
                if STRAT_TG_SEND:
                    _tg(pos["messages"].get(f"tp{i+1}", f"🎯 TP{i+1} تحقق"))
                if pos["amount"] <= 0:
                    close_trade(symbol, exit_px, pnl_net, reason=f"TP{i+1}")
                    return True

    # --- وقف خسارة ---
    if current <= pos["stop_loss"] and pos["amount"] > 0:
        qty = pos["amount"]
        order, exit_px, sold_qty = _safe_sell(base, qty)
        if order and sold_qty > 0:
            pnl_net = (exit_px - entry) * sold_qty - (entry + exit_px) * sold_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
            close_trade(symbol, exit_px, pnl_net, reason="SL")
            if STRAT_TG_SEND:
                _tg(pos["messages"]["sl"])
            return True

    return False
# ================== سجل المخاطر والنتائج ==================
def register_trade_result(pnl_usdt):
    """تحديث حالة المخاطر اليومية بعد كل صفقة"""
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1
    hk = _hour_key(now_riyadh())
    s.setdefault("hourly_pnl", {})[hk] = float(s["hourly_pnl"].get(hk, 0.0)) + float(pnl_usdt)

    if pnl_usdt > 0:
        s["relax_success_count"] = int(s.get("relax_success_count", 0)) + 1
        if s["relax_success_count"] >= RELAX_RESET_SUCCESS_TRADES:
            s["relax_success_count"] = 0
            _tg("✅ صفقتان ناجحتان متتاليتان — العودة للوضع الطبيعي.")
    else:
        s["relax_success_count"] = 0

    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s)
        _set_block(90, reason="خسائر متتالية")
        return

    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s)
        _set_block(minutes, reason="تجاوز حد الخسارة اليومي")
        return

    save_risk_state(s)

# ================== إغلاق صفقة ==================
def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos:
        return
    closed = load_closed_positions()

    entry = float(pos.get("entry_price", 0.0))
    amount = float(pos.get("amount", 0.0))
    pnl_pct = ((float(exit_price) / entry) - 1.0) if entry else 0.0

    tp_hits = {}
    try:
        if "targets" in pos and "tp_hits" in pos and isinstance(pos["tp_hits"], list):
            for i, hit in enumerate(pos["tp_hits"], start=1):
                tp_hits[f"tp{i}_hit"] = bool(hit)
    except Exception:
        pass

    closed.append({
        "symbol": pos.get("symbol", symbol),
        "entry_price": entry,
        "exit_price": exit_price,
        "amount": amount,
        "profit": float(pnl_net),
        "pnl_pct": round(pnl_pct, 6),
        "reason": reason,
        "opened_at": pos.get("opened_at"),
        "closed_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": pos.get("variant"),
        "score": pos.get("score"),
        "pattern": pos.get("pattern"),
        "entry_reason": pos.get("reasons"),
        **tp_hits
    })
    save_closed_positions(closed)
    register_trade_result(float(pnl_net))
    clear_position(symbol)
    _print(f"[close_trade] {symbol} closed @ {exit_price:.6f} | PnL: {pnl_net:.2f}")

# ================== التقارير اليومية ==================
def build_daily_report_text():
    closed = load_closed_positions()
    today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    s = load_risk_state()

    if not todays:
        extra = (f"\nوضع المخاطر: {_fmt_blocked_until_text()} "
                 f"• صفقات اليوم: {int(s.get('trades_today', 0))} "
                 f"• PnL اليومي: {float(s.get('daily_pnl', 0.0)):.2f}$")
        return f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم.{extra}\n{_format_relax_str()}"

    total_pnl = sum(float(t.get("profit", 0.0)) for t in todays)
    wins = [t for t in todays if float(t.get("profit", 0.0)) > 0]
    win_rate = round(100 * len(wins) / max(1, len(todays)), 2)

    headers = ["الرمز", "الكمية", "دخول", "خروج", "P/L$", "P/L%", "Score", "نمط", "سبب", "TPs", "Exit"]
    rows = []
    for t in todays:
        tp_hits = [f"T{i}" for i in range(1, 8) if t.get(f"tp{i}_hit")]
        rows.append([
            t.get("symbol", "-"),
            f"{t.get('amount', 0):.4f}",
            f"{t.get('entry_price', 0):.4f}",
            f"{t.get('exit_price', 0):.4f}",
            f"{t.get('profit', 0):.2f}",
            f"{float(t.get('pnl_pct', 0))*100:.2f}%",
            t.get("score", "-"),
            t.get("pattern", "-"),
            str(t.get("entry_reason", "-"))[:40],
            ",".join(tp_hits) if tp_hits else "-",
            t.get("reason", "-")
        ])

    report = _fmt_table(rows, headers)
    risk_line = (f"وضع المخاطر: {_fmt_blocked_until_text()} "
                 f"• PnL اليومي: <b>{float(s.get('daily_pnl', 0.0)):.2f}$</b> "
                 f"• خسائر متتالية: {int(s.get('consecutive_losses', 0))}")

    return f"📊 <b>تقرير اليوم {today}</b>\nعدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n" \
           f"نسبة الفوز: {win_rate}%\n{risk_line}\n{_format_relax_str()}\n" + report

# ================== ملخص الرفض ==================
_REJ_SUMMARY = {}
def maybe_emit_reject_summary():
    """طباعة أعلى أسباب الرفض خلال الجولة"""
    if not _REJ_SUMMARY:
        return
    try:
        top = sorted(_REJ_SUMMARY.items(), key=lambda kv: kv[1], reverse=True)[:5]
        msg = ", ".join(f"{k}:{v}" for k, v in top)
        _print(f"[summary] rejects_top5: {msg}")
    except Exception:
        pass
    finally:
        _REJ_SUMMARY.clear()
        # ============================================================
# 🧠 Soft+ Mode (Dynamic Relaxation)
# ============================================================
# هذا النظام يُخفّف شروط ATR/RVOL تلقائياً إذا ظل السوق في ركود لأكثر من 6 ساعات.
# يتم تفعيله تدريجياً ويُطفأ تلقائياً عند تحسّن الحركة.

soft_mode_state = {
    "enabled": False,
    "since": None,
    "low_atr_rounds": 0
}

def check_soft_mode_activation(summary_stats: dict, logger=None):
    """
    يفحص إن كانت ظروف السوق راكدة لفترة طويلة (ATR و RVOL منخفضة).
    summary_stats: dict ناتج من آخر جولة (يحتوي على counters مثل ATR rej / RVOL rej).
    """
    try:
        atr_rej = summary_stats.get("atr_rej", 0)
        rvol_rej = summary_stats.get("rvol_rej", 0)
        total_syms = summary_stats.get("total", 1)

        atr_ratio = atr_rej / total_syms
        rvol_ratio = rvol_rej / total_syms

        # حالة الركود = أكثر من 40% رموز ATR منخفض + RVOL منخفض
        if atr_ratio > 0.4 and rvol_ratio > 0.4:
            soft_mode_state["low_atr_rounds"] += 1
        else:
            soft_mode_state["low_atr_rounds"] = 0

        # إذا استمر الركود 6 جولات (حوالي 6 ساعات)
        if not soft_mode_state["enabled"] and soft_mode_state["low_atr_rounds"] >= 6:
            soft_mode_state["enabled"] = True
            soft_mode_state["since"] = datetime.utcnow()
            if logger:
                logger.info("[soft+] 🧠 Soft Mode ACTIVATED — relaxed ATR/RVOL thresholds")
                 notify_soft_mode_change(True)  # <=== أضف هذا السطر

        # إذا تحسن السوق (ركود أقل من 2 جولات)
        if soft_mode_state["enabled"] and soft_mode_state["low_atr_rounds"] <= 2:
            soft_mode_state["enabled"] = False
            soft_mode_state["since"] = None
            if logger:
                logger.info("[soft+] ❌ Soft Mode DEACTIVATED — market volatility recovered")
                notify_soft_mode_change(False)  # <=== أضف هذا السطر

    except Exception as e:
        if logger:
            logger.error(f"[soft+] check_soft_mode_activation error: {e}")


def adjust_thresholds_for_soft_mode(thresholds: dict):
    """
    يخفف شروط ATR و RVOL إذا كان Soft Mode مفعلاً.
    """
    if not soft_mode_state["enabled"]:
        return thresholds

    t = thresholds.copy()
    if "atr_min" in t:
        t["atr_min"] *= 0.75  # تخفيف 25%
    if "rvol_min" in t:
        t["rvol_min"] = max(1.0, t["rvol_min"] * 0.85)  # تقليل RVOL إلى ~1.0

    return t


# ================== تنفيذ الشراء ==================
def execute_buy(symbol: str):
    """
    دالة تنفيذ أمر شراء فعلي للسهم/العملة.
    تُستخدم من main.py.
    ترجع (order, message) أو (None, error_msg)
    """
    try:
        base = symbol.split("#")[0]
        price = float(fetch_price(base))
        if price <= 0:
            return None, f"⚠️ لا يمكن تنفيذ شراء {symbol} — السعر غير صالح."

        # تحديد حجم الصفقة بالدولار من config
        amount_usdt = float(TRADE_AMOUNT_USDT)
        qty = amount_usdt / price

        # تنفيذ أمر السوق
        order = place_market_order(base, "buy", qty)

        # حفظ المركز المفتوح
        pos = {
            "symbol": symbol,
            "entry_price": price,
            "amount": qty,
            "opened_at": now_riyadh().isoformat(timespec="seconds"),
            "variant": symbol.split("#")[1] if "#" in symbol else "new",
            "score": None,
            "pattern": None,
            "reason": "AUTO_SIGNAL"
        }
        save_position(symbol, pos)

        msg = (
            f"✅ <b>تم فتح صفقة شراء</b>\n"
            f"📊 الرمز: <code>{symbol}</code>\n"
            f"💰 السعر: <b>{price:.6f}</b>\n"
            f"📦 الكمية: <b>{qty:.6f}</b>\n"
            f"⏱️ {datetime.now().strftime('%H:%M:%S')}"
        )
        return order, msg

    except Exception as e:
        err = f"❌ خطأ أثناء تنفيذ شراء {symbol}: {e}"
        _print(err)
        return None, err

# ================== أدوات تشخيص الإشارة ==================
def get_last_reject(symbol: str):
    if symbol in _LAST_REJECT:
        return _LAST_REJECT[symbol]
    base = symbol.split("/")[0]
    for k in (f"{base}|new", base):
        if k in _LAST_REJECT:
            return _LAST_REJECT[k]
    return None

def check_signal_debug(symbol: str):
    res = check_signal(symbol)
    if isinstance(res, dict) and res.get("decision") == "buy":
        return res, ["buy_ok"]
    last = get_last_reject(symbol)
    if last:
        return None, [f"{last.get('stage','-')}:{last.get('details',{})}"]
    return None, ["no_buy"]
# ================== وظائف مفقودة تكاملية ==================

# --- استيراد متغيرات إضافية من config ---
from config import (
    TRADE_AMOUNT_USDT, MAX_CONSEC_LOSSES, DAILY_LOSS_LIMIT_USDT
)

# --- إعداد افتراضي للإرسال إلى تيليجرام (لتجنب الخطأ إن لم يُعرّف) ---
STRAT_TG_SEND = True

# ================== إدارة ملفات المراكز ==================

def _pos_file(symbol: str):
    """توليد مسار ملف المركز بناءً على اسم الرمز"""
    safe = symbol.replace("/", "_").replace("#", "_")
    return os.path.join(STATE_PATH, f"pos_{safe}.json")

def save_position(symbol: str, pos: dict):
    """حفظ بيانات المركز في ملف JSON"""
    _atomic_write(_pos_file(symbol), pos)

def load_position(symbol: str):
    """تحميل بيانات المركز إن وُجد"""
    return _read_json(_pos_file(symbol), None)

def clear_position(symbol: str):
    """حذف ملف المركز لإغلاق الصفقة نهائيًا"""
    f = _pos_file(symbol)
    if os.path.exists(f):
        os.remove(f)

def load_closed_positions():
    """تحميل سجل المراكز المغلقة"""
    return _read_json(CLOSED_FILE, [])

def save_closed_positions(data):
    """حفظ سجل المراكز المغلقة"""
    _atomic_write(CLOSED_FILE, data)

# ================== وظائف التقارير اليومية ==================

def _hour_key(dt: datetime):
    """تحويل التوقيت إلى مفتاح الساعة (للتجميع في risk state)"""
    return dt.strftime("%H")

def _fmt_table(rows, headers):
    """تنسيق الجدول لتقرير اليوم بشكل HTML بسيط"""
    txt = "<pre>\n" + "\t".join(headers) + "\n" + "-"*80 + "\n"
    for r in rows:
        txt += "\t".join(str(x) for x in r) + "\n"
    return txt + "</pre>"

def _fmt_blocked_until_text():
    """عرض حالة الحظر الحالية في التقرير"""
    s = load_risk_state()
    bu = s.get("blocked_until")
    if not bu:
        return "✅ طبيعي"
    try:
        t = datetime.fromisoformat(bu)
        if now_riyadh() < t:
            return f"⛔️ حتى {t.strftime('%H:%M')}"
    except Exception:
        pass
    return "✅ طبيعي"

def _format_relax_str():
    """عرض مستوى Auto-Relax الحالي"""
    lvl = _relax_level_current()
    return "🧘 وضع عادي" if lvl == 0 else ("💤 Auto-Relax 1" if lvl == 1 else "🕊️ Auto-Relax 2")

# ================== أدوات الكاش والإحصائيات ==================

def reset_cycle_cache():
    """إعادة ضبط عدادات الرفض في بداية كل دورة"""
    _REJ_COUNTS["atr_low"] = 0
    _REJ_COUNTS["rvol"] = 0
    _REJ_COUNTS["notional_low"] = 0

def metrics_format():
    """تنسيق عدادات الأداء"""
    return (
        "📊 <b>Metrics</b>\n"
        f"ATR rej: {_REJ_COUNTS['atr_low']} | "
        f"RVOL rej: {_REJ_COUNTS['rvol']} | "
        f"Notional rej: {_REJ_COUNTS['notional_low']}\n"
    )

def breadth_status():
    """إرجاع حالة Breadth الحالية لاستخدامها في main.py"""
    br = _get_breadth_ratio_cached()
    bmin = _breadth_min_auto()
    return {
        "ratio": br,
        "min": bmin,
        "ok": (br is not None and br >= bmin)
    }
# ================== عدّ المراكز المفتوحة ==================
def count_open_positions() -> int:
    """
    يحسب عدد المراكز المفتوحة حاليًا بناءً على الملفات في مجلد state/
    """
    try:
        files = [f for f in os.listdir(STATE_PATH) if f.startswith("pos_") and f.endswith(".json")]
        return len(files)
    except Exception:
        return 0

# ================== نهاية الملف ==================
