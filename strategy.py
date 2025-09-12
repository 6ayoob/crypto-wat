# -*- coding: utf-8 -*-
"""
strategy.py — Spot-only (improved)
- كاش OHLCV للدورة + مِقاييس أداء.
- Retry/Backoff لاستدعاءات OHLCV.
- Position sizing ديناميكي (رأس مال × تقلب ATR × مضاعِف score).
- Circuit breaker بالساعة (اختياري عبر البيئة).
- Auto‑Relax ديناميكي عند جفاف الإشارات.
- Market Breadth Guard على HTF.
- دعم Fibonacci للدخول والأهداف.
- حارس Parabolic/Exhaustion لتفادي الاندفاع المبالغ فيه.
- أهداف متعددة (حتى TP5) وجزئيات متكيفة مع ATR%.
- MaxBars→TP1 ديناميكي بحسب ATR% و Score.

واجهات مطلوبة من main.py:
- check_signal(symbol)
- execute_buy(symbol)
- manage_position(symbol)
- close_trade(symbol, exit_price, pnl_net)
- build_daily_report_text()

يعتمد على okx_api: fetch_ohlcv, fetch_price, place_market_order, fetch_balance
وعلى config: TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP,
             TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME
"""

import os, json, requests, logging, time
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

# ===== لوج الاستراتيجية =====
logger = logging.getLogger("strategy")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEBUG_LOG_SIGNALS = os.getenv("DEBUG_LOG_SIGNALS", "0").lower() in ("1","true","yes","y")

def _rej(stage, **kv):
    if DEBUG_LOG_SIGNALS:
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"[REJECT] {stage} | {kvs}")
    return None

def _pass(stage, **kv):
    if DEBUG_LOG_SIGNALS:
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"[PASS]   {stage} | {kvs}")

# ================== إعدادات عامة ==================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

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
RESISTANCE_BUFFER, SUPPORT_BUFFER = 0.005, 0.002

# إدارة الصفقة
TP1_FRACTION = 0.5
MIN_NOTIONAL_USDT = 10.0
TRAIL_MIN_STEP_RATIO = 0.001

# مخاطر عامة
MAX_TRADES_PER_DAY       = 20
MAX_CONSEC_LOSSES        = 3
DAILY_LOSS_LIMIT_USDT    = 200.0

# مفاتيح الميزات
USE_EMA200_TREND_FILTER   = os.getenv("USE_EMA200_TREND_FILTER", "1").lower() in ("1","true","yes","y")
ENABLE_GOLDEN_CROSS_ENTRY = os.getenv("ENABLE_GOLDEN_CROSS_ENTRY", "1").lower() in ("1","true","yes","y")
GOLDEN_CROSS_RVOL_BOOST   = float(os.getenv("GOLDEN_CROSS_RVOL_BOOST", "1.10"))  # 1.10 = تشديد 10%

# ======= مفاتيح التحسينات الجديدة =======
# Position sizing ديناميكي
USE_DYNAMIC_RISK = os.getenv("USE_DYNAMIC_RISK", "1").lower() in ("1","true","yes","y")
RISK_PCT_OF_EQUITY = float(os.getenv("RISK_PCT_OF_EQUITY", "0.02"))  # 2% من رصيد USDT
MIN_TRADE_USDT = float(os.getenv("MIN_TRADE_USDT", "10"))
MAX_TRADE_USDT = float(os.getenv("MAX_TRADE_USDT", "1200"))
ATR_RISK_SCALER = float(os.getenv("ATR_RISK_SCALER", "2.0"))  # معامل تقليل حجم الصفقة عند تقلب عالي

# Circuit breaker بالساعة
HOURLY_DD_BLOCK_ENABLE = os.getenv("HOURLY_DD_BLOCK_ENABLE", "1").lower() in ("1","true","yes","y")
HOURLY_DD_PCT = float(os.getenv("HOURLY_DD_PCT", "0.05"))   # 5% من Equity خلال ساعة → حظر 60 دقيقة

# كاش HTF
_HTF_CACHE: Dict[str, Dict[str, Any]] = {}
_HTF_TTL_SEC = int(os.getenv("HTF_CACHE_TTL_SEC", "150"))

# ======= Auto‑Relax (جفاف الإشارات) =======
AUTO_RELAX_ENABLE = os.getenv("AUTO_RELAX_ENABLE", "1").lower() in ("1","true","yes","y")
# (hours, rvol_mult, atr_mult, ema50_bounds_mult)
_RELAX_LEVELS = [
    (6,  0.95, 0.95, 1.05),
    (12, 0.90, 0.92, 1.10),
]
_LAST_ANY_TRADE_AT: Optional[datetime] = None

# ======= Market Breadth Guard =======
MARKET_BREADTH_ENABLE = os.getenv("MARKET_BREADTH_ENABLE", "1").lower() in ("1","true","yes","y")
BREADTH_MIN_RATIO = float(os.getenv("BREADTH_MIN_RATIO", "0.60"))  # 55% فوق EMA50 على HTF
BREADTH_SAMPLE = int(os.getenv("BREADTH_SAMPLE", "25"))
_BREADTH_CACHE: Dict[str, Any] = {"ts": None, "ratio": None, "n": 0}
BREADTH_TTL_SEC = int(os.getenv("BREADTH_TTL_SEC", "240"))

# ======= كاش OHLCV للدورة + مِقاييس =======
_OHLCV_CACHE: Dict[tuple, list] = {}  # key=(symbol, tf, bars)

_METRICS = {
    "ohlcv_api_calls": 0,
    "ohlcv_cache_hits": 0,
    "ohlcv_cache_misses": 0,
    "htf_cache_hits": 0,
    "htf_cache_misses": 0,
}

def reset_cycle_cache():
    """يمسح كاش OHLCV ويصفر مِقاييس الجولة — نادِها في بداية كل جولة فحص من main.py."""
    _OHLCV_CACHE.clear()
    for k in _METRICS:
        _METRICS[k] = 0

def metrics_snapshot() -> dict:
    """نسخة من إحصاءات الجولة الحالية (للعرض/التشخيص)."""
    return dict(_METRICS)

def metrics_format() -> str:
    """تنسيق لطيف لإحصاءات الجولة (لطباعة/تلغرام)."""
    m = _METRICS
    return (
        "📈 <b>Metrics (this round)</b>\n"
        f"- OHLCV cache hits/misses: <b>{m['ohlcv_cache_hits']}/{m['ohlcv_cache_misses']}</b>\n"
        f"- OHLCV API calls: <b>{m['ohlcv_api_calls']}</b>\n"
        f"- HTF hits/misses: <b>{m['htf_cache_hits']}/{m['htf_cache_misses']}</b>"
    )

# ======= Retry/Backoff بسيط لـ fetch_ohlcv =======
def _retry_fetch_ohlcv(symbol, tf, bars, attempts=3, base_wait=1.2, max_wait=6.0):
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
    """واجهتنا الرسمية لجلب OHLCV مع عدّاد + retry."""
    _METRICS["ohlcv_api_calls"] += 1
    return _retry_fetch_ohlcv(symbol, tf, bars)

def get_ohlcv_cached(symbol: str, tf: str, bars: int) -> list:
    """كاش بسيط على مستوى الجولة (يمسح بـ reset_cycle_cache)."""
    key = (symbol, tf, bars)
    if key in _OHLCV_CACHE:
        _METRICS["ohlcv_cache_hits"] += 1
        return _OHLCV_CACHE[key]
    _METRICS["ohlcv_cache_misses"] += 1
    data = api_fetch_ohlcv(symbol, tf, bars)
    if data:
        _OHLCV_CACHE[key] = data
    return data

# ================== تتبُّع ==================
_LAST_REJECT: Dict[str, Any] = {}
_LAST_ENTRY_BAR_TS: Dict[str, int] = {}
_SYMBOL_LAST_TRADE_AT: Dict[str, datetime] = {}

# ================== إعدادات النسخ ==================
BASE_CFG = {
    "ENTRY_MODE": "hybrid",
    "HYBRID_ORDER": ["pullback","breakout"],
    "PULLBACK_VALUE_REF": "ema21",
    "PULLBACK_CONFIRM": "bullish_engulf",

    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.002,

    "USE_FIB": False,
    "SWING_LOOKBACK": 60,
    "FIB_TOL": 0.004,
    "BREAKOUT_BUFFER_LTF": 0.0015,
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

    "RVOL_MIN": 1.4,
    "ATR_MIN_FOR_TREND": 0.003,

    "USE_FIB": True,
    "BREAKOUT_BUFFER_LTF": 0.0018,
    "RSI_GATE_POLICY": "lenient",

    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,
    # أهداف إضافية (اختيارية)
    "TP3_ATR_MULT": 3.2,
    "TP4_ATR_MULT": 4.5,

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
    "TP3_ATR_MULT": 3.0,
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.1,
    "LOCK_MIN_PROFIT_PCT": 0.004,
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 8,
}

BRT_OVERRIDES = {
    "ENTRY_MODE": "breakout",
    "RVOL_MIN": 1.4,
    "ATR_MIN_FOR_TREND": 0.0022,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True, "SL_ATR_MULT": 0.9, "TP1_ATR_MULT": 1.4, "TP2_ATR_MULT": 2.4,
    "TP3_ATR_MULT": 3.4,
    "TRAIL_AFTER_TP1": True, "TRAIL_ATR_MULT": 1.1,
    "LOCK_MIN_PROFIT_PCT": 0.004, "MAX_HOLD_HOURS": 8, "SYMBOL_COOLDOWN_MIN": 10,
}

VBR_OVERRIDES = {
    "ENTRY_MODE": "pullback",
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True, "SL_ATR_MULT": 0.8, "TP1_ATR_MULT": 1.2, "TP2_ATR_MULT": 2.0,
    "TP3_ATR_MULT": 2.6,
    "TRAIL_AFTER_TP1": True, "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.003, "MAX_HOLD_HOURS": 6, "SYMBOL_COOLDOWN_MIN": 8,
}

RSI_MIN_PULLBACK, RSI_MAX_PULLBACK = 45, 65
RSI_MIN_BREAKOUT, RSI_MAX_BREAKOUT = 50, 80

PER_STRAT_MGMT = {
    "new": {"SL":"atr", "SL_MULT":0.9, "TP1":"sr_or_atr", "TP1_ATR":1.2, "TP2_ATR":2.2,
             "TP3_ATR":3.2, "TP4_ATR":4.5,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":1.0, "TIME_HRS":6},
    "old": {"SL":"pct", "SL_PCT":0.02, "TP1_PCT":0.03, "TP2_PCT":0.06,
             "TRAIL_AFTER_TP1":False, "TIME_HRS":12},
    "srr": {"SL":"atr_below_sweep", "SL_MULT":0.8, "TP1":"sr_or_atr", "TP1_ATR":1.0, "TP2_ATR":2.2,
             "TP3_ATR":3.0,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":1.0, "TIME_HRS":4},
    "brt": {"SL":"atr_below_retest", "SL_MULT":1.0, "TP1":"range_or_atr", "TP1_ATR":1.5, "TP2_ATR":2.5,
             "TP3_ATR":3.5,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":0.9, "TIME_HRS":8},
    "vbr": {"SL":"atr", "SL_MULT":1.0, "TP1":"vwap_or_sr", "TP2_ATR":1.8, "TP3_ATR":2.6,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":0.8, "TIME_HRS":3},
}

def _mgmt(variant: str): return PER_STRAT_MGMT.get(variant, PER_STRAT_MGMT["new"])

# ======= فلترة متعددة الفريمات =======
ENABLE_MTF_STRICT = True
SCORE_THRESHOLD = 60

SR_LEVELS_CFG = [
    ("micro", LTF_TIMEFRAME,  50, 0.8),
    ("meso",  "1h",  50, 1.0),
    ("macro", "4h",  50, 1.3),
]

# ================== Helpers ==================
def _tg(text, parse_mode="HTML"):
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        if parse_mode: data["parse_mode"] = parse_mode
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass

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

def _split_symbol_variant(symbol: str):
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower()
        if variant not in ("old","new","srr","brt","vbr"):
            variant = "new"
        return base, variant
    return symbol, "new"

def get_cfg(variant: str):
    cfg = dict(BASE_CFG)
    if variant == "new": cfg.update(NEW_SCALP_OVERRIDES)
    elif variant == "srr": cfg.update(SRR_OVERRIDES)
    elif variant == "brt": cfg.update(BRT_OVERRIDES)
    elif variant == "vbr": cfg.update(VBR_OVERRIDES)
    return cfg

# ================== تخزين الصفقات ==================
def _pos_path(symbol):
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return f"{POSITIONS_DIR}/{symbol.replace('/', '_')}.json"

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

# ================== حالة المخاطر اليومية/الساعة ==================
def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0,
            "trades_today": 0, "blocked_until": None, "hourly_pnl": {}, "relax_wins": 0}

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str():
        s = _default_risk_state(); save_risk_state(s)
    # تهيئة hourly_pnl
    if "hourly_pnl" not in s or not isinstance(s["hourly_pnl"], dict):
        s["hourly_pnl"] = {}
    if "relax_wins" not in s:
        s["relax_wins"] = 0
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)

def register_trade_opened():
    global _LAST_ANY_TRADE_AT
    _LAST_ANY_TRADE_AT = now_riyadh()
    s = load_risk_state(); s["trades_today"] = int(s.get("trades_today", 0)) + 1; save_risk_state(s)

def _set_block(minutes, reason="risk"):
    s = load_risk_state(); until = now_riyadh() + timedelta(minutes=minutes)
    s["blocked_until"] = until.isoformat(timespec="seconds"); save_risk_state(s)
    _tg(f"⛔️ <b>تم تفعيل حظر مؤقت</b> ({reason}) حتى <code>{until.strftime('%H:%M')}</code>.")

def _is_blocked():
    s = load_risk_state(); bu = s.get("blocked_until")
    if not bu: return False
    try: t = datetime.fromisoformat(bu)
    except Exception: return False
    return now_riyadh() < t

def register_trade_result(pnl_usdt):
    """تحديث حالة المخاطر بعد كل خروج جزئي/كامل + Circuit breaker بالساعة."""
    global _LAST_ANY_TRADE_AT
    _LAST_ANY_TRADE_AT = now_riyadh()
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1


    # تتبع نجاحات للتخفيف: نعيد النظام للوضع الطبيعي بعد صفقتين ناجحتين (ليستا بالضرورة متتاليتين)
    try:
        if float(pnl_usdt) > 0:
            s["relax_wins"] = int(s.get("relax_wins", 0)) + 1
        else:
            s["relax_wins"] = int(s.get("relax_wins", 0))
    except Exception:
        pass
    # حساب PnL الساعة الحالية
    hk = _hour_key(now_riyadh())
    s["hourly_pnl"][hk] = float(s["hourly_pnl"].get(hk, 0.0)) + float(pnl_usdt)

    # منطق الحظر الأساسي (خسائر متتالية/حد يومي)
    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s); _set_block(90, reason="خسائر متتالية"); return
    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s); _set_block(minutes, reason="تجاوز حد الخسارة اليومي"); return

    # Circuit breaker بالساعة (اختياري)
    if HOURLY_DD_BLOCK_ENABLE:
        try:
            # Equity = رصيد USDT الحالي (تبسيط)
            equity = float(fetch_balance("USDT") or 0.0)
            hour_pnl = float(s["hourly_pnl"].get(hk, 0.0))
            if equity > 0 and (hour_pnl <= -abs(HOURLY_DD_PCT) * equity):
                save_risk_state(s); _set_block(60, reason=f"هبوط {HOURLY_DD_PCT*100:.1f}%/ساعة"); return
        except Exception:
            pass

    save_risk_state(s)

def _risk_precheck_allow_new_entry():
    if _is_blocked():  return False, "blocked"
    s = load_risk_state()
    if MAX_TRADES_PER_DAY and s.get("trades_today", 0) >= MAX_TRADES_PER_DAY: return False, "max_trades"
    if s.get("daily_pnl", 0.0) <= -abs(DAILY_LOSS_LIMIT_USDT): return False, "daily_loss_limit"
    if s.get("consecutive_losses", 0) >= MAX_CONSEC_LOSSES: return False, "consec_losses"
    return True, ""

# ================== مؤشرات ==================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff(); gain = d.where(d > 0, 0.0); loss = -d.where(d < 0, 0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean(); al = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    rs = ag / al; return 100 - (100 / (1 + rs))

def macd_cols(df, fast=12, slow=26, signal=9):
    df["ema_fast"], df["ema_slow"] = ema(df["close"], fast), ema(df["close"], slow)
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_indicators(df):
    df["ema9"]  = ema(df["close"], EMA_FAST)
    df["ema21"] = ema(df["close"], EMA_SLOW)
    df["ema50"] = ema(df["close"], EMA_TREND)
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
    vol_ma = df["volume"].rolling(RVOL_WINDOW).mean()
    df["rvol"] = df["volume"] / vol_ma.replace(0, 1e-9)
    rng = df["high"] - df["low"]; rng_ma = rng.rolling(NR_WINDOW).mean()
    df["is_nr"] = rng < (NR_FACTOR * rng_ma)
    df["body"] = (df["close"] - df["open"]).abs(); df["avg_body20"] = df["body"].rolling(20).mean()
    return df

def _atr_from_df(df, period=ATR_PERIOD):
    c = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-c).abs(), (df["low"]-c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-2])

# ===== Swing/SR/Fib =====
def _swing_points(df, left=2, right=2):
    highs, lows = df["high"], df["low"]
    idx = len(df) - 3
    swing_high = swing_low = None
    for i in range(max(0, idx-10), idx+1):
        if i-left < 0 or i+right >= len(df): continue
        if highs[i] == max(highs[i-left:i+right+1]): swing_high = float(highs[i])
        if lows[i]  == min(lows[i-left:i+right+1]): swing_low  = float(lows[i])
    return swing_high, swing_low

def _bullish_engulf(prev, cur):
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and \
           (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])

def get_sr_on_closed(df, window=SR_WINDOW):
    if len(df) < window + 3: return None, None
    df_prev = df.iloc[:-1]; w = min(window, len(df_prev))
    resistance = df_prev["high"].rolling(w, min_periods=max(5, w//3)).max().iloc[-1]
    support    = df_prev["low"].rolling(w,  min_periods=max(5, w//3)).min().iloc[-1]
    if pd.isna(resistance) or pd.isna(support): return None, None
    return float(support), float(resistance)

def recent_swing(df, lookback=60):
    if len(df) < lookback + 5: return None, None
    seg = df.iloc[-(lookback+1):-1]; hhv = seg["high"].max(); llv = seg["low"].min()
    if pd.isna(hhv) or pd.isna(llv) or hhv <= llv: return None, None
    return float(hhv), float(llv)

def _rolling_sr(symbol, tf: str, window: int, bars: int = 300):
    data = get_ohlcv_cached(symbol, tf, bars)
    if not data: return None, None
    df = _df(data)
    if len(df) < window + 3: return None, None
    df_prev = df.iloc[:-1]
    w = min(window, len(df_prev))
    res = df_prev["high"].rolling(w, min_periods=max(5, w//3)).max().iloc[-1]
    sup = df_prev["low"].rolling(w,  min_periods=max(5, w//3)).min().iloc[-1]
    if pd.isna(res) or pd.isna(sup): return None, None
    return float(sup), float(res)

def get_sr_multi(symbol: str):
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

# Fib helpers
def _fib_levels_from_recent(df, lookback=60):
    h, l = recent_swing(df, lookback=lookback)
    if not h or not l or h <= l:
        return None
    rng = h - l
    # مستويات شائعة للارتداد والامتداد
    levels = {
        "0.382": h - 0.382 * rng,
        "0.5":   h - 0.5   * rng,
        "0.618": h - 0.618 * rng,
        "1.272": h + 0.272 * rng,
        "1.618": h + 0.618 * rng,
    }
    return {"high": h, "low": l, "levels": levels, "range": rng}

# ===== MACD/RSI Gate =====
def macd_rsi_gate(prev_row, closed_row, policy):
    if not policy:  # None → بوابة متوقفة
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

# ================== HTF سياق ==================
def _get_htf_context(symbol):
    base, _ = _split_symbol_variant(symbol)
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
    resistance = df_prev["high"].rolling(w).max().iloc[-1]
    support    = df_prev["low"].rolling(w).min().iloc[-1]
    closed = df.iloc[-2]
    ema_now  = float(closed["ema50_htf"])
    ema_prev = float(df["ema50_htf"].iloc[-7]) if len(df) >= 7 else ema_now

    ctx: Dict[str, Any] = {"close": float(closed["close"]), "ema50_now": ema_now, "ema50_prev": ema_prev,
           "support": float(support), "resistance": float(resistance), "mtf": {}}

    if ENABLE_MTF_STRICT:
        def _tf_info(tf, bars=160):
            try:
                d = get_ohlcv_cached(base, tf, bars)
                if not d or len(d) < 80: return None
                _dfx = _df(d); _dfx[f"ema{HTF_EMA_TREND_PERIOD}"] = ema(_dfx["close"], HTF_EMA_TREND_PERIOD)
                row = _dfx.iloc[-2]
                return {"tf": tf, "price": float(row["close"]),
                        "ema": float(row[f"ema{HTF_EMA_TREND_PERIOD}"]),
                        "trend_up": bool(row["close"] > row[f"ema{HTF_EMA_TREND_PERIOD}"])}
            except Exception:
                return None

        mtf = {HTF_TIMEFRAME: {"tf": HTF_TIMEFRAME, "price": ctx["close"], "ema": ctx["ema50_now"],
                         "trend_up": bool(ctx["close"] > ctx["ema50_now"])}}
        for tf in ("1h","4h"):
            info = _tf_info(tf)
            if info: mtf[tf] = info
        ctx["mtf"] = mtf

    _HTF_CACHE[base] = {"t": now, "ctx": ctx}
    return ctx

# ======= Market breadth (HTF > EMA50) =======
def _market_breadth_ratio():
    if not MARKET_BREADTH_ENABLE:
        return 1.0, 0
    now = now_riyadh()
    ts = _BREADTH_CACHE.get("ts")
    if ts and (now - ts).total_seconds() <= BREADTH_TTL_SEC:
        return float(_BREADTH_CACHE.get("ratio", 1.0)), int(_BREADTH_CACHE.get("n", 0))

    sample = []
    seen = set()
    for s in SYMBOLS:
        base = s.split("#")[0]
        if base in seen: continue
        seen.add(base)
        sample.append(base)
        if len(sample) >= BREADTH_SAMPLE: break

    ups = 0
    for base in sample:
        try:
            ctx = _get_htf_context(base)
            if not ctx: continue
            if ctx["close"] > ctx["ema50_now"] and (ctx["ema50_now"] >= ctx["ema50_prev"]):
                ups += 1
        except Exception:
            continue
    ratio = (ups / max(1, len(sample))) if sample else 1.0
    _BREADTH_CACHE["ts"] = now
    _BREADTH_CACHE["ratio"] = ratio
    _BREADTH_CACHE["n"] = len(sample)
    return ratio, len(sample)

# ================== منطق الدخول الأساسية ==================

def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx, cfg, fib=None):
    ref_val = closed["ema21"] if cfg["PULLBACK_VALUE_REF"]=="ema21" else closed.get("vwap", closed["ema21"])
    if pd.isna(ref_val): ref_val = closed["ema21"]
    near_val = (closed["close"] >= ref_val) and (closed["low"] <= ref_val)
    if not near_val: return False
    # تأكيد النمط
    if cfg.get("PULLBACK_CONFIRM") == "bullish_engulf":
        if not _bullish_engulf(prev, closed):
            return False
    elif cfg.get("PULLBACK_CONFIRM") == "bos":
        swing_high, _ = _swing_points(df); 
        if not (swing_high and closed["close"] > swing_high):
            return False
    # شرط فيبوناتشي (اختياري)
    if cfg.get("USE_FIB") and fib:
        try:
            fib_tol = float(cfg.get("FIB_TOL", 0.004))
            zone_low  = fib["levels"]["0.618"]
            zone_high = fib["levels"]["0.382"]
            price = float(closed["close"])
            in_zone = (price >= zone_low * (1 - fib_tol)) and (price <= zone_high * (1 + fib_tol))
            if not in_zone:
                return False
        except Exception:
            pass
    return True

def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
    is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
    vwap_ok = closed["close"] > float(closed.get("vwap", closed["ema21"]))
    buf = float(cfg.get("BREAKOUT_BUFFER_LTF", 0.0015))
    return (closed["close"] > hi_range * (1.0 + buf)) and (is_nr_recent or vwap_ok)

# ---------- Scoring ----------

def _opportunity_score(df, prev, closed):
    score, why, pattern = 0, [], ""
    try:
        if closed["close"] > closed["open"]:
            score += 10; why.append("BullishClose")
        if closed["close"] > closed.get("ema21", closed["close"]):
            score += 10; why.append("AboveEMA21")
        if closed["close"] > closed.get("ema50", closed["close"]):
            score += 10; why.append("AboveEMA50")
        rvol = float(closed.get("rvol", 0) or 0)
        if rvol >= 1.5:
            score += 15; why.append("HighRVOL")
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        if nr_recent and (closed["close"] > hi_range):
            score += 20; why.append("NR_Breakout"); pattern = "NR_Breakout"
        if _bullish_engulf(prev, closed):
            score += 20; why.append("BullishEngulf"); pattern = pattern or "BullishEngulf"
    except Exception:
        pass
    return score, ", ".join(why), (pattern or "Generic")

# ===== Exhaustion / Parabolic guard =====
EXHAUSTION_GUARD_ENABLE = os.getenv("EXHAUSTION_GUARD_ENABLE", "1").lower() in ("1","true","yes","y")
EXH_RSI_MAX = float(os.getenv("EXH_RSI_MAX", "76"))
EXH_UP_BARS = int(os.getenv("EXH_UP_BARS", "5"))
EXH_MAX_DIST_ATR = float(os.getenv("EXH_MAX_DIST_ATR", "4.8"))

def _exhaustion_block(df, closed) -> bool:
    if not EXHAUSTION_GUARD_ENABLE:
        return False
    try:
        rsi_val = float(closed.get("rsi", 50))
        if rsi_val >= EXH_RSI_MAX:
            return True
        # متتالية ارتفاعات
        last = df.iloc[-(EXH_UP_BARS+1):-1]
        ups = int((last["close"] > last["open"]).sum())
        if ups >= EXH_UP_BARS:
            return True
        # بعد مبالغ فيه عن EMA50
        atr_val = _atr_from_df(df)
        if atr_val and atr_val > 0:
            dist_atr = (float(closed["close"]) - float(closed.get("ema50", closed["close"])))/atr_val
            if dist_atr > EXH_MAX_DIST_ATR:
                return True
    except Exception:
        return False
    return False

# ================== NEW/SRR — متكّيف بالـ ATR + EMA200 + GoldenCross + Auto‑Relax + Breadth ==================

def _auto_relax_factors():
    if not AUTO_RELAX_ENABLE:
        return 1.0, 1.0, 1.0
    # إذا تحقّقت صفقتان ناجحتان، نعيد عوامل التخفيف للوضع الطبيعي ونصفر العداد
    s = load_risk_state()
    try:
        if int(s.get("relax_wins", 0)) >= 2:
            s["relax_wins"] = 0
            save_risk_state(s)
            global _LAST_ANY_TRADE_AT
            _LAST_ANY_TRADE_AT = now_riyadh()
            return 1.0, 1.0, 1.0
    except Exception:
        pass  # rvol_mult, atr_mult, ema50_bounds_mult
    if _LAST_ANY_TRADE_AT is None:
        return 1.0, 1.0, 1.0
    hours = (now_riyadh() - _LAST_ANY_TRADE_AT).total_seconds() / 3600.0
    rvol_m, atr_m, ema_bounds_m = 1.0, 1.0, 1.0
    for h, rv, at, eb in _RELAX_LEVELS:
        if hours >= h:
            rvol_m, atr_m, ema_bounds_m = rv, at, eb
    return rvol_m, atr_m, ema_bounds_m

def check_signal_new(symbol):
    """يفحص إشارة شراء Spot فقط على الرمز المحدد (نسخ: new/srr/brt/vbr). يعيد dict{'decision':'buy', ...} أو None."""
    # حارس المخاطر العامة
    ok, reason = _risk_precheck_allow_new_entry()
    if not ok: return _rej("risk_precheck", reason=reason)

    # حارس Market Breadth
    ratio, n = _market_breadth_ratio()
    if MARKET_BREADTH_ENABLE and ratio < BREADTH_MIN_RATIO:
        return _rej("breadth_guard", ratio=round(ratio,2), min=BREADTH_MIN_RATIO, n=n)

    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)
    key = f"{base}|{variant}"

    last_t = _SYMBOL_LAST_TRADE_AT.get(key)
    if last_t and (now_riyadh() - last_t) < timedelta(minutes=cfg["SYMBOL_COOLDOWN_MIN"]):
        return _rej("cooldown")
    if load_position(symbol):
        return _rej("already_open")

    ctx = _get_htf_context(symbol)
    if not ctx: return _rej("htf_none")
    if not ((ctx["ema50_now"] - ctx["ema50_prev"]) > 0 and ctx["close"] > ctx["ema50_now"]):
        return _rej("htf_trend")

    data = get_ohlcv_cached(base, LTF_TIMEFRAME, 260)
    if not data: return _rej("ltf_fetch")
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 200: return _rej("ltf_len", n=len(df))  # نحتاج EMA200 أيضًا

    prev, closed = df.iloc[-3], df.iloc[-2]
    if _exhaustion_block(df, closed):
        return _rej("exhaustion_guard")

    ts = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == ts: return _rej("dup_bar")

    price = float(closed["close"])
    atr = _atr_from_df(df)
    if not atr or atr <= 0: return _rej("atr_nan")
    atrp = atr / max(price, 1e-9)

    # Auto‑Relax adjustments
    rv_mult, atr_mult, ema_bounds_mult = _auto_relax_factors()

    if atrp < float(cfg.get("ATR_MIN_FOR_TREND", 0.002)) * atr_mult:
        return _rej("atr_low", atrp=round(atrp,5))

    notional = price * float(closed["volume"])
    if notional < 60000: return _rej("notional_low", notional=int(notional))

    rvol = float(closed.get("rvol", 0) or 0)
    need_rvol = float(cfg.get("RVOL_MIN", 1.2)) * rv_mult
    if pd.isna(rvol) or rvol < need_rvol:
        return _rej("rvol", rvol=round(rvol,2), need=round(need_rvol,2))

    # فلتر ترند EMA200 (اختياري)
    if USE_EMA200_TREND_FILTER:
        if not (float(closed.get("ema50", price)) > float(closed.get("ema200", price)) and price > float(closed.get("ema200", price))):
            return _rej("ema200_trend")

    # تحضير فيبوناتشي (إن وُجد)
    fib = _fib_levels_from_recent(df, lookback=int(cfg.get("SWING_LOOKBACK", 60))) if cfg.get("USE_FIB") else None

    # اختيار النمط الأساسي (hybrid)
    def _brk_ok():
        hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        vwap_ok = price > float(closed.get("vwap", closed.get("ema21", price)))
        buf = float(cfg.get("BREAKOUT_BUFFER_LTF", 0.0015))
        return (price > hi_range * (1.0 + buf)) and (is_nr_recent or vwap_ok)

    chosen_mode = None; mode_ok = False
    entry_mode = cfg.get("ENTRY_MODE", "hybrid")
    if entry_mode == "pullback":
        chosen_mode = "pullback"; mode_ok = _entry_pullback_logic(df, closed, prev, atr, ctx, cfg, fib=fib)
    elif entry_mode == "breakout":
        chosen_mode = "breakout"; mode_ok = _brk_ok()
    else:
        for m in cfg.get("HYBRID_ORDER", ["breakout","pullback"]):
            if m == "breakout" and _brk_ok():
                chosen_mode = "breakout"; mode_ok = True; break
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr, ctx, cfg, fib=fib):
                chosen_mode = "pullback"; mode_ok = True; break

    # بديل: دخول Golden Cross (اختياري)
    golden_cross_ok = False
    if ENABLE_GOLDEN_CROSS_ENTRY:
        try:
            prev50, prev200 = float(prev["ema50"]), float(prev["ema200"])
            now50, now200 = float(closed["ema50"]), float(closed["ema200"])
            golden_cross_ok = (prev50 <= prev200) and (now50 > now200) and (price > now50)
            if golden_cross_ok and rvol < need_rvol * GOLDEN_CROSS_RVOL_BOOST:
                golden_cross_ok = False
        except Exception:
            golden_cross_ok = False

    if not mode_ok:
        if golden_cross_ok:
            chosen_mode = "golden_cross"; mode_ok = True
        else:
            return _rej("entry_mode", mode=entry_mode)

    # شرط المسافة الديناميكي عن EMA50
    dist = price - float(closed["ema50"])
    dist_atr = dist / atr
    if chosen_mode == "pullback":
        lb, ub = 0.15, 2.5
        if rvol >= need_rvol * 1.20: lb = 0.10
    elif chosen_mode == "breakout":
        lb, ub = 0.50, 4.0
        if rvol >= need_rvol * 1.30: ub = 4.5
    else:  # golden_cross
        lb, ub = 0.00, 4.0
        if rvol >= need_rvol * 1.30: ub = 4.5
    # Auto‑Relax يوسّع الحدود قليلًا
    lb, ub = lb/ema_bounds_mult, ub*ema_bounds_mult
    if not (lb <= dist_atr <= ub):
        return _rej("dist_to_ema50", dist_atr=round(dist_atr,3), lb=round(lb,2), ub=round(ub,2))

    # نطاقات RSI حسب النمط
    rsi_val = float(closed.get("rsi", 50))
    if chosen_mode == "pullback" and not (RSI_MIN_PULLBACK - 3 < rsi_val < RSI_MAX_PULLBACK + 2):
        return _rej("rsi_pullback", rsi=rsi_val)
    if chosen_mode in ("breakout","golden_cross") and not (RSI_MIN_BREAKOUT - 2 < rsi_val < RSI_MAX_BREAKOUT + 2):
        return _rej("rsi_breakout", rsi=rsi_val)

    # قرب مقاومة من طبقات متعددة + LTF/HTF
    sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
    sr_multi = get_sr_multi(symbol)
    near_res_any = False
    for name, ent in sr_multi.items():
        res = ent.get("resistance")
        if res and (res - price) < (ent["near_mult"] * atr):
            near_res_any = True; break
    near_res = (res_ltf and (res_ltf - price) < 0.8*atr) or (ctx.get("resistance") and (ctx["resistance"] - price) < 1.2*atr) or near_res_any
    if near_res and chosen_mode != "breakout":
        return _rej("near_res_block")

    score, why, patt = _opportunity_score(df, prev, closed)
    if chosen_mode == "breakout": patt = "NR_Breakout"
    if chosen_mode == "golden_cross":
        patt = "EMA50x200_Golden"; score += 12
        if score < (SCORE_THRESHOLD - 15):
            return _rej("score_low", score=score)
    else:
        if score < SCORE_THRESHOLD:
            return _rej("score_low", score=score)

    _LAST_ENTRY_BAR_TS[key] = ts
    _pass("signal_ok", mode=chosen_mode, score=score, rvol=round(rvol,2), atrp=round(atrp,4))
    return {"decision": "buy", "score": score, "reason": why, "pattern": patt, "ts": ts, "fib": fib}

# ================== OLD/SRR/BRT/VBR ==================

def check_signal_old(symbol): return check_signal_new(symbol)

def check_signal_brt(symbol): return check_signal_new(symbol)

def check_signal_vbr(symbol): return check_signal_new(symbol)

# ================== Router ==================

def check_signal(symbol):
    """Router لفحص الإشارة حسب النسخة suffix#."""
    base, variant = _split_symbol_variant(symbol)
    if variant == "old": return check_signal_old(symbol)
    if variant == "srr": return check_signal_new(symbol)
    if variant == "brt": return check_signal_brt(symbol)
    if variant == "vbr": return check_signal_vbr(symbol)
    return check_signal_new(symbol)

# ================== SL/TP ==================

def _compute_sl_and_targets(entry, atr_val, cfg, variant, symbol=None, df=None, ctx=None, closed=None, fib=None):
    mg = _mgmt(variant)
    sl = None
    try:
        if mg.get("SL") == "atr":
            sl = entry - mg.get("SL_MULT", 1.0) * atr_val
        elif mg.get("SL") == "pct":
            sl = entry * (1 - float(mg.get("SL_PCT", cfg.get("STOP_LOSS_PCT", 0.02))))
        elif mg.get("SL") in ("atr_below_sweep", "atr_below_retest"):
            base_level = None
            try:
                if df is not None and len(df) > 10:
                    _, sw_low = _swing_points(df, left=2, right=2)
                    _, llv = recent_swing(df, lookback=60)
                    base_level = max(float(sw_low or 0), float(llv or 0)) or None
            except Exception:
                base_level = None
            if base_level and base_level < entry:
                sl = float(base_level) - mg.get("SL_MULT", 1.0) * atr_val
            else:
                sl = entry - mg.get("SL_MULT", 1.0) * atr_val
        else:
            if cfg.get("USE_ATR_SL_TP") and atr_val and atr_val > 0:
                sl  = entry - cfg.get("SL_ATR_MULT", 1.6)  * atr_val
            else:
                sl  = entry * (1 - cfg.get("STOP_LOSS_PCT", 0.02))
    except Exception:
        sl = entry - 1.0 * atr_val

    # أهداف افتراضية بالـ ATR
    tp1 = entry + float(mg.get("TP1_ATR", cfg.get("TP1_ATR_MULT", 1.2))) * atr_val
    tp2 = entry + float(mg.get("TP2_ATR", cfg.get("TP2_ATR_MULT", 2.2))) * atr_val
    targets = [float(tp1), float(tp2)]

    # دمج SR/VWAP للـ TP1 عند الحاجة
    try:
        nearest_res = None
        if symbol:
            sr_multi = get_sr_multi(symbol)
            for name, ent in sr_multi.items():
                res = ent.get("resistance")
                if res and res > entry:
                    nearest_res = res if nearest_res is None else min(nearest_res, res)
        if mg.get("TP1") in ("sr_or_atr", "range_or_atr", "vwap_or_sr"):
            cand = targets[0]
            if mg.get("TP1") == "vwap_or_sr" and closed is not None:
                vwap_val = float(closed.get("vwap")) if closed.get("vwap") else None
                if vwap_val and vwap_val > entry:
                    cand = min(cand, vwap_val)
            if nearest_res and nearest_res > entry:
                cand = min(cand, nearest_res)
            targets[0] = float(cand)
    except Exception:
        pass

    # أهداف إضافية من الإعداد أو من Fib extension
    try:
        if mg.get("TP3_ATR") or cfg.get("TP3_ATR_MULT"):
            targets.append(entry + float(mg.get("TP3_ATR", cfg.get("TP3_ATR_MULT", 3.2))) * atr_val)
        if mg.get("TP4_ATR") or cfg.get("TP4_ATR_MULT"):
            targets.append(entry + float(mg.get("TP4_ATR", cfg.get("TP4_ATR_MULT", 4.5))) * atr_val)
        # Fib extensions
        if fib and isinstance(fib, dict) and fib.get("levels"):
            ext1618 = fib["levels"].get("1.618")
            if ext1618 and ext1618 > entry:
                targets.append(float(ext1618))
    except Exception:
        pass

    # إزالة التكرار وترتيب تصاعدي
    targets = sorted(list({round(float(t), 10) for t in targets}))
    # فلترة المستويات القريبة جدًا (أقل من 0.2*ATR بينها)
    filtered = []
    for t in targets:
        if not filtered or abs(t - filtered[-1]) >= 0.2 * atr_val:
            filtered.append(t)
    return float(sl), filtered[:5]

# ================== تنفيذ الشراء ==================

def execute_buy(symbol):
    """
    تنفيذ شراء Spot-only للرمز المحدّد.
    - يستخدم Position sizing ديناميكي إذا USE_DYNAMIC_RISK=1 (افتراضي) مع مضاعف score.
    - يخضع لمنطق الحظر/المخاطر.
    """
    base, variant = _split_symbol_variant(symbol)

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."
    if _is_blocked():
        return None, "🚫 ممنوع فتح صفقات الآن (حظر مخاطرة)."
    if load_position(symbol):
        return None, "🚫 لديك صفقة مفتوحة على هذا الرمز/الاستراتيجية."

    ohlcv = get_ohlcv_cached(base, LTF_TIMEFRAME, 220)
    if not ohlcv:
        return None, "⚠️ فشل جلب بيانات الشموع."

    _sig_inner = check_signal(symbol)
    if not _sig_inner:
        return None, "❌ لا توجد إشارة مطابقة."

    df_exec = _df(ohlcv)
    df_exec = _ensure_ltf_indicators(df_exec)
    price_fallback = float(df_exec.iloc[-2]["close"])
    closed = df_exec.iloc[-2]
    atr_val = _atr_from_df(df_exec)
    cfg = get_cfg(variant)
    ctx = _get_htf_context(symbol)

    sl, targets = _compute_sl_and_targets(price_fallback, atr_val, cfg, variant, symbol=symbol, df=df_exec, ctx=ctx, closed=closed, fib=_sig_inner.get("fib"))
    mg = _mgmt(variant)
    custom = (_sig_inner.get("custom") if isinstance(_sig_inner, dict) else {}) or {}
    if "sl" in custom and isinstance(custom["sl"], (int, float)):
        sl = float(custom["sl"])
    if "tp1" in custom and isinstance(custom["tp1"], (int, float)) and targets:
        targets[0] = min(targets[0], float(custom["tp1"]))

    score = float(_sig_inner.get("score", 70))
    price = float(price_fallback)
    usdt = float(fetch_balance("USDT") or 0)

    # ===== Position sizing ديناميكي + score multiplier =====
    if USE_DYNAMIC_RISK:
        equity = usdt  # تبسيط: نعتبر USDT المتاح ≈ Equity
        base_risk = equity * RISK_PCT_OF_EQUITY
        risk_usdt = min(MAX_TRADE_USDT, max(MIN_TRADE_USDT, base_risk))
        atrp = (atr_val / max(price, 1e-9)) if atr_val else 0.0
        vol_factor = 1.0 / (1.0 + ATR_RISK_SCALER * max(0.0, atrp))  # كلما زاد ATR% قلّ الحجم
        # مضاعِف score
        score_mult = 1.0
        if score >= 92: score_mult = 1.35
        elif score >= 85: score_mult = 1.20
        elif score <= 65: score_mult = 0.80
        trade_usdt = max(MIN_TRADE_USDT, risk_usdt * vol_factor * score_mult)
    else:
        trade_usdt = TRADE_AMOUNT_USDT

    if usdt < max(MIN_TRADE_USDT, trade_usdt):
        return None, "🚫 رصيد USDT غير كافٍ."

    amount = trade_usdt / price
    if amount * price < MIN_NOTIONAL_USDT:
        return None, "🚫 قيمة الصفقة أقل من الحد الأدنى."

    order = place_market_order(base, "buy", amount)
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    fill_px = float(order.get("average") or order.get("price") or price)

    # ===== جزئيات متكيفة مع ATR% =====
    atrp = (atr_val / max(fill_px, 1e-9)) if atr_val else 0.0
    if len(targets) < 3:
        # ضمّن على الأقل 3 أهداف
        last = targets[-1] if targets else fill_px + 2.2*atr_val
        targets = [targets[0], targets[1], last + 0.8*atr_val] if len(targets) >= 2 else [fill_px + 1.2*atr_val, fill_px + 2.2*atr_val, fill_px + 3.2*atr_val]

    if atrp < 0.007:
        partials = [0.40, 0.30, 0.20, 0.10][:len(targets)]
    elif atrp < 0.015:
        partials = [0.50, 0.30, 0.15, 0.05][:len(targets)]
    else:
        partials = [0.60, 0.25, 0.10, 0.05][:len(targets)]
    # اضبط الطول (حتى 5 أهداف)
    if len(targets) > 5:
        targets = targets[:5]
        partials = partials[:5]
    # إذا عدد الجزئيات أقل من الأهداف، أكمل بجزئيات صغيرة
    while len(partials) < len(targets):
        partials.append(max(0.05, 1 - sum(partials)))
    # طبّع المجموع = 1.0
    ssum = sum(partials)
    if ssum != 0:
        partials = [p/ssum for p in partials]

    # ===== Max Bars to TP1 ديناميكي =====
    max_bars_to_tp1 = 12
    if atrp <= 0.006: max_bars_to_tp1 = 14
    if atrp >= 0.015: max_bars_to_tp1 = 9
    if score >= 90:   max_bars_to_tp1 += 1
    if score <= 65:   max_bars_to_tp1 -= 1
    max_bars_to_tp1 = int(max(6, min(16, max_bars_to_tp1)))

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(fill_px),
        "stop_loss": float(sl),
        "targets": [float(t) for t in targets],
        "partials": [float(p) for p in partials],
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": variant,
        "htf_stop": None,
        "max_bars_to_tp1": max_bars_to_tp1,
        "messages": {"entry": f"🚀 دخول {_sig_inner.get('pattern','Opportunity')}"},
        "tp_hits": [False] * len(targets),
        "score": score,
        "pattern": _sig_inner.get("pattern"),
        "reason": _sig_inner.get("reason"),
    }
    save_position(symbol, pos)
    register_trade_opened()

    _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()

    try:
        _tg(
            f"{pos['messages']['entry']}\n"
            f"دخول: <code>{fill_px:.6f}</code>\n"
            f"SL: <code>{pos['stop_loss']:.6f}</code>\n"
            f"أهداف: {', '.join(str(round(t,6)) for t in pos['targets'])}\n"
            f"🎯 Partials: {', '.join(f'{int(p*100)}%' for p in pos['partials'])}\n"
            f"💰 حجم الصفقة: <b>{trade_usdt:.2f}$</b>"
        )
    except Exception:
        pass

    return order, f"✅ شراء {symbol} | SL: {pos['stop_loss']:.6f} | 💰 {trade_usdt:.2f}$"

# ================== إدارة الصفقة ==================

def manage_position(symbol):
    """
    يدير صفقة مفتوحة: TP/SL/Trailing/Time-based exits.
    يعيد True إذا أُغلِقت الصفقة (كليًا) في هذه الاستدعاء.
    """
    pos = load_position(symbol)
    if not pos: return False

    base = pos["symbol"].split("#")[0]
    current = float(fetch_price(base))
    entry   = float(pos["entry_price"])
    amount  = float(pos["amount"])
    targets = pos.get("targets")
    partials = pos.get("partials")
    variant = pos.get("variant", "new")
    mgmt = _mgmt(variant)

    if amount <= 0:
        clear_position(symbol); return False

    # (1) HTF Stop Rule
    stop_rule = pos.get("htf_stop")
    if stop_rule:
        tf = stop_rule.get("tf")
        tf_map = {"H1": "1h", "H4": "4h", "D1": "1d"}
        tf_fetch = tf_map.get(str(tf).upper(), "4h")
        data_htf = get_ohlcv_cached(base, tf_fetch, 200)
        if data_htf:
            dfh = _df(data_htf)
            closed = dfh.iloc[-2]
            try:
                level = float(stop_rule.get("level", pos["stop_loss"]))
            except Exception:
                level = float(pos["stop_loss"])
            if float(closed["close"]) < level:
                order = place_market_order(base, "sell", amount)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_net = (exit_px - entry) * amount - (entry + exit_px) * amount * (FEE_BPS_ROUNDTRIP/10000.0)
                    close_trade(symbol, exit_px, pnl_net, reason="HTF_STOP")
                    try: _tg(f"🛑 وقف HTF {symbol} عند <code>{exit_px:.6f}</code>")
                    except Exception: pass
                    return True

    # (2) خروج زمني لـ TP1
    max_bars = pos.get("max_bars_to_tp1")
    if max_bars and isinstance(max_bars, int):
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
            bars_passed = int((now_riyadh() - opened_at) // timedelta(minutes=5))
            if bars_passed >= max_bars and not pos["tp_hits"][0]:
                order = place_market_order(base, "sell", amount)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_net = (exit_px - entry) * amount - (entry + exit_px) * amount * (FEE_BPS_ROUNDTRIP/10000.0)
                    close_trade(symbol, exit_px, pnl_net, reason="TIME_EXIT")
                    try: _tg(pos["messages"]["time"] if pos.get("messages") else "⌛ خروج زمني")
                    except Exception: pass
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
                order = place_market_order(base, "sell", amount)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_net = (exit_px - entry) * amount - (entry + exit_px) * amount * (FEE_BPS_ROUNDTRIP/10000.0)
                    close_trade(symbol, exit_px, pnl_net, reason="TIME_HOLD_MAX")
                    try: _tg("⌛ خروج لانتهاء مدة الاحتفاظ")
                    except Exception: pass
                    return True
        except Exception:
            pass

    # (3) إدارة الأهداف + Trailing + قفل ربح بعد TP1
    if targets and partials:
        for i, tp in enumerate(targets):
            if i >= len(partials): break
            if not pos["tp_hits"][i] and current >= tp and amount > 0:
                part_qty = amount * partials[i]
                if part_qty * current < MIN_NOTIONAL_USDT:
                    part_qty = amount

                order = place_market_order(base, "sell", part_qty)
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
                        if pos.get("messages"):
                            _tg(pos["messages"].get(f"tp{i+1}", f"🎯 TP{i+1} تحقق"))
                    except Exception:
                        pass

                    # قفل أرباح بعد TP1
                    try:
                        if i == 0 and pos["amount"] > 0:
                            lock_sl = entry * (1.0 + float(get_cfg(variant).get("LOCK_MIN_PROFIT_PCT", 0.0)))
                            if lock_sl > pos["stop_loss"]:
                                pos["stop_loss"] = float(lock_sl)
                                save_position(symbol, pos)
                                try: _tg(f"🔒 تحريك وقف الخسارة لقفل ربح مبدئي: <code>{lock_sl:.6f}</code>")
                                except Exception: pass
                    except Exception:
                        pass

                    # تريلينغ بعد أي TP ≥ 2
                    if i >= 1 and pos["amount"] > 0:
                        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
                        if data_for_atr:
                            df_atr = _df(data_for_atr); atr_val2 = _atr_from_df(df_atr)
                            if atr_val2 and atr_val2 > 0:
                                new_sl = current - get_cfg(variant).get("TRAIL_ATR_MULT", 1.0) * atr_val2
                                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                                    pos["stop_loss"] = float(new_sl); save_position(symbol, pos)
                                    try: _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
                                    except Exception: pass

    # (3b) تريلينغ عام بعد أي TP
    if mgmt.get("TRAIL_AFTER_TP1") and pos["amount"] > 0 and any(pos.get("tp_hits", [])):
        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
        if data_for_atr:
            df_atr = _df(data_for_atr)
            atr_val = _atr_from_df(df_atr)
            if atr_val and atr_val > 0:
                new_sl = current - mgmt.get("TRAIL_ATR", 1.0) * atr_val
                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                    pos["stop_loss"] = float(new_sl); save_position(symbol, pos)
                    try: _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
                    except Exception: pass

    # (4) وقف الخسارة
    if current <= pos["stop_loss"] and pos["amount"] > 0:
        sellable = float(pos["amount"])
        order = place_market_order(base, "sell", sellable)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sellable
            fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            close_trade(symbol, exit_px, pnl_net, reason="SL")
            try:
                if pos.get("messages"): _tg(pos["messages"].get("sl", "🛑 SL"))
            except Exception:
                pass
            return True

    return False

# ================== إغلاق وتسجيل ==================

def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    """يغلق الصفقة ويدوّنها في closed_positions.json ويحدّث مخاطر اليوم/الساعة."""
    pos = load_position(symbol)
    if not pos: return
    closed = load_closed_positions()

    entry = float(pos["entry_price"]); amount = float(pos["amount"])  # المتبقّي عند الإغلاق (قد يكون جزئيًا)
    pnl_pct = ((exit_price / entry) - 1.0) if entry else 0.0

    tp_hits = {}
    try:
        if "targets" in pos and "tp_hits" in pos:
            for i, hit in enumerate(pos["tp_hits"], start=1):
                tp_hits[f"tp{i}_hit"] = bool(hit)
    except Exception:
        pass

    closed.append({
        "symbol": pos.get("symbol", symbol),
        "entry_price": entry,
        "exit_price": float(exit_price),
        "amount": amount,
        "profit": float(pnl_net),
        "pnl_pct": round(pnl_pct, 6),
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
    register_trade_result(pnl_net)
    clear_position(symbol)

# ================== تقرير يومي ==================

def _fmt_table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r): widths[i] = max(widths[i], len(str(c)))
    def fmt_row(r): return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "<pre>" + fmt_row(headers) + "\n" + "\n".join(fmt_row(r) for r in rows) + "</pre>"

def build_daily_report_text():
    """ينشئ نص تقرير يومي مضغوط (HTML) مع ملخص المخاطر وصفقات اليوم."""
    closed = load_closed_positions(); today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    s = load_risk_state()

    if not todays:
        extra = f"\nوضع المخاطر: {'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'} • صفقات اليوم: {s.get('trades_today',0)} • PnL اليومي: {s.get('daily_pnl',0.0):.2f}$"
        return f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم.{extra}"

    total_pnl = sum(float(t.get("profit", 0.0)) for t in todays)
    wins = [t for t in todays if float(t.get("profit", 0.0)) > 0]
    win_rate = round(100 * len(wins) / max(1, len(todays)), 2)

    headers = ["الرمز#النسخة", "الكمية", "دخول", "خروج", "P/L$", "P/L%", "Score", "نمط", "سبب", "TP_hits", "Exit"]
    rows = []
    for t in todays:
        tp_hits = []
        for i in range(1, 6):
            if t.get(f"tp{i}_hit"): tp_hits.append(f"T{i}")
        tp_str = ",".join(tp_hits) if tp_hits else "-"

        rows.append([
            t.get("symbol","-"),
            f"{float(t.get('amount',0)):, .6f}".replace(' ', ''),
            f"{float(t.get('entry_price',0)):, .6f}".replace(' ', ''),
            f"{float(t.get('exit_price',0)):, .6f}".replace(' ', ''),
            f"{float(t.get('profit',0)):, .2f}".replace(' ', ''),
            f"{round(float(t.get('pnl_pct',0))*100,2)}%",
            str(t.get("score","-")),
            t.get("pattern","-"),
            (t.get("entry_reason", t.get('reason','-'))[:40] + ("…" if len(t.get("entry_reason", t.get('reason','')))>40 else "")),
            tp_str,
            t.get("reason","-")
        ])

    table = _fmt_table(rows, headers)

    risk_line = f"وضع المخاطر: {'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'} • " \
                f"اليومي: <b>{s.get('daily_pnl',0.0):.2f}$</b> • " \
                f"متتالية خسائر: <b>{s.get('consecutive_losses',0)}</b> • " \
                f"صفقات اليوم: <b>{s.get('trades_today',0)}</b>"

    summary = (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b>\n"
        f"{risk_line}\n"
    )
    return summary + table

# ===== اختبار محلي سريع =====
if __name__ == "__main__":
    if DEBUG_LOG_SIGNALS:
        logger.setLevel("INFO")
    test = [s if "#" in s else s+"#new" for s in SYMBOLS[:10]]
    for sym in test:
        r = check_signal(sym)
        print(sym, "→", r)
