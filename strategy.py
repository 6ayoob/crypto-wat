# -*- coding: utf-8 -*-
"""
strategy.py — Spot-only (v3.2a AUTO-BREADTH)
- كاش OHLCV للجولة + مِقاييس أداء.
- Retry/Backoff لاستدعاءات OHLCV.
- Position sizing ديناميكي (نسبة من رأس المال + معامل تقلب ATR + تعزيز حسب Score).
- Circuit breaker بالساعة (اختياري عبر البيئة).
- Auto-Relax (6/12 ساعة) مع رجوع للوضع الطبيعي بعد صفقتين ناجحتين.
- Market Breadth Guard تلقائي: اختيار بين الثابت BREADTH_MIN_RATIO والديناميكي _effective_breadth_min حسب قوة BTC على 4h.
- استثناء القائد (Relative Leader vs BTC) بحجم مخفّض في بيئة سِعة ضعيفة.
- تحجيم حجم الصفقة تدريجيًا حسب السِعة.
- حارس Parabolic/Exhaustion (RSI/المسافة عن EMA50).
- دعم أهداف متعددة (حتى 5) + Partials متكيّفة.
- Dynamic Max Bars to TP1.
- ملخص رفضات دوري كل 30 دقيقة.
- دوال تشخيص اختيارية.

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
from __future__ import annotations

import os, json, requests, logging, time, math
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

# ================== إعدادات عامة ==================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# إشعارات من داخل الاستراتيجية (لمنع التكرار مع main.py)
STRAT_TG_SEND = os.getenv("STRAT_TG_SEND", "0").lower() in ("1","true","yes","y")

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
GOLDEN_CROSS_RVOL_BOOST   = float(os.getenv("GOLDEN_CROSS_RVOL_BOOST", "1.10"))

# ======= Auto-Relax =======
AUTO_RELAX_AFTER_HRS_1 = float(os.getenv("AUTO_RELAX_AFTER_HRS_1", "6"))
AUTO_RELAX_AFTER_HRS_2 = float(os.getenv("AUTO_RELAX_AFTER_HRS_2", "12"))
RELAX_RVOL_DELTA_1 = float(os.getenv("RELAX_RVOL_DELTA_1", "0.05"))
RELAX_RVOL_DELTA_2 = float(os.getenv("RELAX_RVOL_DELTA_2", "0.10"))
RELAX_ATR_MIN_SCALE_1 = float(os.getenv("RELAX_ATR_MIN_SCALE_1", "0.9"))
RELAX_ATR_MIN_SCALE_2 = float(os.getenv("RELAX_ATR_MIN_SCALE_2", "0.85"))
RELAX_RESET_SUCCESS_TRADES = int(os.getenv("RELAX_RESET_SUCCESS_TRADES", "2"))

# ======= Market Breadth =======
BREADTH_MIN_RATIO = float(os.getenv("BREADTH_MIN_RATIO", "0.60"))  # حد أساسي (يُضبط ديناميكيًا)
BREADTH_TF = os.getenv("BREADTH_TF", "1h")
BREADTH_TTL_SEC = int(os.getenv("BREADTH_TTL_SEC", "180"))
BREADTH_SYMBOLS_ENV = os.getenv("BREADTH_SYMBOLS", "")  # CSV اختياري

# Exhaustion
EXH_RSI_MAX = float(os.getenv("EXH_RSI_MAX", "76"))
EXH_EMA50_DIST_ATR = float(os.getenv("EXH_EMA50_DIST_ATR", "2.8"))

# Multi-targets
ENABLE_MULTI_TARGETS = os.getenv("ENABLE_MULTI_TARGETS", "1").lower() in ("1","true","yes","y")
MAX_TP_COUNT = int(os.getenv("MAX_TP_COUNT", "5"))
TP_ATR_MULTS_TREND = tuple(float(x) for x in os.getenv("TP_ATR_MULTS_TREND", "1.2,2.2,3.5,4.5,6.0").split(","))
TP_ATR_MULTS_VBR   = tuple(float(x) for x in os.getenv("TP_ATR_MULTS_VBR",   "0.6,1.2,1.8,2.4").split(","))

# Dynamic Max Bars to TP1
USE_DYNAMIC_MAX_BARS = os.getenv("USE_DYNAMIC_MAX_BARS", "1").lower() in ("1","true","yes","y")
MAX_BARS_BASE = int(os.getenv("MAX_BARS_TO_TP1_BASE", "12"))

# Tunables عبر ENV (تخفيف رفض ATR/RVOL/Notional)
MIN_BAR_NOTIONAL_USD = float(os.getenv("MIN_BAR_NOTIONAL_USD", "25000"))
ATR_MIN_BASE = float(os.getenv("ATR_MIN_FOR_TREND_BASE", "0.0020"))
ATR_MIN_NEW  = float(os.getenv("ATR_MIN_FOR_TREND_NEW",  "0.0026"))
ATR_MIN_BRT  = float(os.getenv("ATR_MIN_FOR_TREND_BRT",  "0.0022"))
RVOL_MIN_NEW = float(os.getenv("RVOL_MIN_NEW", "1.25"))
RVOL_MIN_BRT = float(os.getenv("RVOL_MIN_BRT", "1.30"))

# ======= كاش HTF =======
_HTF_CACHE: Dict[str, Dict[str, Any]] = {}
_HTF_TTL_SEC = int(os.getenv("HTF_CACHE_TTL_SEC", "150"))

# ======= كاش OHLCV للجولة + مِقاييس =======
_OHLCV_CACHE: Dict[tuple, list] = {}  # key=(symbol, tf, bars)
_METRICS = {"ohlcv_api_calls": 0, "ohlcv_cache_hits": 0, "ohlcv_cache_misses": 0, "htf_cache_hits": 0, "htf_cache_misses": 0}

# ======= عدّاد رفضات الجولة (لتليين محلي) + ملخص عام =======
_REJ_COUNTS = {"atr_low": 0, "rvol": 0, "notional_low": 0}
_REJ_SUMMARY: Dict[str, int] = {}

def reset_cycle_cache():
    """يمسح كاش OHLCV ويصفر مِقاييس الجولة + عدّادات الرفض — تُنادى من main.py بداية كل جولة."""
    _OHLCV_CACHE.clear()
    for k in _METRICS: _METRICS[k] = 0
    for k in _REJ_COUNTS: _REJ_COUNTS[k] = 0
    # _REJ_SUMMARY لا نصفره داخل الجولة؛ الملخص كل 30 دقيقة

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
    _METRICS["ohlcv_api_calls"] += 1
    return _retry_fetch_ohlcv(symbol, tf, bars)

def get_ohlcv_cached(symbol: str, tf: str, bars: int) -> list:
    key = (symbol, tf, bars)
    if key in _OHLCV_CACHE:
        _METRICS["ohlcv_cache_hits"] += 1
        return _OHLCV_CACHE[key]
    _METRICS["ohlcv_cache_misses"] += 1
    data = api_fetch_ohlcv(symbol, tf, bars)
    if data: _OHLCV_CACHE[key] = data
    return data

# ================== تتبُّع ==================
_LAST_REJECT: Dict[str, Any] = {}
_LAST_ENTRY_BAR_TS: Dict[str, int] = {}
_SYMBOL_LAST_TRADE_AT: Dict[str, datetime] = {}
_CURRENT_SYMKEY: Optional[str] = None  # لتسجيل آخر رمز/نسخة أثناء الفحص

# ================== حالة/نسخ ==================
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
    "RVOL_MIN": 1.4,
    "ATR_MIN_FOR_TREND": 0.0022,
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

# ✨ حقن قيم ENV اللينة
BASE_CFG["ATR_MIN_FOR_TREND"] = ATR_MIN_BASE
NEW_SCALP_OVERRIDES.update({"ATR_MIN_FOR_TREND": ATR_MIN_NEW, "RVOL_MIN": RVOL_MIN_NEW})
BRT_OVERRIDES.update({"ATR_MIN_FOR_TREND": ATR_MIN_BRT, "RVOL_MIN": RVOL_MIN_BRT})

RSI_MIN_PULLBACK, RSI_MAX_PULLBACK = 45, 65
RSI_MIN_BREAKOUT, RSI_MAX_BREAKOUT = 50, 80

PER_STRAT_MGMT = {
    "new": {"SL":"atr", "SL_MULT":0.9, "TP1":"sr_or_atr", "TP1_ATR":1.2, "TP2_ATR":2.2,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":1.0, "TIME_HRS":6},
    "old": {"SL":"pct", "SL_PCT":0.02, "TP1_PCT":0.03, "TP2_PCT":0.06,
             "TRAIL_AFTER_TP1":False, "TIME_HRS":12},
    "srr": {"SL":"atr_below_sweep", "SL_MULT":0.8, "TP1":"sr_or_atr", "TP1_ATR":1.0, "TP2_ATR":2.2,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":1.0, "TIME_HRS":4},
    "brt": {"SL":"atr_below_retest", "SL_MULT":1.0, "TP1":"range_or_atr", "TP1_ATR":1.5, "TP2_ATR":2.5,
             "TRAIL_AFTER_TP1":True, "TRAIL_ATR":0.9, "TIME_HRS":8},
    "vbr": {"SL":"atr", "SL_MULT":1.0, "TP1":"vwap_or_sr", "TP2_ATR":1.8,
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
    if not STRAT_TG_SEND: return
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
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

# ================== حالة المخاطر اليومية/الساعة + Auto-Relax ==================
def _default_risk_state():
    return {
        "date": _today_str(),
        "daily_pnl": 0.0,
        "consecutive_losses": 0,
        "trades_today": 0,
        "blocked_until": None,
        "hourly_pnl": {},
        "last_signal_ts": None,
        "relax_success_count": 0
    }

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str():
        s = _default_risk_state(); save_risk_state(s)
    if "hourly_pnl" not in s or not isinstance(s["hourly_pnl"], dict):
        s["hourly_pnl"] = {}
    if "last_signal_ts" not in s: s["last_signal_ts"] = None
    if "relax_success_count" not in s: s["relax_success_count"] = 0
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)
def register_trade_opened():
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
    except Exception:
        return None
    return max(0.0, (now_riyadh() - dt).total_seconds() / 3600.0)

def _relax_level_current() -> int:
    """يرجع مستوى التخفيف 0/1/2 حسب الساعات منذ آخر إشارة، مع احترام reset بالنجاحات."""
    s = load_risk_state()
    # لو حقّقنا عدد الصفقات الناجحة المطلوبة، نعود للوضع الطبيعي
    if int(s.get("relax_success_count", 0)) >= RELAX_RESET_SUCCESS_TRADES:
        return 0

    hrs = _hours_since_last_signal()
    if hrs >= AUTO_RELAX_AFTER_HRS_2:
        return 2
    if hrs >= AUTO_RELAX_AFTER_HRS_1:
        return 1
    return 0

def _format_relax_str() -> str:
    """سطر جاهز لعرض حالة Auto-Relax في التقارير."""
    hrs = _hours_since_last_signal()
    # بعض النسخ القديمة كانت ترجع قيمة ضخمة كـ "بدون إشارات" — نتعامل معها هنا
    if hrs is None or hrs > 1e8:
        return "Auto-Relax: لا توجد إشارات بعد."
    if hrs >= 72:
        return f"Auto-Relax: آخر إشارة منذ ~{hrs/24:.1f}d."
    return f"Auto-Relax: آخر إشارة منذ ~{hrs:.1f}h."


def register_trade_result(pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1

    if pnl_usdt > 0:
        s["relax_success_count"] = int(s.get("relax_success_count", 0)) + 1
        if s["relax_success_count"] >= RELAX_RESET_SUCCESS_TRADES:
            s["relax_success_count"] = 0
            s["last_signal_ts"] = now_riyadh().isoformat(timespec="seconds")
            try: _tg("✅ تمت صفقتان ناجحتان — إلغاء تخفيف القيود (عودة للوضع الطبيعي).")
            except Exception: pass
    else:
        s["relax_success_count"] = 0

    hk = _hour_key(now_riyadh())
    s["hourly_pnl"][hk] = float(s["hourly_pnl"].get(hk, 0.0)) + float(pnl_usdt)

    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s); _set_block(90, reason="خسائر متتالية"); return
    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s); _set_block(minutes, reason="تجاوز حد الخسارة اليومي"); return

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

# ===== Swing/SR =====
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

# ===== MACD/RSI Gate =====
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

# ================== Breadth Guard ==================
_BREADTH_CACHE = {"t": 0.0, "ratio": None}

def _breadth_refs() -> List[str]:
    if BREADTH_SYMBOLS_ENV.strip():
        out = []
        for s in BREADTH_SYMBOLS_ENV.split(","):
            s = s.strip()
            if s:
                out.append(s.replace("-", "/").upper().split("#")[0])
        return out
    uniq = []
    seen = set()
    for s in SYMBOLS:
        base = s.split("#")[0].replace("-", "/").upper()
        if base not in seen:
            uniq.append(base); seen.add(base)
        if len(uniq) >= 12: break
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
            if not d or len(d) < 60: continue
            df = _df(d)
            df["ema50"] = ema(df["close"], 50)
            row = df.iloc[-2]
            c = float(row["close"]); e = float(row["ema50"])
            if c>0 and e>0:
                tot += 1
                if c > e: ok += 1
        except Exception:
            continue
    if tot == 0: return None
    return ok / float(tot)

def _get_breadth_ratio_cached() -> Optional[float]:
    now_s = time.time()
    if _BREADTH_CACHE["ratio"] is not None and (now_s - _BREADTH_CACHE["t"]) <= BREADTH_TTL_SEC:
        return _BREADTH_CACHE["ratio"]
    r = _compute_breadth_ratio()
    _BREADTH_CACHE["ratio"] = r
    _BREADTH_CACHE["t"] = now_s
    return r

# ===== سِعة ديناميكية
def _effective_breadth_min() -> float:
    """يضبط الحد الأدنى المطلوب للسعة بناءً على حالة BTC/USDT على 4h."""
    base = BREADTH_MIN_RATIO
    try:
        d = get_ohlcv_cached("BTC/USDT", "4h", 220)
        if not d or len(d) < 100: 
            return base
        df = _df(d); df["ema50"] = ema(df["close"], 50)
        row = df.iloc[-2]
        above = float(row["close"]) > float(row["ema50"])
        rsi_btc = float(rsi(df["close"], 14).iloc[-2])
        if above and rsi_btc >= 55:  return max(0.40, base - 0.15)  # مرونة للأسفل
        if (not above) or rsi_btc <= 45: return min(0.75, base + 0.10)  # تشديد للأعلى
    except Exception:
        pass
    return base

# ===== اختيار تلقائي بين الثابت والديناميكي (جديد) =====
def _btc_strong_on_4h() -> bool:
    """يُرجع True إذا كان BTC/USDT قوي على 4h: فوق EMA50 و RSI≥55."""
    try:
        d = get_ohlcv_cached("BTC/USDT", "4h", 220)
        if not d or len(d) < 100:
            return False
        df = _df(d)
        df["ema50"] = ema(df["close"], 50)
        rsi_btc = float(rsi(df["close"], 14).iloc[-2])
        row = df.iloc[-2]
        above = float(row["close"]) > float(row["ema50"])
        return bool(above and rsi_btc >= 55)
    except Exception:
        return False

def _breadth_min_auto() -> float:
    """
    اختيار تلقائي بين الثابت والديناميكي:
    - إذا BTC قوي على 4h → ديناميكي (_effective_breadth_min)
    - غير ذلك → ثابت (BREADTH_MIN_RATIO)
    """
    try:
        return _effective_breadth_min() if _btc_strong_on_4h() else BREADTH_MIN_RATIO
    except Exception:
        return BREADTH_MIN_RATIO

# ===== سلوك القائد =====
def _is_relative_leader_vs_btc(symbol_base: str, tf="1h", lookback=24, edge=0.02) -> bool:
    """يقيس متوسط التفوق النسبي لعوائد الرمز - عوائد BTC خلال نافذة بسيطة."""
    try:
        d1 = get_ohlcv_cached(symbol_base, tf, lookback+10)
        d2 = get_ohlcv_cached("BTC/USDT", tf, lookback+10)
        if not d1 or not d2: return False
        s1 = _df(d1)["close"].iloc[-(lookback+1):-1]
        s2 = _df(d2)["close"].iloc[-(lookback+1):-1]
        if len(s1)!=len(s2):
            n=min(len(s1),len(s2)); s1=s1.iloc[-n:]; s2=s2.iloc[-n:]
        rel = (s1.pct_change().fillna(0) - s2.pct_change().fillna(0)).mean()
        return float(rel) >= edge
    except Exception:
        return False

# ================== أدوات رفض/تمرير ==================
def _rej(stage, **kv):
    # عدّاد التليين المحلي
    if stage in _REJ_COUNTS:
        _REJ_COUNTS[stage] += 1
    # ملخص شامل لأي سبب
    try:
        _REJ_SUMMARY[stage] = int(_REJ_SUMMARY.get(stage, 0)) + 1
    except Exception:
        pass
    # خزّن آخر سبب رفض لهذا الرمز/النسخة إن كان محددًا
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
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"[REJECT] {stage} | {kvs}")
    return None

def _pass(stage, **kv):
    if DEBUG_LOG_SIGNALS:
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"[PASS]   {stage} | {kvs}")

def _round_relax_factors():
    """تليين خفيف داخل الجولة بناءً على كثافة الرفض الفوري."""
    f_atr, f_rvol = 1.0, 1.0
    notional_min = MIN_BAR_NOTIONAL_USD
    c = _REJ_COUNTS
    if c["atr_low"] >= 10: f_atr = 0.90
    if c["atr_low"] >= 30: f_atr = 0.80
    if c["rvol"]    >= 10: f_rvol = 0.95
    if c["rvol"]    >= 30: f_rvol = 0.90
    if c["notional_low"] >= 10: notional_min *= 0.80
    return f_atr, f_rvol, notional_min

# ================== منطق الدخول الأساسية ==================
def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    ref_val = closed["ema21"] if cfg["PULLBACK_VALUE_REF"]=="ema21" else closed.get("vwap", closed["ema21"])
    if pd.isna(ref_val): ref_val = closed["ema21"]
    near_val = (closed["close"] >= ref_val) and (closed["low"] <= ref_val)
    if not near_val: return False
    if cfg["PULLBACK_CONFIRM"] == "bullish_engulf":
        return _bullish_engulf(prev, closed)
    elif cfg["PULLBACK_CONFIRM"] == "bos":
        swing_high, _ = _swing_points(df); return bool(swing_high and closed["close"] > swing_high)
    return True

def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx, cfg):
    hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
    is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
    vwap_ok = closed["close"] > float(closed.get("vwap", closed["ema21"]))
    return (closed["close"] > hi_range) and (is_nr_recent and vwap_ok or vwap_ok)

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

# ================== NEW/SRR — مع السِعة التلقائية + قائد ==================
def check_signal_new(symbol):
    """يفحص إشارة شراء Spot فقط على الرمز المحدد (نسخ: new/srr/brt/vbr). يعيد dict{'decision':'buy', ...} أو None."""
    ok, reason = _risk_precheck_allow_new_entry()
    if not ok: return _rej("risk_precheck", reason=reason)

    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)
    key = f"{base}|{variant}"

    # لتسجيل آخر سبب رفض لهذا الرمز/النسخة
    global _CURRENT_SYMKEY
    _CURRENT_SYMKEY = key

    try:
        last_t = _SYMBOL_LAST_TRADE_AT.get(key)
        if last_t and (now_riyadh() - last_t) < timedelta(minutes=cfg["SYMBOL_COOLDOWN_MIN"]):
            return _rej("cooldown")
        if load_position(symbol):
            return _rej("already_open")

        # Market breadth (تلقائي)
        br = _get_breadth_ratio_cached()
        eff_min = _breadth_min_auto()   # ← استخدام الحدّ التلقائي
        leader_flag = False
        if br is not None and br < eff_min:
            # استثناء القائد
            leader_flag = _is_relative_leader_vs_btc(base)
            if not leader_flag:
                return _rej("breadth_low", ratio=round(br,2), min=round(eff_min,2))

        ctx = _get_htf_context(symbol)
        if not ctx: return _rej("htf_none")
        if not ((ctx["ema50_now"] - ctx["ema50_prev"]) > 0 and ctx["close"] > ctx["ema50_now"]):
            return _rej("htf_trend")

        data = get_ohlcv_cached(base, LTF_TIMEFRAME, 260)
        if not data: return _rej("ltf_fetch")
        df = _df(data); df = _ensure_ltf_indicators(df)
        if len(df) < 200: return _rej("ltf_len", n=len(df))

        prev, closed = df.iloc[-3], df.iloc[-2]
        ts = int(closed["timestamp"])
        if _LAST_ENTRY_BAR_TS.get(key) == ts: return _rej("dup_bar")

        price = float(closed["close"])
        atr = _atr_from_df(df)
        if not atr or atr <= 0: return _rej("atr_nan")
        atrp = atr / max(price, 1e-9)

        # تليين الجولة + Auto-Relax بالساعات
        f_atr, f_rvol, notional_min = _round_relax_factors()
        relax_lvl = _relax_level_current()

        atr_min = float(cfg.get("ATR_MIN_FOR_TREND", 0.002))
        if relax_lvl == 1: atr_min *= RELAX_ATR_MIN_SCALE_1
        if relax_lvl == 2: atr_min *= RELAX_ATR_MIN_SCALE_2
        atr_min *= f_atr
        if atrp < atr_min: return _rej("atr_low", atrp=round(atrp,5), min=round(atr_min,5))

        notional = price * float(closed["volume"])
        if notional < notional_min:
            return _rej("notional_low", notional=int(notional))

        rvol = float(closed.get("rvol", 0) or 0)
        need_rvol_base = float(cfg.get("RVOL_MIN", 1.2)) * 0.95
        need_rvol = need_rvol_base
        if relax_lvl == 1: need_rvol = max(0.8, need_rvol_base - RELAX_RVOL_DELTA_1)
        if relax_lvl == 2: need_rvol = max(0.75, need_rvol_base - RELAX_RVOL_DELTA_2)
        need_rvol = max(0.70, need_rvol * f_rvol)
        if pd.isna(rvol) or rvol < need_rvol: return _rej("rvol", rvol=round(rvol,2), need=round(need_rvol,2))

        # فلتر ترند EMA200 (اختياري)
        if USE_EMA200_TREND_FILTER:
            if not (float(closed.get("ema50", price)) > float(closed.get("ema200", price)) and price > float(closed.get("ema200", price))):
                return _rej("ema200_trend")

        # اختيار النمط (hybrid)
        def _brk_ok():
            hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
            is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
            vwap_ok = price > float(closed.get("vwap", closed.get("ema21", price)))
            buf = float(cfg.get("BREAKOUT_BUFFER_LTF", 0.0015))
            return (price > hi_range * (1.0 + buf)) and (is_nr_recent or vwap_ok)

        chosen_mode = None; mode_ok = False
        entry_mode = cfg.get("ENTRY_MODE", "hybrid")
        if entry_mode == "pullback":
            chosen_mode = "pullback"; mode_ok = _entry_pullback_logic(df, closed, prev, atr, ctx, cfg)
        elif entry_mode == "breakout":
            chosen_mode = "breakout"; mode_ok = _brk_ok()
        else:
            for m in cfg.get("HYBRID_ORDER", ["breakout","pullback"]):
                if m == "breakout" and _brk_ok():
                    chosen_mode = "breakout"; mode_ok = True; break
                if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr, ctx, cfg):
                    chosen_mode = "pullback"; mode_ok = True; break

        # بديل: Golden Cross
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
        if not (lb <= dist_atr <= ub):
            return _rej("dist_to_ema50", dist_atr=round(dist_atr,3), lb=lb, ub=ub)

        # Exhaustion guard
        rsi_val = float(closed.get("rsi", 50))
        if (chosen_mode in ("breakout","pullback","golden_cross")) and (rsi_val >= EXH_RSI_MAX) and (dist_atr >= EXH_EMA50_DIST_ATR):
            return _rej("exhaustion_guard", rsi=rsi_val, dist_atr=round(dist_atr,2))

        # نطاقات RSI حسب النمط
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
        ctx_res_near = (ctx.get("resistance") and (ctx["resistance"] - price) < 1.2*atr)
        near_res = (res_ltf and (res_ltf - price) < 0.8*atr) or ctx_res_near or near_res_any
        if near_res and chosen_mode != "breakout":
            return _rej("near_res_block")

        score, why, patt = _opportunity_score(df, prev, closed)
        if chosen_mode == "breakout": patt = "NR_Breakout"
        if chosen_mode == "golden_cross": patt = "EMA50x200_Golden"; score += 12

        if score < SCORE_THRESHOLD:
            return _rej("score_low", score=score)

        _LAST_ENTRY_BAR_TS[key] = ts
        _pass("signal_ok", mode=chosen_mode, score=score, rvol=round(rvol,2), atrp=round(atrp,4))
        _mark_signal_now()

        return {
            "decision": "buy",
            "score": score, "reason": why, "pattern": patt, "ts": ts,
            "chosen_mode": chosen_mode, "atrp": atrp, "rsi": rsi_val, "dist_ema50_atr": dist_atr,
            "leader_flag": bool(leader_flag)
        }
    finally:
        _CURRENT_SYMKEY = None

# ================== OLD/SRR/BRT/VBR ==================
def check_signal_old(symbol): return check_signal_new(symbol)
def check_signal_brt(symbol): return check_signal_new(symbol)
def check_signal_vbr(symbol): return check_signal_new(symbol)

# ================== Router ==================
def check_signal(symbol):
    base, variant = _split_symbol_variant(symbol)
    if variant == "old": return check_signal_old(symbol)
    if variant == "srr": return check_signal_new(symbol)
    if variant == "brt": return check_signal_brt(symbol)
    if variant == "vbr": return check_signal_vbr(symbol)
    return check_signal_new(symbol)

# ===== آخر سبب رفض لرمز/نسخة (اختياري للتشخيص) =====
def get_last_reject(symbol: str) -> Optional[Dict[str, Any]]:
    base, variant = _split_symbol_variant(symbol)
    return _LAST_REJECT.get(f"{base}|{variant}")

# ================== SL/TP ==================
def _compute_sl_tp(entry, atr_val, cfg, variant, symbol=None, df=None, ctx=None, closed=None):
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

    vwap_val = None; nearest_res = None
    try:
        sr_multi = get_sr_multi(symbol) if symbol else {}
        for name, ent in sr_multi.items():
            res = ent.get("resistance")
            if res and res > entry:
                nearest_res = res if nearest_res is None else min(nearest_res, res)
    except Exception:
        pass
    try:
        if closed is None and df is not None:
            closed = df.iloc[-2]
        vwap_val = float(closed.get("vwap")) if closed is not None else None
    except Exception:
        vwap_val = None

    mg_tp1 = float(mg.get("TP1_ATR", 1.2))
    mg_tp2 = float(mg.get("TP2_ATR", 2.2)) if mg.get("TP2_ATR") else 2.2
    atr_tp1 = entry + mg_tp1 * atr_val
    atr_tp2 = entry + mg_tp2 * atr_val

    mode = mg.get("TP1")
    try:
        if mode == "sr_or_atr":
            sr_tp = nearest_res if (nearest_res and nearest_res > entry) else None
            tp1 = float(min(sr_tp, atr_tp1)) if sr_tp else float(atr_tp1)
        elif mode == "range_or_atr":
            sr_tp = None
            try:
                if df is not None and len(df) > 20:
                    hhv = float(df.iloc[:-1]["high"].rolling(SR_WINDOW, min_periods=10).max().iloc[-1])
                    sr_tp = hhv if hhv > entry else None
            except Exception:
                sr_tp = None
            sr_tp = nearest_res if (nearest_res and nearest_res > entry) else sr_tp
            tp1 = float(min(sr_tp, atr_tp1)) if sr_tp else float(atr_tp1)
        elif mode == "vwap_or_sr":
            candidates = []
            if vwap_val and vwap_val > entry: candidates.append(float(vwap_val))
            if nearest_res and nearest_res > entry: candidates.append(float(nearest_res))
            tp1 = float(min(candidates)) if candidates else float(atr_tp1)
        else:
            if mg.get("TP1_PCT"):
                tp1 = entry * (1 + float(mg.get("TP1_PCT")))
            else:
                if cfg.get("USE_ATR_SL_TP") and atr_val and atr_val > 0:
                    tp1 = entry + cfg.get("TP1_ATR_MULT", 1.6) * atr_val
                    atr_tp2 = entry + cfg.get("TP2_ATR_MULT", 3.2) * atr_val
                else:
                    tp1 = entry * (1 + cfg.get("TP1_PCT", 0.03))
                    atr_tp2 = entry * (1 + cfg.get("TP2_PCT", 0.06))
    except Exception:
        tp1 = atr_tp1

    tp2 = atr_tp2
    return float(sl), float(tp1), float(tp2)

def _build_extra_targets(entry: float, atr_val: float, variant: str, first_two: Tuple[float, float]) -> List[float]:
    if not ENABLE_MULTI_TARGETS:
        return [first_two[0], first_two[1]]
    mults = TP_ATR_MULTS_VBR if variant == "vbr" else TP_ATR_MULTS_TREND
    base_list = [first_two[0], first_two[1]]
    for m in mults:
        t = entry + m * atr_val
        base_list.append(float(t))
    uniq = []
    for x in sorted(set(base_list)):
        if x > entry:
            uniq.append(x)
    if len(uniq) > MAX_TP_COUNT:
        uniq = uniq[:MAX_TP_COUNT]
    return uniq

def _partials_for(score: int, n: int, atrp: float) -> List[float]:
    if n <= 1:
        return [1.0]
    if score >= 88:
        base = [0.30, 0.25, 0.20, 0.15, 0.10]
    elif score >= 80:
        base = [0.35, 0.25, 0.20, 0.12, 0.08]
    else:
        base = [0.40, 0.25, 0.18, 0.10, 0.07]
    if atrp >= 0.02:
        base = [max(0.25, base[0]-0.05)] + base[1:]
    return base[:n]

# ================== تنفيذ الشراء ==================
def execute_buy(symbol):
    """
    تنفيذ شراء Spot-only للرمز المحدّد.
    - يستخدم Position sizing ديناميكي إذا USE_DYNAMIC_RISK=1 (افتراضي) + تعزيز حسب Score.
    - يخضع لمنطق الحظر/المخاطر والسِعة.
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

    sl, tp1, tp2 = _compute_sl_tp(price_fallback, atr_val, cfg, variant, symbol=symbol, df=df_exec, ctx=ctx, closed=closed)
    targets = _build_extra_targets(price_fallback, atr_val, variant, (tp1, tp2))

    mg = _mgmt(variant)
    custom = (_sig_inner.get("custom") if isinstance(_sig_inner, dict) else {}) or {}
    if "sl" in custom and isinstance(custom["sl"], (int, float)): sl = float(custom["sl"])
    if "tp1" in custom and isinstance(custom["tp1"], (int, float)): targets[0] = min(targets[0], float(custom["tp1"]))

    max_bars_to_tp1 = MAX_BARS_BASE
    try:
        atrp = float(_sig_inner.get("atrp", atr_val / max(price_fallback, 1e-9)))
    except Exception:
        atrp = atr_val / max(price_fallback, 1e-9)
    if USE_DYNAMIC_MAX_BARS:
        if atrp <= 0.008:   max_bars_to_tp1 = max(10, MAX_BARS_BASE)
        elif atrp >= 0.020: max_bars_to_tp1 = min(6, MAX_BARS_BASE)
        else:                max_bars_to_tp1 = 8
        if _sig_inner.get("chosen_mode") in ("breakout", "golden_cross"):
            max_bars_to_tp1 = max(6, max_bars_to_tp1 - 2)

    sig = {
        "entry": price_fallback, "sl": sl, "targets": targets, "partials": [],
        "messages": {"entry": f"🚀 دخول {_sig_inner.get('pattern','Opportunity')}"},
        "score": _sig_inner.get("score"), "pattern": _sig_inner.get("pattern"), "reason": _sig_inner.get("reason"),
        "max_hold_hours": mg.get("TIME_HRS"), "max_bars_to_tp1": max_bars_to_tp1
    }

    price = float(sig["entry"]) if isinstance(sig, dict) else None
    usdt = float(fetch_balance("USDT") or 0)

    USE_DYNAMIC_RISK = os.getenv("USE_DYNAMIC_RISK", "1").lower() in ("1","true","yes","y")
    RISK_PCT_OF_EQUITY = float(os.getenv("RISK_PCT_OF_EQUITY", "0.02"))
    MIN_TRADE_USDT = float(os.getenv("MIN_TRADE_USDT", "10"))
    MAX_TRADE_USDT = float(os.getenv("MAX_TRADE_USDT", "1200"))
    ATR_RISK_SCALER = float(os.getenv("ATR_RISK_SCALER", "2.0"))

    if USE_DYNAMIC_RISK:
        equity = usdt
        base_risk = equity * RISK_PCT_OF_EQUITY
        risk_usdt = min(MAX_TRADE_USDT, max(MIN_TRADE_USDT, base_risk))
        atrp_now = (atr_val / max(price, 1e-9)) if atr_val else 0.0
        vol_factor = 1.0 / (1.0 + ATR_RISK_SCALER * max(0.0, atrp_now))
        sc = int(_sig_inner.get("score", 0))
        score_boost = 1.0
        if sc >= 88: score_boost = 1.25
        elif sc >= 80: score_boost = 1.10
        trade_usdt = max(MIN_TRADE_USDT, min(MAX_TRADE_USDT, risk_usdt * vol_factor * score_boost))
    else:
        trade_usdt = TRADE_AMOUNT_USDT

    # تحجيم بالحِسبان السِعة + استثناء القائد
    br = _get_breadth_ratio_cached()
    if br is not None:
        if br < 0.45:  trade_usdt *= 0.70
        elif br < 0.55: trade_usdt *= 0.85
    if bool(_sig_inner.get("leader_flag", False)):
        trade_usdt *= 0.50  # نصف الحجم عند الدخول باستثناء القائد

    if usdt < max(MIN_TRADE_USDT, trade_usdt):
        return None, "🚫 رصيد USDT غير كافٍ."

    amount = trade_usdt / price
    if amount * price < MIN_NOTIONAL_USDT:
        return None, "🚫 قيمة الصفقة أقل من الحد الأدنى."

    order = place_market_order(base, "buy", amount)
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    fill_px = float(order.get("average") or order.get("price") or price)

    sig["partials"] = _partials_for(int(sig["score"] or 0), len(sig["targets"]), atrp)
    ssum = sum(sig["partials"])
    if ssum <= 0: sig["partials"] = [1.0] + [0.0]*(len(sig["targets"])-1)
    else: sig["partials"] = [round(x/ssum, 6) for x in sig["partials"]][:len(sig["targets"])]

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(fill_px),
        "stop_loss": float(sig["sl"]),
        "targets": sig["targets"],
        "partials": sig["partials"],
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": variant,
        "htf_stop": sig.get("stop_rule"),
        "max_bars_to_tp1": sig.get("max_bars_to_tp1"),
        "messages": sig.get("messages"),
        "tp_hits": [False] * len(sig["targets"]),
        "score": sig.get("score"),
        "pattern": sig.get("pattern"),
        "reason": sig.get("reason"),
    }
    save_position(symbol, pos)
    register_trade_opened()
    _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()

    try:
        if STRAT_TG_SEND:
            _tg(f"{pos['messages'].get('entry','✅ دخول')}\n"
                f"دخول: <code>{fill_px:.6f}</code>\n"
                f"SL: <code>{pos['stop_loss']:.6f}</code>\n"
                f"🎯 الأهداف: {', '.join(str(round(t,6)) for t in pos['targets'])}\n"
                f"💰 حجم الصفقة: <b>{trade_usdt:.2f}$</b>")
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
                    try: 
                        if STRAT_TG_SEND: _tg(f"🛑 وقف HTF {symbol} عند <code>{exit_px:.6f}</code>")
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
                    try: 
                        if STRAT_TG_SEND: _tg(pos["messages"]["time"] if pos.get("messages") else "⌛ خروج زمني")
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
                    try: 
                        if STRAT_TG_SEND: _tg("⌛ خروج لانتهاء مدة الاحتفاظ")
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
                    part_qty = amount  # إذا كان صغير جداً، اخرج بالكامل

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
                        if STRAT_TG_SEND and pos.get("messages"):
                            _tg(pos["messages"].get(f"tp{i+1}", f"🎯 TP{i+1} تحقق"))
                    except Exception:
                        pass

                    # قفل أرباح بعد TP1
                    try:
                        cfg = get_cfg(variant)
                        if i == 0 and pos["amount"] > 0:
                            lock_sl = entry * (1.0 + float(cfg.get("LOCK_MIN_PROFIT_PCT", 0.0)))
                            if lock_sl > pos["stop_loss"]:
                                pos["stop_loss"] = float(lock_sl)
                                save_position(symbol, pos)
                                try: 
                                    if STRAT_TG_SEND: _tg(f"🔒 تحريك وقف الخسارة لقفل ربح مبدئي: <code>{lock_sl:.6f}</code>")
                                except Exception: pass
                    except Exception:
                        pass

                    # تريلينغ بعد TP2
                    if i >= 1 and pos["amount"] > 0:
                        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
                        if data_for_atr:
                            df_atr = _df(data_for_atr); atr_val2 = _atr_from_df(df_atr)
                            if atr_val2 and atr_val2 > 0:
                                new_sl = current - atr_val2
                                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                                    pos["stop_loss"] = float(new_sl); save_position(symbol, pos)
                                    try: 
                                        if STRAT_TG_SEND: _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
                                    except Exception: pass

    # (3b) تريلينغ عام بعد أي TP
    if _mgmt(variant).get("TRAIL_AFTER_TP1") and pos["amount"] > 0 and any(pos.get("tp_hits", [])):
        data_for_atr = get_ohlcv_cached(base, LTF_TIMEFRAME, 140)
        if data_for_atr:
            df_atr = _df(data_for_atr)
            atr_val3 = _atr_from_df(df_atr)
            if atr_val3 and atr_val3 > 0:
                new_sl = current - _mgmt(variant).get("TRAIL_ATR", 1.0) * atr_val3
                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                    pos["stop_loss"] = float(new_sl); save_position(symbol, pos)
                    try: 
                        if STRAT_TG_SEND: _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
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
                if STRAT_TG_SEND and pos.get("messages"): _tg(pos["messages"].get("sl", "🛑 SL"))
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

    entry = float(pos["entry_price"]); amount = float(pos["amount"])
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
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))

    header_line = fmt_row(headers)
    body_lines = "\n".join(fmt_row(r) for r in rows)
    return "<pre>" + header_line + "\n" + body_lines + "</pre>"

def build_daily_report_text():
    """ينشئ نص تقرير يومي مضغوط (HTML) مع ملخص المخاطر وصفقات اليوم)."""
    closed = load_closed_positions()
    today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    s = load_risk_state()

    if not todays:
        hrs = _hours_since_last_signal()
        extra = (
            f"\nوضع المخاطر: "
            f"{'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'}"
            f" • صفقات اليوم: {s.get('trades_today', 0)}"
            f" • PnL اليومي: {float(s.get('daily_pnl', 0.0)):.2f}$"
        )
        return f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم.{extra}\nAuto-Relax: آخر إشارة منذ ~{(hrs or 0):.1f}h."

    total_pnl = sum(float(t.get("profit", 0.0)) for t in todays)
    wins = [t for t in todays if float(t.get("profit", 0.0)) > 0]
    win_rate = round(100 * len(wins) / max(1, len(todays)), 2)

    headers = ["الرمز#النسخة", "الكمية", "دخول", "خروج", "P/L$", "P/L%", "Score", "نمط", "سبب", "TP_hits", "Exit"]
    rows = []
    for t in todays:
        tp_hits = []
        for i in range(1, 8):
            if t.get(f"tp{i}_hit"):
                tp_hits.append(f"T{i}")
        tp_str = ",".join(tp_hits) if tp_hits else "-"

        rows.append([
            t.get("symbol", "-"),
            f"{float(t.get('amount', 0)):, .6f}".replace(' ', ''),
            f"{float(t.get('entry_price', 0)):, .6f}".replace(' ', ''),
            f"{float(t.get('exit_price', 0)):, .6f}".replace(' ', ''),
            f"{float(t.get('profit', 0)):, .2f}".replace(' ', ''),
            f"{round(float(t.get('pnl_pct', 0)) * 100, 2)}%",
            str(t.get("score", "-")),
            t.get("pattern", "-"),
            (t.get("entry_reason", t.get("reason", "-"))[:40] +
             ("…" if len(t.get("entry_reason", t.get("reason", ""))) > 40 else "")),
            tp_str,
            t.get("reason", "-")
        ])

    table = _fmt_table(rows, headers)

    risk_line = (
        f"وضع المخاطر: "
        f"{'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'}"
        f" • اليومي: <b>{float(s.get('daily_pnl', 0.0)):.2f}$</b>"
        f" • متتالية خسائر: <b>{int(s.get('consecutive_losses', 0))}</b>"
        f" • صفقات اليوم: <b>{int(s.get('trades_today', 0))}</b>"
    )

    hrs = _hours_since_last_signal()
    summary = (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b> • Auto-Relax منذ آخر إشارة: ~<b>{(hrs or 0):.1f}h</b>\n"
        f"{risk_line}\n"
    )
    return summary + table

# ===== دوال تشخيص/ملخّص رفض =====
_last_emit_ts = 0
def maybe_emit_reject_summary():
    """يرسل ملخصًا لأكثر أسباب الرفض خلال آخر 30 دقيقة + حالة السِعة والمرونة."""
    global _last_emit_ts
    now = time.time()
    if now - _last_emit_ts < 1800:  # كل 30 دقيقة
        return
    _last_emit_ts = now
    try:
        top = sorted(_REJ_SUMMARY.items(), key=lambda x: x[1], reverse=True)[:5]
        items = [f"{k}:{v}" for k, v in top]
        top_line = " | ".join(items) if items else "No data"

        br = _BREADTH_CACHE.get("ratio")
        br_txt = "—" if br is None else f"{br:.2f}"
        eff_min = _breadth_min_auto()
        eff_min_txt = f"{eff_min:.2f}"

        f_atr, f_rvol, notional_min = _round_relax_factors()
        msg = (
            "🧪 <b>Reject Summary (30m)</b>\n"
            f" • {top_line}\n"
            f" • breadth={br_txt} (eff_min≈{eff_min_txt})\n"
            f" • soften: ATR×{f_atr:.2f}, RVOL×{f_rvol:.2f}, Notional≥{int(notional_min)}"
        )
        _tg(msg)
    except Exception:
        pass

def check_signal_debug(symbol: str):
    """إرجاع نتيجة الإشارة + سبب الرفض الأخير إن وُجد."""
    r = check_signal(symbol)
    reasons = []
    if r is None:
        last = get_last_reject(symbol)
        if last:
            reasons = [f"{last.get('stage')}:{last.get('details',{})}"]
        else:
            reasons = ["no_buy"]
    elif isinstance(r, dict) and r.get("decision") == "buy":
        reasons = ["buy_ok"]
    else:
        reasons = ["other"]
    return r, reasons

def breadth_status():
    try:
        r = _get_breadth_ratio_cached()
        eff_min = _breadth_min_auto()
        if r is None:
            return {"ok": True, "ratio": None, "min": eff_min}
        return {"ok": r >= eff_min, "ratio": r, "min": eff_min}
    except Exception:
        return {"ok": True, "ratio": None, "min": BREADTH_MIN_RATIO}
