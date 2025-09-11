# -*- coding: utf-8 -*-
"""
strategy.py — ثلاث استراتيجيات مجرَّبة + تحسينات:
- #old: Hybrid كلاسيكي مع فلترة SR محسّنة
- #new: Scalp متكّف بالـ ATR + Regime-aware + TP1 ذكي
- #srr: Sweep & Reclaim (كسر سيولة واستعادة المستوى)
- #brt: Break & Retest (اختراق + إعادة اختبار)
- #vbr: VWAP-Band Reversion (ارتداد من نطاق VWAP)
- #sr : SR-only (دعوم/مقاومات + RSI فقط)

الملف يوفّر:
- check_signal(symbol) — Router يُعيد None أو dict يحوي decision/score/…
- execute_buy(symbol), manage_position(symbol), close_trade(...), build_daily_report_text()

يعتمد على okx_api (fetch_ohlcv, fetch_price, place_market_order, fetch_balance)
وعلى config (TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
"""

import os, json, requests, logging
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

# ================== إعدادات عامة (أساس) ==================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# أطر زمنية (من config)
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME   # إطار الاتجاه (سياق)
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME   # إطار التنفيذ (سكالب)

# مؤشرات أساسية و نوافذ ثابتة
EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 50
VOL_MA, SR_WINDOW = 20, 50
ATR_PERIOD = 14
RVOL_WINDOW = 20
NR_WINDOW = 10
NR_FACTOR = 0.75
HTF_EMA_TREND_PERIOD = 50
HTF_SR_WINDOW = 50
RESISTANCE_BUFFER, SUPPORT_BUFFER = 0.005, 0.002

# إدارة الصفقة: ثوابت عامة
TP1_FRACTION = 0.5
MIN_NOTIONAL_USDT = 10.0
TRAIL_MIN_STEP_RATIO = 0.001

# مخاطر عامة
MAX_TRADES_PER_DAY       = 20
MAX_CONSEC_LOSSES        = 3
DAILY_LOSS_LIMIT_USDT    = 200.0

# تتبُّع
_LAST_REJECT: Dict[str, Any] = {}
_LAST_ENTRY_BAR_TS: Dict[str, int] = {}      # key: f"{base}|{variant}"
_SYMBOL_LAST_TRADE_AT: Dict[str, datetime] = {}   # key: f"{base}|{variant}"

# كاش بسيط لسياق HTF
_HTF_CACHE: Dict[str, Dict[str, Any]] = {}          # key = base symbol, val={"t": datetime, "ctx": {...}}
_HTF_TTL_SEC = 150       # ~ دقيقتين ونصف

# ================== إعدادات النسخ + Overrides ==================
# قاعدة (#old)
BASE_CFG = {
    # نمط الدخول
    "ENTRY_MODE": "hybrid",                # "pullback" | "breakout" | "hybrid"
    "HYBRID_ORDER": ["pullback","breakout"],
    "PULLBACK_VALUE_REF": "ema21",        # "ema21" | "vwap"
    "PULLBACK_CONFIRM": "bullish_engulf", # "bullish_engulf" | "bos"

    # فلاتر جودة LTF
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.002,

    # بدائل محسّنة (تُستخدم عادةً في #new)
    "USE_FIB": False,
    "SWING_LOOKBACK": 60,
    "FIB_TOL": 0.004,
    "BREAKOUT_BUFFER_LTF": 0.0015,
    "RSI_GATE_POLICY": None,  # None=بدون بوابة

    # إدارة الصفقة (نِسَب ثابتة)
    "USE_ATR_SL_TP": False,
    "STOP_LOSS_PCT": 0.02,
    "TP1_PCT": 0.03,
    "TP2_PCT": 0.06,

    # تريلينغ/وقت/تبريد
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.01,  # 1%
    "MAX_HOLD_HOURS": 12,
    "SYMBOL_COOLDOWN_MIN": 30,
}

# تخصيص (#new) — سكالب متكّيف بالـ ATR
NEW_SCALP_OVERRIDES = {
    "HYBRID_ORDER": ["breakout","pullback"],
    "PULLBACK_VALUE_REF": "vwap",
    "PULLBACK_CONFIRM": "bos",

    "RVOL_MIN": 1.4,
    "ATR_MIN_FOR_TREND": 0.003,

    "USE_FIB": True,
    "BREAKOUT_BUFFER_LTF": 0.0018,
    "RSI_GATE_POLICY": "lenient",

    # إدارة عبر ATR
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,

    # تريلينغ/وقت/تبريد
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.003,
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 8,
}

# SRR — Sweep & Reclaim
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

# BRT — Break & Retest
BRT_OVERRIDES = {
    "ENTRY_MODE": "breakout",
    "RVOL_MIN": 1.4,
    "ATR_MIN_FOR_TREND": 0.0022,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True, "SL_ATR_MULT": 0.9, "TP1_ATR_MULT": 1.4, "TP2_ATR_MULT": 2.4,
    "TRAIL_AFTER_TP1": True, "TRAIL_ATR_MULT": 1.1,
    "LOCK_MIN_PROFIT_PCT": 0.004, "MAX_HOLD_HOURS": 8, "SYMBOL_COOLDOWN_MIN": 10,
}

# VBR — VWAP Band Reversion
VBR_OVERRIDES = {
    "ENTRY_MODE": "pullback",
    "RVOL_MIN": 1.2,
    "ATR_MIN_FOR_TREND": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True, "SL_ATR_MULT": 0.8, "TP1_ATR_MULT": 1.2, "TP2_ATR_MULT": 2.0,
    "TRAIL_AFTER_TP1": True, "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.003, "MAX_HOLD_HOURS": 6, "SYMBOL_COOLDOWN_MIN": 8,
}

# SR-only — Support/Resistance + RSI فقط
SR_ONLY_OVERRIDES = {
    "SYMBOL_COOLDOWN_MIN": 8,   # تبريد بسيط
    "RSI_GATE_POLICY": None,    # بوابة RSI تُطبّق يدويًا داخل الدالة
}

# نطاقات RSI حسب النمط (عامّة)
RSI_MIN_PULLBACK, RSI_MAX_PULLBACK = 45, 65
RSI_MIN_BREAKOUT, RSI_MAX_BREAKOUT = 50, 80

# ===== إدارة صفقة حسب الاستراتيجية =====
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
    # SR-only: SL أسفل السوينغ/الدعم مباشرة (SL_MULT=0), TP1 سيُمرَّر custom عند المقاومة الأقرب
    "sr":  {"SL":"atr_below_sweep", "SL_MULT":0.0, "TP1":"range_or_atr",
            "TRAIL_AFTER_TP1":False, "TIME_HRS":6},
}

def _mgmt(variant: str):
    return PER_STRAT_MGMT.get(variant, PER_STRAT_MGMT["new"])

# ======= فلترة متعددة الفريمات =======
ENABLE_MTF_STRICT = True
MTF_UP_TFS = ("4h", "1h", "15m")
SCORE_THRESHOLD = 60  # يمكن تقويته/تليينه

# --------- طبقات SR متعددة ---------
SR_LEVELS_CFG = [
    ("micro", LTF_TIMEFRAME,  50, 0.8),   # (اسم, TF, نافذة رولينغ, مضاعِف ATR_LTF للقرب)
    ("meso",  "1h",  50, 1.0),
    ("macro", "4h",  50, 1.3),
    # ("macro2","1d", 60, 1.6),  # فعّلها لو تريد اليومي
]

# ================== Helpers عامة ==================

def _tg(text, parse_mode="HTML"):
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        if parse_mode:
            data["parse_mode"] = parse_mode
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass

def now_riyadh(): return datetime.now(RIYADH_TZ)
def _today_str(): return now_riyadh().strftime("%Y-%m-%d")

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
    # توحيد الطابع الزمني إلى milliseconds
    try:
        if len(df) and df["timestamp"].iloc[-1] < 10**12:
            df["timestamp"] = df["timestamp"] * 1000
    except Exception:
        pass
    return df

# تقسيم الرمز إلى أساس/نسخة (#old/#new/#srr/#brt/#vbr/#sr)
def _split_symbol_variant(symbol: str):
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower()
        if variant not in ("old","new","srr","brt","vbr","sr"):
            variant = "new"
        return base, variant
    return symbol, "new"

# دمج إعدادات النسخة
def get_cfg(variant: str):
    cfg = dict(BASE_CFG)
    if variant == "new":
        cfg.update(NEW_SCALP_OVERRIDES)
    elif variant == "srr":
        cfg.update(SRR_OVERRIDES)
    elif variant == "brt":
        cfg.update(BRT_OVERRIDES)
    elif variant == "vbr":
        cfg.update(VBR_OVERRIDES)
    elif variant == "sr":
        cfg.update(SR_ONLY_OVERRIDES)
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

# ================== حالة المخاطر اليومية ==================
def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0,
            "trades_today": 0, "blocked_until": None}

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str():
        s = _default_risk_state(); save_risk_state(s)
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

def register_trade_result(pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1

    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s); _set_block(BLOCK_AFTER_LOSSES_MIN := 90, reason="خسائر متتالية"); return
    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s); _set_block(minutes, reason="تجاوز حد الخسارة اليومي"); return
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
    df["ema9"], df["ema21"], df["ema50"] = ema(df["close"], EMA_FAST), ema(df["close"], EMA_SLOW), ema(df["close"], EMA_TREND)
    df["rsi"] = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA).mean()
    df = macd_cols(df)
    return df

# LTF: VWAP/RVol/NR
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

# ===== ATR =====
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

# -------- SR متعددة الطبقات --------
def _rolling_sr(symbol, tf: str, window: int, bars: int = 300):
    data = fetch_ohlcv(symbol, tf, bars)
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

# ================== سياق HTF (مع كاش) ==================
def _get_htf_context(symbol):
    base, _ = _split_symbol_variant(symbol)

    # كاش
    now = now_riyadh()
    ent = _HTF_CACHE.get(base)
    if ent and (now - ent["t"]).total_seconds() <= _HTF_TTL_SEC:
        return ent["ctx"]

    data = fetch_ohlcv(base, HTF_TIMEFRAME, 200)
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

    # فلتر اتجاهي إضافي (تصويتي)
    if ENABLE_MTF_STRICT:
        def _tf_info(tf, bars=160):
            try:
                d = fetch_ohlcv(base, tf, bars)
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

# ================== منطق الدخول (النسخة القديمة) ==================
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
    return (closed["close"] > hi_range) and is_nr_recent and vwap_ok

# ================== فحص الإشارة — OLD ==================
def check_signal_old(symbol):
    ok, _ = _risk_precheck_allow_new_entry()
    if not ok: return None

    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)
    key = f"{base}|{variant}"

    last_t = _SYMBOL_LAST_TRADE_AT.get(key)
    if last_t and (now_riyadh() - last_t) < timedelta(minutes=cfg["SYMBOL_COOLDOWN_MIN"]):
        return None
    if load_position(symbol): return None

    ctx = _get_htf_context(symbol)
    if not ctx: return None
    if not ((ctx["ema50_now"] - ctx["ema50_prev"]) > 0 and ctx["close"] > ctx["ema50_now"]):
        return None

    data = fetch_ohlcv(base, LTF_TIMEFRAME, 260)
    if not data: return None
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 120: return None

    prev, closed = df.iloc[-3], df.iloc[-2]
    last_ts_closed = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == last_ts_closed: return None

    price = float(closed["close"]); atr_ltf = _atr_from_df(df)
    if not atr_ltf or atr_ltf <= 0: return None
    if (atr_ltf / max(1e-9, price)) < cfg["ATR_MIN_FOR_TREND"]: return None

    dist = price - float(closed["ema50"])
    if dist < 0.5 * atr_ltf: return None
    if dist > 3.0 * atr_ltf: return None

    if (closed["close"] * closed["volume"]) < 60000: return None
    if pd.isna(closed.get("rvol")) or closed["rvol"] < max(cfg["RVOL_MIN"], 1.2): return None
    if closed["close"] <= closed["open"]: return None

    # قرب مقاومة (أشد قليلًا في OLD)
    if ctx.get("resistance") and (ctx["resistance"] - price) < 1.4 * atr_ltf: return None
    if ctx.get("support") and price <= ctx["support"] * (1 + SUPPORT_BUFFER): return None

    chosen_mode = None; mode_ok = False
    if cfg["ENTRY_MODE"] == "pullback":
        chosen_mode = "pullback"; mode_ok = _entry_pullback_logic(df, closed, prev, atr_ltf, ctx, cfg)
    elif cfg["ENTRY_MODE"] == "breakout":
        chosen_mode = "breakout"; mode_ok = _entry_breakout_logic(df, closed, prev, atr_ltf, ctx, cfg)
    elif cfg["ENTRY_MODE"] == "hybrid":
        for m in cfg["HYBRID_ORDER"]:
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr_ltf, ctx, cfg):
                chosen_mode = "pullback"; mode_ok = True; break
            if m == "breakout" and _entry_breakout_logic(df, closed, prev, atr_ltf, ctx, cfg):
                chosen_mode = "breakout"; mode_ok = True; break
    else:
        crossed = (prev["ema9"] < prev["ema21"]) and (closed["ema9"] > closed["ema21"])
        macd_ok = float(df["macd"].iloc[-2]) > float(df["macd_signal"].iloc[-2])
        chosen_mode = "crossover"; mode_ok = crossed and macd_ok

    if not mode_ok:
        # فرصة بديلة: Breakout أعلى قمة داخلية بشرط عدم الاصطدام بمقاومة قريبة
        sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
        # قرب مقاومة/دعم من طبقات متعددة + LTF/HTF
        sr_multi = get_sr_multi(symbol)
        near_res_any = False
        nearest_res = None
        for name, ent in sr_multi.items():
            res = ent.get("resistance")
            if res:
                if nearest_res is None or res < nearest_res:
                    nearest_res = res
                if (res - price) < (ent["near_mult"] * atr_ltf):
                    near_res_any = True

        near_res_ltf = bool(res_ltf and (res_ltf - price) < 0.8 * atr_ltf)
        near_res_htf = bool(ctx.get("resistance") and (ctx["resistance"] - price) < 1.3 * atr_ltf)

        # سكّور بسيط + تعزيز بسيط لسيناريو SRR
        score, why, patt = _opportunity_score(df, prev, closed)
        score += 15; patt = "SweepReclaim"; why = (why + ", SRR")
        # إذا قرب مقاومة قوية، اسمح فقط إذا السكور قوي أو RVOL مرتفع
        if (near_res_ltf or near_res_htf or near_res_any) and not (score >= 62 or float(closed.get("rvol",0)) >= max(1.2, cfg["RVOL_MIN"]) * 1.05):
            return _rej("near_res_block")

        # تلميحات SL/TP
        _, llv = recent_swing(df, lookback=60)
        if llv is None:
            llv = float(closed["low"])
        sl_hint = min(float(closed["low"]), float(llv)) * 0.999
        tp1_hint = None
        if res_ltf and (res_ltf - price) > 0 and (res_ltf - price) < 2.5 * atr_ltf:
            tp1_hint = float(res_ltf)
        if nearest_res and (nearest_res > price) and ((nearest_res - price) < 2.5 * atr_ltf):
            tp1_hint = float(min(tp1_hint, nearest_res)) if tp1_hint else float(nearest_res)

        _LAST_ENTRY_BAR_TS[key] = last_ts_closed
        return {"decision": "buy", "score": score, "reason": why, "pattern": patt,
                "ts": last_ts_closed, "custom": {"sl": sl_hint, "tp1": tp1_hint}}

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

# ================== NEW/SRR — متكّيف بالـ ATR + فلترة MTF/SR ==================
def check_signal_new(symbol):
    ok, reason = _risk_precheck_allow_new_entry()
    if not ok: return _rej("risk_precheck", reason=reason)

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

    data = fetch_ohlcv(base, LTF_TIMEFRAME, 260)
    if not data: return _rej("ltf_fetch")
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 120: return _rej("ltf_len", n=len(df))

    prev, closed = df.iloc[-3], df.iloc[-2]
    ts = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == ts: return _rej("dup_bar")

    price = float(closed["close"]); atr = _atr_from_df(df)
    if not atr or atr <= 0: return _rej("atr_nan")
    atrp = atr / max(price, 1e-9)
    if atrp < float(cfg.get("ATR_MIN_FOR_TREND", 0.002)): return _rej("atr_low", atrp=round(atrp,5))

    # مسافة عن EMA50 — (تحسين: 0.20 بدل 0.30) + سقف كما هو
    dist = price - float(closed["ema50"])
    if not (0.20*atr <= dist <= 4.00*atr):
        return _rej("dist_to_ema50", dist_atr=round(dist/atr,3))

    # سيولة/حجم + RVOL (تحسين: 0.90 × RVOL_MIN)
    notional = price * float(closed["volume"])
    if notional < 60000: return _rej("notional_low", notional=int(notional))
    rvol = float(closed.get("rvol", 0) or 0)
    need_rvol = float(cfg.get("RVOL_MIN", 1.2)) * 0.90
    if pd.isna(rvol) or rvol < need_rvol: return _rej("rvol", rvol=round(rvol,2), need=round(need_rvol,2))

    # بوابة MACD/RSI
    policy = cfg.get("RSI_GATE_POLICY") or "lenient"
    if not macd_rsi_gate(prev, closed, policy=policy): return _rej("macd_rsi_gate", policy=policy)

    # اختيار النمط
    def _brk_ok():
        hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        vwap_ok = price > float(closed.get("vwap", closed.get("ema21", price)))
        return (price > hi_range * (1.0 + float(cfg.get("BREAKOUT_BUFFER_LTF", 0.0015)))) and (is_nr_recent or vwap_ok)

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
    if not mode_ok: return _rej("entry_mode", mode=entry_mode)

    # نطاقات RSI حسب النمط — بهوامش
    rsi_val = float(closed.get("rsi", 50))
    if chosen_mode == "pullback" and not (RSI_MIN_PULLBACK - 3 < rsi_val < RSI_MAX_PULLBACK + 2):
        return _rej("rsi_pullback", rsi=rsi_val)
    if chosen_mode == "breakout" and not (RSI_MIN_BREAKOUT - 2 < rsi_val < RSI_MAX_BREAKOUT + 2):
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

    score, why, patt = _opportunity_score(df, prev, closed)
    if near_res and not (score >= (SCORE_THRESHOLD-2) or chosen_mode == "breakout" or rvol >= need_rvol*1.05):
        return _rej("near_res_block", score=score)

    if score < SCORE_THRESHOLD: return _rej("score_low", score=score)

    _LAST_ENTRY_BAR_TS[key] = ts
    _pass("signal_ok", mode=chosen_mode, score=score, rvol=round(rvol,2), atrp=round(atrp,4))
    return {"decision": "buy", "score": score, "reason": why, "pattern": (patt if chosen_mode!="breakout" else "NR_Breakout"), "ts": ts}

# ================== BRT — Break & Retest ==================
def check_signal_brt(symbol):
    ok, reason = _risk_precheck_allow_new_entry()
    if not ok: return None
    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)
    key = f"{base}|{variant}"
    if _SYMBOL_LAST_TRADE_AT.get(key) and (now_riyadh() - _SYMBOL_LAST_TRADE_AT[key]) < timedelta(minutes=cfg["SYMBOL_COOLDOWN_MIN"]): return None
    if load_position(symbol): return None

    ctx = _get_htf_context(symbol)
    if not ctx or not ((ctx["ema50_now"] - ctx["ema50_prev"]) > 0 and ctx["close"] > ctx["ema50_now"]): return None

    data = fetch_ohlcv(base, LTF_TIMEFRAME, 260)
    if not data: return None
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 140: return None
    prev, closed = df.iloc[-3], df.iloc[-2]
    ts = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == ts: return None

    price = float(closed["close"]); atr = _atr_from_df(df)
    if not atr or atr <= 0: return None
    if (atr / max(1e-9, price)) < cfg["ATR_MIN_FOR_TREND"]: return None

    hi_range = float(df["high"].iloc[-SR_WINDOW-2:-2].max())
    breakout = price > hi_range * (1.0 + 0.0015) and (closed.get("rvol", 0) >= cfg["RVOL_MIN"])
    if not breakout: return None

    retest_touched = (float(prev["low"]) <= hi_range * 1.002) or (float(closed["low"]) <= hi_range * 1.002)
    reclaimed = (price > hi_range) and (price > float(closed.get("vwap", closed.get("ema21", price))))
    if not (retest_touched and reclaimed): return None

    if not macd_rsi_gate(prev, closed, policy=cfg["RSI_GATE_POLICY"]): return None
    sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
    # تشديد: منع الإشارات إذا قرب أي مقاومة متعددة الطبقات
    sr_multi = get_sr_multi(symbol)
    near_res_any = False
    for name, ent in sr_multi.items():
        res = ent.get("resistance")
        if res and (res - price) < (ent["near_mult"] * atr):
            near_res_any = True
            break

    near_res = (res_ltf and (res_ltf - price) < 0.8*atr) or (ctx.get("resistance") and (ctx["resistance"] - price) < 1.3*atr) or near_res_any
    if near_res: return None

    _LAST_ENTRY_BAR_TS[key] = ts
    score, why, patt = _opportunity_score(df, prev, closed); patt = "Break&Retest"; why = (why + ", BRT")
    return {"decision": "buy", "score": score+10, "reason": why, "pattern": patt, "ts": ts}

# ================== VBR — VWAP Band Reversion ==================
def check_signal_vbr(symbol):
    ok, reason = _risk_precheck_allow_new_entry()
    if not ok: return None
    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)
    key = f"{base}|{variant}"
    if _SYMBOL_LAST_TRADE_AT.get(key) and (now_riyadh() - _SYMBOL_LAST_TRADE_AT[key]) < timedelta(minutes=cfg["SYMBOL_COOLDOWN_MIN"]): return None
    if load_position(symbol): return None

    ctx = _get_htf_context(symbol)
    if not ctx: return None  # نسمح حتى لو الميل خفيف

    data = fetch_ohlcv(base, LTF_TIMEFRAME, 260)
    if not data: return None
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 120: return None
    prev, closed = df.iloc[-3], df.iloc[-2]
    ts = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == ts: return None

    price = float(closed["close"]); atr = _atr_from_df(df)
    if not atr or atr <= 0: return None
    if (atr / max(1e-9, price)) < cfg["ATR_MIN_FOR_TREND"]: return None

    # Z-score لانحراف السعر عن VWAP (60 شمعة)
    dev = (df["close"] - df["vwap"]).rolling(60).apply(lambda x: (x.iloc[-1] - x.mean())/max(x.std(),1e-9), raw=False)
    z = float(dev.iloc[-2]) if not pd.isna(dev.iloc[-2]) else 0.0

    # LONG: تشبّع سفلي ثم استعادة فوق EMA21/VWAP
    oversold_band = (z <= -2.0)
    reclaimed = (price > float(closed.get("ema21", price))) and (price > float(closed.get("vwap", price)))
    if not (oversold_band and reclaimed): return None

    if (closed.get("rvol", 0) < cfg["RVOL_MIN"]): return None
    if not macd_rsi_gate(prev, closed, policy=cfg["RSI_GATE_POLICY"]): return None
    sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
    sr_multi = get_sr_multi(symbol)
    nearest_res = None
    near_res_any = False
    for name, ent in sr_multi.items():
        res = ent.get("resistance")
        if res:
            if nearest_res is None or res < nearest_res:
                nearest_res = res
            if (res - price) < (ent["near_mult"] * atr):
                near_res_any = True
    near_res_vbr = (res_ltf and (res_ltf - price) < 0.7*atr) or near_res_any
    if near_res_vbr: return None

    _LAST_ENTRY_BAR_TS[key] = ts
    score, why, patt = _opportunity_score(df, prev, closed)
    patt = "VWAP_Reversion"; why = (why + f", z={round(z,2)}")
    swing_hi, swing_lo = _swing_points(df, left=2, right=2)
    sl_hint = (swing_lo * 0.999) if swing_lo else (price - 0.9*atr)
    tp1_vwap = float(closed.get("vwap", price + 1.2*atr))
    tp1_hint = tp1_vwap
    if nearest_res and nearest_res > price:
        tp1_hint = float(min(tp1_vwap, nearest_res))
    return {"decision": "buy", "score": score+8, "reason": why, "pattern": patt, "ts": ts,
            "custom": {"sl": float(sl_hint), "tp1": float(tp1_hint)}}

# ================== SR-only — دعم/مقاومة + RSI ==================
def _nearest_sr_from_levels(price: float, sr_multi: Dict[str, Dict[str, Any]], sup_ltf: Optional[float], res_ltf: Optional[float]):
    nearest_sup = None
    nearest_res = None
    # مرشّحات LTF أولًا
    if sup_ltf and sup_ltf < price:
        nearest_sup = sup_ltf
    if res_ltf and res_ltf > price:
        nearest_res = res_ltf
    # ثم بقية الطبقات (1h, 4h, ...)
    for ent in sr_multi.values():
        sup = ent.get("support"); res = ent.get("resistance")
        if sup and sup < price:
            nearest_sup = sup if nearest_sup is None else max(nearest_sup, sup)
        if res and res > price:
            nearest_res = res if nearest_res is None else min(nearest_res, res)
    return nearest_sup, nearest_res

def check_signal_sr(symbol):
    ok, reason = _risk_precheck_allow_new_entry()
    if not ok: return _rej("risk_precheck", reason=reason)

    base, variant = _split_symbol_variant(symbol)
    cfg = get_cfg(variant)
    key = f"{base}|{variant}"

    # تبريد & عدم وجود صفقة
    last_t = _SYMBOL_LAST_TRADE_AT.get(key)
    if last_t and (now_riyadh() - last_t) < timedelta(minutes=cfg.get("SYMBOL_COOLDOWN_MIN", 8)):
        return _rej("cooldown")
    if load_position(symbol):
        return _rej("already_open")

    # بيانات فريم التنفيذ فقط
    data = fetch_ohlcv(base, LTF_TIMEFRAME, 240)
    if not data: return _rej("ltf_fetch")
    df = _df(data)
    if len(df) < max(SR_WINDOW+3, 60): return _rej("ltf_len", n=len(df))

    # RSI فقط
    df["rsi"] = rsi(df["close"], 14)

    prev, closed = df.iloc[-3], df.iloc[-2]
    ts = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == ts: return _rej("dup_bar")

    price = float(closed["close"])
    rsi_now = float(closed["rsi"]); rsi_prev = float(prev["rsi"])

    # مستويات SR
    sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
    sr_multi = get_sr_multi(symbol)
    nearest_sup, nearest_res = _nearest_sr_from_levels(price, sr_multi, sup_ltf, res_ltf)

    # فلتر مساحة حركة لأقرب مقاومة فوق السعر (0.2%)
    room_ok = True
    if nearest_res:
        room_ok = ((nearest_res - price) / price) >= 0.002

    # نمط 1) استعادة دعم
    reclaim_support = False
    if nearest_sup:
        touched = float(closed["low"]) <= nearest_sup * 1.001  # لمس/كسر بسيط ثم إغلاق فوق
        reclaimed = price > nearest_sup
        rsi_gate = (rsi_now > 50) and (rsi_now > rsi_prev)
        reclaim_support = touched and reclaimed and rsi_gate

    # نمط 2) اختراق مقاومة
    breakout_res = False
    base_res = None
    if res_ltf and res_ltf > price:
        base_res = res_ltf
    elif nearest_res and nearest_res > price:
        base_res = nearest_res
    if base_res:
        breakout_res = (price > base_res * 1.001) and (rsi_now > 55) and (rsi_now > rsi_prev)

    if not (room_ok and (reclaim_support or breakout_res)):
        return _rej("sr_rsi_gate", rsi=round(rsi_now,2))

    # مخرجات: SL أسفل الدعم، TP1 عند أقرب مقاومة
    custom = {}
    if nearest_sup:
        custom["sl"] = float(nearest_sup * 0.999)  # حماية 0.1% تحت الدعم
    if nearest_res and nearest_res > price:
        custom["tp1"] = float(nearest_res)

    _LAST_ENTRY_BAR_TS[key] = ts
    _pass("signal_ok_sr", rsi=round(rsi_now,2))
    return {"decision": "buy", "score": 60, "reason": ("SR Reclaim" if reclaim_support else "SR Breakout"),
            "pattern": "SR_RSI", "ts": ts, "custom": custom}

# ================== Router ==================
def check_signal(symbol):
    base, variant = _split_symbol_variant(symbol)
    if variant == "old":
        return check_signal_old(symbol)
    if variant == "srr":
        # نستخدم منطق NEW مع إعدادات SRR
        return check_signal_new(symbol)
    if variant == "brt":
        return check_signal_brt(symbol)
    if variant == "vbr":
        return check_signal_vbr(symbol)
    if variant == "sr":
        return check_signal_sr(symbol)
    return check_signal_new(symbol)  # الافتراضي

# ================== SL/TP ==================
def _compute_sl_tp(entry, atr_val, cfg, variant, symbol=None, df=None, ctx=None, closed=None):
    """يحسِب SL/TP1/TP2 وفق سياسة الاستراتيجية (PER_STRAT_MGMT).
    يعتمد على ATR وطبقات SR وVWAP إن لزم. يُفضَّل تمرير df/closed والسِمبل لتحسين TP1 الذكي.
    """
    mg = _mgmt(variant)

    # 1) وقف الخسارة
    sl = None
    try:
        if mg.get("SL") == "atr":
            sl = entry - mg.get("SL_MULT", 1.0) * atr_val
        elif mg.get("SL") == "pct":
            sl = entry * (1 - float(mg.get("SL_PCT", cfg.get("STOP_LOSS_PCT", 0.02))))
        elif mg.get("SL") in ("atr_below_sweep", "atr_below_retest"):
            base_level = None
            # تقدير قاع السويب/منطقة الريتست
            try:
                if df is not None and len(df) > 10:
                    # للسويب: أدنى قاع سوينغ حديث
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
            # رجوع لإعدادات cfg الافتراضية
            if cfg.get("USE_ATR_SL_TP") and atr_val and atr_val > 0:
                sl  = entry - cfg.get("SL_ATR_MULT", 1.6)  * atr_val
            else:
                sl  = entry * (1 - cfg.get("STOP_LOSS_PCT", 0.02))
    except Exception:
        sl = entry - 1.0 * atr_val

    # 2) حساب TP1/TP2
    tp1 = None; tp2 = None

    # تحضير SR متعدد الطبقات للحصول على أقرب مقاومة فوق السعر
    nearest_res = None
    try:
        if symbol:
            sr_multi = get_sr_multi(symbol)
            for name, ent in sr_multi.items():
                res = ent.get("resistance")
                if res and res > entry:
                    nearest_res = res if nearest_res is None else min(nearest_res, res)
    except Exception:
        pass

    # VWAP من الشمعة المغلقة
    vwap_val = None
    try:
        if closed is None and df is not None:
            closed = df.iloc[-2]
        vwap_val = float(closed.get("vwap")) if closed is not None else None
    except Exception:
        vwap_val = None

    # ATR TP defaults
    atr_tp1 = entry + float(mg.get("TP1_ATR", 1.2)) * atr_val
    atr_tp2 = entry + float(mg.get("TP2_ATR", 2.2)) * atr_val if mg.get("TP2_ATR") else entry + 2.2 * atr_val

    mode = mg.get("TP1")
    try:
        if mode == "sr_or_atr":
            sr_tp = nearest_res if (nearest_res and nearest_res > entry) else None
            tp1 = float(min(sr_tp, atr_tp1)) if sr_tp else float(atr_tp1)
        elif mode == "range_or_atr":
            # تقدير قمة الرينج/المقاومة
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
            if vwap_val and vwap_val > entry:
                candidates.append(float(vwap_val))
            if nearest_res and nearest_res > entry:
                candidates.append(float(nearest_res))
            tp1 = float(min(candidates)) if candidates else float(atr_tp1)
        else:
            # نسب/افتراضيات cfg
            if mg.get("TP1_PCT"):
                tp1 = entry * (1 + float(mg.get("TP1_PCT")))
            else:
                if cfg.get("USE_ATR_SL_TP") and atr_val and atr_val > 0:
                    tp1 = entry + cfg.get("TP1_ATR_MULT", 1.6) * atr_val
                    tp2 = entry + cfg.get("TP2_ATR_MULT", 3.2) * atr_val
                else:
                    tp1 = entry * (1 + cfg.get("TP1_PCT", 0.03"))
                    tp2 = entry * (1 + cfg.get("TP2_PCT", 0.06))
    except Exception:
        tp1 = atr_tp1

    if tp2 is None:
        tp2 = atr_tp2

    return float(sl), float(tp1), float(tp2)

# ================== تنفيذ الشراء ==================
def execute_buy(symbol):
    base, variant = _split_symbol_variant(symbol)

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."
    if _is_blocked():
        return None, "🚫 ممنوع فتح صفقات الآن (حظر مخاطرة)."
    if load_position(symbol):
        return None, "🚫 لديك صفقة مفتوحة على هذا الرمز/الاستراتيجية."

    # جلب بيانات LTF و HTF للإشارة الجديدة
    ohlcv = fetch_ohlcv(base, LTF_TIMEFRAME, 200)
    if not ohlcv:
        return None, "⚠️ فشل جلب بيانات الشموع."
    htf = {"H1": fetch_ohlcv(base, "1h", 200), "H4": fetch_ohlcv(base, "4h", 200), "D1": fetch_ohlcv(base, "1d", 200)}

    # فحص الإشارة عبر الراوتر الداخلي
    _sig_inner = check_signal(symbol)
    if not _sig_inner:
        return None, "❌ لا توجد إشارة مطابقة."

    # تجهيز SL/TP عبر ATR، مع التقاط تلميحات الاستراتيجية
    df_exec = _df(ohlcv)
    df_exec = _ensure_ltf_indicators(df_exec)
    price_fallback = float(df_exec.iloc[-2]["close"])
    closed = df_exec.iloc[-2]
    atr_val = _atr_from_df(df_exec)
    cfg = get_cfg(variant)
    ctx = _get_htf_context(symbol)

    sl, tp1, tp2 = _compute_sl_tp(price_fallback, atr_val, cfg, variant, symbol=symbol, df=df_exec, ctx=ctx, closed=closed)
    mg = _mgmt(variant)
    custom = (_sig_inner.get("custom") if isinstance(_sig_inner, dict) else {}) or {}
    if "sl" in custom and isinstance(custom["sl"], (int, float)):
        sl = float(custom["sl"])
    if "tp1" in custom and isinstance(custom["tp1"], (int, float)):
        tp1 = min(tp1, float(custom["tp1"]))

    sig = {
        "entry": price_fallback,
        "sl": sl,
        "targets": [tp1, tp2],
        "partials": [TP1_FRACTION, 1.0 - TP1_FRACTION],
        "messages": {"entry": f"🚀 دخول {_sig_inner.get('pattern','Opportunity')}"},
        "score": _sig_inner.get("score"),
        "pattern": _sig_inner.get("pattern"),
        "reason": _sig_inner.get("reason"),
        "max_hold_hours": mg.get("TIME_HRS"),
    }

    # قفل زمني افتراضي للوصول لـ TP1 — 12 شمعة 5m
    sig["max_bars_to_tp1"] = 12

    # تنفيذ شراء
    price = float(sig["entry"]) if isinstance(sig, dict) else None
    usdt = float(fetch_balance("USDT") or 0)
    if usdt < TRADE_AMOUNT_USDT:
        return None, "🚫 رصيد USDT غير كافٍ."
    amount = TRADE_AMOUNT_USDT / price
    if amount * price < MIN_NOTIONAL_USDT:
        return None, "🚫 قيمة الصفقة أقل من الحد الأدنى."

    order = place_market_order(base, "buy", amount)
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    fill_px = float(order.get("average") or order.get("price") or price)

    # حفظ الصفقة
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
        # معلومات إضافية اختيارية
        "score": sig.get("score"),
        "pattern": sig.get("pattern"),
        "reason": sig.get("reason"),
    }
    save_position(symbol, pos)
    register_trade_opened()

    # حدّث تبريد الرمز/النسخة (Cooldown)
    _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()

    # رسالة دخول
    try:
        if pos.get("messages") and pos["messages"].get("entry"):
            _tg(f"{pos['messages']['entry']}\n"
                f"دخول: <code>{fill_px:.6f}</code>\n"
                f"SL: <code>{pos['stop_loss']:.6f}</code>\n"
                f"أهداف: {', '.join(str(round(t,6)) for t in pos['targets'])}")
        else:
            _tg(f"✅ دخول {symbol} عند <code>{fill_px:.6f}</code> | SL <code>{pos['stop_loss']:.6f}</code>")
    except Exception:
        pass

    return order, f"✅ شراء {symbol} | SL: {pos['stop_loss']:.6f}"

# ================== إدارة الصفقة ==================
def manage_position(symbol):
    pos = load_position(symbol)
    if not pos:
        return False

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
        tf_fetch = tf_map.get(tf.upper(), "4h")
        data_htf = fetch_ohlcv(base, tf_fetch, 200)
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
                        _tg(f"🛑 وقف HTF {symbol} عند <code>{exit_px:.6f}</code>")
                    except Exception:
                        pass
                    return True

    # (2) الخروج الزمني للوصول لـ TP1
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
                        _tg(pos["messages"]["time"] if pos.get("messages") else "⌛ خروج زمني")
                    except Exception:
                        pass
                    return True
        except Exception:
            pass

    # (2b) خروج بسبب أقصى مدة احتفاظ عامة حسب الاستراتيجية
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
                    part_qty = amount  # ضمان فوق الحد الأدنى

                order = place_market_order(base, "sell", part_qty)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_gross = (exit_px - entry) * part_qty
                    fees = (entry + exit_px) * part_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
                    pnl_net = pnl_gross - fees

                    # تحديث الصفقة
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
                        variant = pos.get("variant", "new")
                        cfg = get_cfg(variant)
                        if i == 0 and pos["amount"] > 0:
                            lock_sl = entry * (1.0 + float(cfg.get("LOCK_MIN_PROFIT_PCT", 0.0)))
                            if lock_sl > pos["stop_loss"]:
                                pos["stop_loss"] = float(lock_sl)
                                save_position(symbol, pos)
                                try: _tg(f"🔒 تحريك وقف الخسارة لقفل ربح مبدئي: <code>{lock_sl:.6f}</code>")
                                except Exception: pass
                    except Exception:
                        pass

                    # تريلينغ بعد TP2
                    if i >= 1 and pos["amount"] > 0:
                        data_for_atr = fetch_ohlcv(base, LTF_TIMEFRAME, 140)
                        if data_for_atr:
                            df_atr = _df(data_for_atr); atr_val = _atr_from_df(df_atr)
                            if atr_val and atr_val > 0:
                                new_sl = current - atr_val
                                if new_sl > pos["stop_loss"] * (1 + TRAIL_MIN_STEP_RATIO):
                                    pos["stop_loss"] = float(new_sl); save_position(symbol, pos)
                                    try:
                                        _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")
                                    except Exception:
                                        pass

    # (3b) تريلينغ عام حسب سياسة الاستراتيجية بعد تحقق أي TP
    if mgmt.get("TRAIL_AFTER_TP1") and pos["amount"] > 0 and any(pos.get("tp_hits", [])):
        data_for_atr = fetch_ohlcv(base, LTF_TIMEFRAME, 140)
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
    pos = load_position(symbol)
    if not pos: return
    closed = load_closed_positions()

    entry = float(pos["entry_price"]); amount = float(pos["amount"])
    pnl_pct = ((exit_price / entry) - 1.0) if entry else 0.0

    # تسجيل الأهداف المحققة (TP1–TP5)
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
        # تظهر إن وُجدت
        "score": pos.get("score"),
        "pattern": pos.get("pattern"),
        "entry_reason": pos.get("reason"),
        # تخزين الأهداف المحققة
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
    # جرّب أول 10 رموز
    test = [s if "#" in s else s+"#new" for s in SYMBOLS[:10]]
    for sym in test:
        r = check_signal(sym)
        print(sym, "→", r)
