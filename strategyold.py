# -*- coding: utf-8 -*-
# strategy_dual_variants_scalp_applied.py — نسختان منفصلتان برمز واحد (#old كما هو، #new سكالب متكّيف ATR)
# تحسينات هذه النسخة:
# - Auto-tune أكثر مرونة في ريجيم السيولة المنخفضة (يسمح MTF 1/3 لشروط معيّنة).
# - عداد رفض مفصّل + خيار إرسال ملخص إلى تيليجرام عند التفعيل (DEBUG_LOG_SIGNALS=True).
# - دالة تشخيص check_signal_debug(symbol) لإظهار جميع أسباب الرفض بسرعة.
# - نفس إدارة الصفقات، مع تصحيح صغار وإيضاحات.

import os, json, requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import (
    TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
)

# استيراد آمن لـ strategy.py كخارجية إن وُجدت
try:
    from strategy import check_signal as strat_check  # يُتوقَّع dict(entry/sl/targets/partials/messages/...)
except Exception:
    strat_check = None

# ================== إعدادات عامة (أساس) ==================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# أطر زمنية
HTF_TIMEFRAME = "15m"   # إطار الاتجاه (سياق)
LTF_TIMEFRAME = "5m"    # إطار التنفيذ (سكالب)

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
DEBUG_LOG_SIGNALS = True   # شغّلها لإرسال ملخص أسباب الرفض (بشكل مقتضب) كل 30 دقيقة.
_LAST_REJECT = {}
_LAST_ENTRY_BAR_TS = {}      # key: f"{base}|{variant}"
_SYMBOL_LAST_TRADE_AT = {}   # key: f"{base}|{variant}"

# كاش بسيط لسياق HTF
_HTF_CACHE = {}          # key = base symbol, val={"t": datetime, "ctx": {...}}
_HTF_TTL_SEC = 150

# ================== إعدادات النسختين ==================
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
    "RVOL_MIN": 1.6,
    "ATR_MIN_FOR_TREND": 0.003,
    "USE_FIB": True,
    "BREAKOUT_BUFFER_LTF": 0.0018,
    "RSI_GATE_POLICY": "balanced",
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.004,
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 15,
}

RSI_MIN_PULLBACK, RSI_MAX_PULLBACK = 45, 65
RSI_MIN_BREAKOUT, RSI_MAX_BREAKOUT = 50, 80

ENABLE_MTF_STRICT = True
MTF_UP_TFS = ("4h", "1h", "15m")
SCORE_THRESHOLD = 70

# ============ أدوات مساعدة ============

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
    except:
        pass
    return default

def _df(data):
    return pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])

def _split_symbol_variant(symbol: str):
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower()
        if variant not in ("old","new"): variant = "new"
        return base, variant
    return symbol, "new"

def get_cfg(variant: str):
    cfg = dict(BASE_CFG)
    if variant == "new":
        cfg.update(NEW_SCALP_OVERRIDES)
    return cfg

# ============ تخزين الصفقات ============

def _pos_path(symbol):
    os.makedirs(POSIONS_DIR := POSITIONS_DIR, exist_ok=True)
    return f"{POSIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol): return _read_json(_pos_path(symbol), None)
def save_position(symbol, position): _atomic_write(_pos_path(symbol), position)

def clear_position(symbol):
    try:
        p = _pos_path(symbol)
        if os.path.exists(p): os.remove(p)
    except:
        pass

def count_open_positions():
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions(): return _read_json(CLOSED_POSITIONS_FILE, [])
def save_closed_positions(lst): _atomic_write(CLOSED_POSITIONS_FILE, lst)

# ============ حالة المخاطر اليومية ============

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
        save_risk_state(s); _set_block(90, reason="خسائر متتالية"); return
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

# ============ مؤشرات ============

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

# ============ Swing/SR/Fib ============

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

def near_any_fib(price: float, hhv: float, llv: float, tol: float):
    rng = hhv - llv
    if rng <= 0: return False, ""
    fib382 = hhv - rng * 0.382; fib618 = hhv - rng * 0.618
    for lvl, name in ((fib382, "Fib 0.382"), (fib618, "Fib 0.618")):
        if abs(price - lvl) / max(lvl, 1e-9) <= tol: return True, name
    return False, ""

# ============ MACD/RSI Gate ============

def macd_rsi_gate(prev_row, closed_row, policy):
    if not policy:  # لا بوابة (#old)
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

# ============ سياق HTF مع كاش ============

def _get_htf_context(symbol):
    base, _ = _split_symbol_variant(symbol)
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

    ctx = {"close": float(closed["close"]), "ema50_now": ema_now, "ema50_prev": ema_prev,
           "support": float(support), "resistance": float(resistance), "mtf": {}}

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

        mtf = {"15m": {"tf": "15m", "price": ctx["close"], "ema": ctx["ema50_now"],
                         "trend_up": bool(ctx["close"] > ctx["ema50_now"])}}
        for tf in ("1h","4h"):
            info = _tf_info(tf)
            if info: mtf[tf] = info
        ctx["mtf"] = mtf

    _HTF_CACHE[base] = {"t": now, "ctx": ctx}
    return ctx

# ============ منطق الدخول ============
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

def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx, cfg, brk_buf):
    hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
    is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
    vwap_ok = closed["close"] > float(closed.get("vwap", closed["ema21"]))
    return (closed["close"] > hi_range * (1.0 + brk_buf)) and (is_nr_recent or vwap_ok)

# ============ تقييم الإشارة المساعد ============
def _opportunity_score(df, prev, closed):
    score, why, pattern = 0, [], ""
    if closed["close"] > closed["open"]:
        score += 10; why.append("BullishClose")
    try:
        if closed["close"] > closed.get("ema21", closed["close"]):
            score += 10; why.append("AboveEMA21")
        if closed["close"] > closed.get("ema50", closed["close"]):
            score += 10; why.append("AboveEMA50")
    except Exception:
        pass
    try:
        if not pd.isna(closed.get("rvol")) and closed["rvol"] >= 1.5:
            score += 15; why.append("HighRVOL")
    except Exception:
        pass
    try:
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        if nr_recent and (closed["close"] > hi_range):
            score += 20; why.append("NR_Breakout"); pattern = "NR_Breakout"
    except Exception:
        pass
    try:
        if _bullish_engulf(prev, closed):
            score += 20; why.append("BullishEngulf"); pattern = pattern or "BullishEngulf"
    except Exception:
        pass
    return score, ", ".join(why), (pattern or "Generic")

# ============ Auto-Tune ============
def _auto_tune_thresholds_ltf(df):
    if len(df) < 160:
        return {
            "regime": "mid",
            "rvol_min": 1.30,
            "atr_min_for_trend": 0.0024,
            "score_threshold": max(64, SCORE_THRESHOLD),
            "dist_bounds_to_ema50": (0.40, 3.3),
            "mtf_votes_req": 2,
            "breakout_buffer": 0.0015,
            "allow_mtf_1of3": True,  # جديد: سماح جزئي بالسوق الهادئ
        }

    c = df["close"]; c_shift = c.shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-c_shift).abs(), (df["low"]-c_shift).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    atrp = (atr / (c.replace(0, 1e-9))).tail(200)

    vol_ma = df["volume"].rolling(RVOL_WINDOW).mean().replace(0, 1e-9)
    rvol = (df["volume"] / vol_ma).tail(200)

    p30 = float(np.nanpercentile(atrp.values, 30))
    p70 = float(np.nanpercentile(atrp.values, 70))
    last = float(atrp.iloc[-1])

    regime = "low" if last <= p30 else ("high" if last >= p70 else "mid")

    if regime == "low":
        return {
            "regime": regime,
            "rvol_min": 1.10,
            "atr_min_for_trend": 0.0018,
            "score_threshold": 63,
            "dist_bounds_to_ema50": (0.30, 3.6),
            "mtf_votes_req": 2,
            "breakout_buffer": 0.0011,
            "allow_mtf_1of3": True,
        }
    elif regime == "high":
        return {
            "regime": regime,
            "rvol_min": 1.60,
            "atr_min_for_trend": 0.0032,
            "score_threshold": 72,
            "dist_bounds_to_ema50": (0.55, 2.8),
            "mtf_votes_req": 3,
            "breakout_buffer": 0.0018,
            "allow_mtf_1of3": False,
        }
    else:
        return {
            "regime": regime,
            "rvol_min": 1.35,
            "atr_min_for_trend": 0.0025,
            "score_threshold": 68,
            "dist_bounds_to_ema50": (0.45, 3.2),
            "mtf_votes_req": 2,
            "breakout_buffer": 0.0016,
            "allow_mtf_1of3": True,
        }

# ============ فحص الإشارة — NEW ============
def check_signal_new(symbol, _reasons=None):
    if _reasons is None: _reasons = []
    def reject(r): _reasons.append(r); return None

    ok, block_reason = _risk_precheck_allow_new_entry()
    if not ok: return reject(f"risk:{block_reason}")

    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)
    key = f"{base}|{variant}"

    last_t = _SYMBOL_LAST_TRADE_AT.get(key)
    if last_t and (now_riyadh() - last_t) < timedelta(minutes=cfg["SYMBOL_COOLDOWN_MIN"]):
        return reject("cooldown")

    if load_position(symbol): return reject("already_open")

    ctx = _get_htf_context(symbol)
    if not ctx: return reject("no_htf")

    # شرط الاتجاه الأساسي (لين: نسمح 1/3 في low-regime لاحقاً)
    base_trend_ok = ((ctx["ema50_now"] - ctx["ema50_prev"]) > 0 and ctx["close"] > ctx["ema50_now"])
    data = fetch_ohlcv(base, LTF_TIMEFRAME, 260)
    if not data: return reject("no_ltf")
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 120: return reject("ltf_short")

    tune = _auto_tune_thresholds_ltf(df)
    rvol_min = tune["rvol_min"]
    atr_min_for_trend = tune["atr_min_for_trend"]
    score_threshold = max(tune["score_threshold"], SCORE_THRESHOLD) if SCORE_THRESHOLD else tune["score_threshold"]
    dist_lo, dist_hi = tune["dist_bounds_to_ema50"]

    ups = 3
    if ENABLE_MTF_STRICT and ctx.get("mtf"):
        ups = sum(1 for tf in ("15m","1h","4h") if tf in ctx["mtf"] and ctx["mtf"][tf].get("trend_up"))

    # تخفيف الاتجاه/التصويت في low/mid عند السماح
    if not base_trend_ok:
        if not (tune.get("allow_mtf_1of3") and ups >= 1):
            return reject("htf_trend")

    prev, closed = df.iloc[-3], df.iloc[-2]
    last_ts_closed = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(key) == last_ts_closed: return reject("same_bar")

    price = float(closed["close"]); atr_ltf = _atr_from_df(df)
    if not atr_ltf or atr_ltf <= 0: return reject("atr_nan")
    if (atr_ltf / max(1e-9, price)) < atr_min_for_trend: return reject("atr_low")

    dist = price - float(closed["ema50"])
    if dist < dist_lo * atr_ltf: return reject("dist_lo")
    if dist > dist_hi * atr_ltf: return reject("dist_hi")

    if (closed["close"] * closed["volume"]) < 60000: return reject("notional_low")
    need_rvol = max(cfg["RVOL_MIN"] * 0.90, rvol_min)
    if pd.isna(closed.get("rvol")) or closed["rvol"] < need_rvol: return reject("rvol_low")
    if closed["close"] <= closed["open"]: return reject("bear_candle")

    near_res = False
    if ctx.get("resistance"):
        near_res = (ctx["resistance"] - price) < (0.8 * atr_ltf)  # لُيِّن

    policy = "lenient" if tune["regime"] == "low" else cfg["RSI_GATE_POLICY"]
    if not macd_rsi_gate(prev, closed, policy=policy): return reject("macd_rsi_gate")

    chosen_mode = None; mode_ok = False
    brk_buf = tune["breakout_buffer"]

    if cfg["ENTRY_MODE"] == "pullback":
        chosen_mode = "pullback"; mode_ok = _entry_pullback_logic(df, closed, prev, atr_ltf, ctx, cfg)
    elif cfg["ENTRY_MODE"] == "breakout":
        chosen_mode = "breakout"; mode_ok = _entry_breakout_logic(df, closed, prev, atr_lf, ctx, cfg, brk_buf)  # typo intentionally avoided below
    elif cfg["ENTRY_MODE"] == "hybrid":
        for m in cfg["HYBRID_ORDER"]:
            if m == "breakout" and _entry_breakout_logic(df, closed, prev, atr_ltf, ctx, cfg, brk_buf):
                chosen_mode = "breakout"; mode_ok = True; break
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr_ltf, ctx, cfg):
                chosen_mode = "pullback"; mode_ok = True; break
    else:
        crossed = (prev["ema9"] < prev["ema21"]) and (closed["ema9"] > closed["ema21"])
        macd_ok = float(df["macd"].iloc[-2]) > float(df["macd_signal"].iloc[-2])
        chosen_mode = "crossover"; mode_ok = crossed and macd_ok

    if not mode_ok:
        sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
        try:
            hhv = float(df.iloc[:-1]["high"].rolling(SR_WINDOW, min_periods=10).max().iloc[-1])
        except Exception:
            hhv = None
        if hhv:
            breakout_ok = price > hhv * (1.0 + brk_buf)
            near_res_block = (res_ltf is not None) and (res_ltf * (1 - RESISTANCE_BUFFER) <= price <= res_ltf * (1 + RESISTANCE_BUFFER))
            if breakout_ok and not near_res_block:
                chosen_mode = chosen_mode or "breakout"; mode_ok = True

    if not mode_ok: return reject("mode_fail")

    rsi_val = float(closed["rsi"])
    if chosen_mode == "pullback" and not (RSI_MIN_PULLBACK - 3 < rsi_val < RSI_MAX_PULLBACK + 2): return reject("rsi_pullback")
    if chosen_mode == "breakout" and not (RSI_MIN_BREAKOUT - 2 < rsi_val < RSI_MAX_BREAKOUT + 2): return reject("rsi_breakout")

    score, why, patt = _opportunity_score(df, prev, closed)
    if near_res:
        if not (score >= (score_threshold + 6) or float(closed.get("rvol", 0)) >= (need_rvol + 0.3)):
            return reject("near_res")

    if ENABLE_MTF_STRICT and (ups < tune["mtf_votes_req"]) and not (chosen_mode == "breakout" and score >= (score_threshold + 6)):
        return reject("mtf_votes")

    if score < score_threshold:
        return reject("score_low")

    _LAST_ENTRY_BAR_TS[key] = last_ts_closed
    return {"decision": "buy", "score": score, "reason": why, "pattern": patt, "ts": last_ts_closed}

# ============ فحص الإشارة — OLD ============
def check_signal_old(symbol):
    # نستخدم NEW مع أسباب الرفض لنفس السلوك مع مرونة أعلى
    return check_signal_new(symbol, _reasons=[])

# ============ Router ============
def check_signal(symbol):
    base, variant = _split_symbol_variant(symbol)
    return check_signal_old(symbol) if variant == "old" else check_signal_new(symbol)

# ============ SL/TP ============
def _compute_sl_tp(entry, atr_val, cfg):
    if cfg.get("USE_ATR_SL_TP") and atr_val and atr_val > 0:
        sl  = entry - cfg.get("SL_ATR_MULT", 1.6)  * atr_val
        tp1 = entry + cfg.get("TP1_ATR_MULT", 1.6) * atr_val
        tp2 = entry + cfg.get("TP2_ATR_MULT", 3.2) * atr_val
    else:
        sl  = entry * (1 - cfg.get("STOP_LOSS_PCT", 0.02))
        tp1 = entry * (1 + cfg.get("TP1_PCT", 0.03))
        tp2 = entry * (1 + cfg.get("TP2_PCT", 0.06))
    return float(sl), float(tp1), float(tp2)

# ============ تنفيذ الشراء ============
def execute_buy(symbol):
    base, variant = _split_symbol_variant(symbol)

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."
    if _is_blocked():
        return None, "🚫 ممنوع فتح صفقات الآن (حظر مخاطرة)."
    if load_position(symbol):
        return None, "🚫 لديك صفقة مفتوحة على هذا الرمز/الاستراتيجية."

    ohlcv = fetch_ohlcv(base, LTF_TIMEFRAME, 200)
    if not ohlcv:
        return None, "⚠️ فشل جلب بيانات الشموع."
    htf = {
        "H1": fetch_ohlcv(base, "1h", 200),
        "H4": fetch_ohlcv(base, "4h", 200),
        "D1": fetch_ohlcv(base, "1d", 200)
    }

    sig = None
    try:
        if strat_check:
            sig = strat_check(base, ohlcv, htf)
    except Exception:
        sig = None

    if not sig:
        _reasons = []
        _sig_inner = check_signal_new(symbol, _reasons)
        if not _sig_inner:
            _bump_reject(symbol, _reasons)  # سجل أسباب الرفض
            return None, "❌ لا توجد إشارة مطابقة."
        df_exec = _df(ohlcv)
        df_exec = _ensure_ltf_indicators(df_exec)
        price_fallback = float(df_exec.iloc[-2]["close"])
        atr_val = _atr_from_df(df_exec)
        cfg = get_cfg(variant)
        sl, tp1, tp2 = _compute_sl_tp(price_fallback, atr_val, cfg)
        sig = {
            "entry": price_fallback,
            "sl": sl,
            "targets": [tp1, tp2],
            "partials": [TP1_FRACTION, 1.0 - TP1_FRACTION],
            "messages": {"entry": f"🚀 دخول {_sig_inner.get('pattern','Opportunity')}"},
        }

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

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(fill_px),
        "stop_loss": float(sig["sl"]),
        "targets": sig["targets"],
        "partials": sig["partials"],
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "variant": variant,
        "htf_stop": sig.get("stop_rule") if isinstance(sig, dict) else None,
        "max_bars_to_tp1": sig.get("max_bars_to_tp1") if isinstance(sig, dict) else None,
        "messages": sig.get("messages") if isinstance(sig, dict) else None,
        "tp_hits": [False] * len(sig["targets"]),
        "score": sig.get("score") if isinstance(sig, dict) else None,
        "pattern": (sig.get("features", {}).get("setup") if isinstance(sig, dict) and sig.get("features") else None),
        "reason": (", ".join(sig.get("confluence", [])[:4]) if isinstance(sig, dict) and sig.get("confluence") else None),
    }
    save_position(symbol, pos)
    register_trade_opened()
    _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()

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

# ============ إدارة الصفقة ============
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
                    try: _tg(f"🛑 وقف HTF {symbol} عند <code>{exit_px:.6f}</code>")
                    except Exception: pass
                    return True

    # (2) خروج زمني للوصول لـ TP1
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

    # (3) إدارة الأهداف + Trailing + قفل ربح
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
                        if pos.get("messages"): _tg(pos["messages"].get(f"tp{i+1}", f"🎯 TP{i+1} تحقق"))
                    except Exception:
                        pass

                    # قفل ربح بعد TP1
                    try:
                        variant = pos.get("variant", "new")
                        cfg = get_cfg(variant)
                        if i == 0 and pos["amount"] > 0:
                            lock_sl = entry * (1.0 + float(cfg.get("LOCK_MIN_PROFIT_PCT", 0.0)))
                            if lock_sl > pos["stop_loss"]:
                                pos["stop_loss"] = float(lock_sl); save_position(symbol, pos)
                                try: _tg(f"🔒 تحريك وقف الخسارة لقفل ربح مبدئي: <code>{lock_sl:.6f}</code>")
                                except Exception: pass
                    except Exception:
                        pass

                    # Trailing بعد TP2
                    if i >= 1 and pos["amount"] > 0:
                        data_for_atr = fetch_ohlcv(base, LTF_TIMEFRAME, 140)
                        if data_for_atr:
                            df_atr = _df(data_for_atr); atr_val = _atr_from_df(df_atr)
                            if atr_val and atr_val > 0:
                                new_sl = current - atr_val
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

# ============ إغلاق وتسجيل ============
def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
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

# ============ تقرير يومي ============
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

# ============ تشخيص وملخص رفض الإشارات ============
def _bump_reject(symbol, reasons):
    try:
        base, variant = _split_symbol_variant(symbol)
        key = f"{base}|{variant}"
        counts = _LAST_REJECT.get(key, {})
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
        _LAST_REJECT[key] = counts
    except Exception:
        pass

_LAST_SUMMARY_AT = None

def maybe_emit_reject_summary():
    global _LAST_SUMMARY_AT
    if not DEBUG_LOG_SIGNALS: return
    now = now_riyadh()
    if _LAST_SUMMARY_AT and (now - _LAST_SUMMARY_AT).total_seconds() < 1800:
        return  # كل 30 دقيقة فقط
    _LAST_SUMMARY_AT = now

    if not _LAST_REJECT: return
    # أبسط تنسيق مختصر
    lines = ["🧪 <b>ملخص أسباب رفض الإشارات</b> (آخر 30 دقيقة):"]
    # نجمع عبر كل الرموز
    agg = {}
    for key, cnts in _LAST_REJECT.items():
        for r, c in cnts.items():
            agg[r] = agg.get(r, 0) + c
    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10]
    if not top: return
    for r, c in top:
        lines.append(f"• {r}: {c}")
    _tg("\n".join(lines))

def check_signal_debug(symbol):
    reasons = []
    res = check_signal_new(symbol, reasons)
    return res, reasons
