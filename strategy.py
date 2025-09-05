from pathlib import Path

# Build the final patched file content by embedding the user's original and applying the "minimal-invasive" updates.
patched_code = r'''# strategy_dual_variants_scalp_applied.py — نسختان منفصلتان برمز واحد (#old كما هو، #new سكالب متكّيف ATR) — هدفان (TP1/TP2)
# - لا قنوات إضافية ولا قفل نسختين — التنفيذ محلي باستخدام okx_api.
# - #old: يحافظ على إعداداتك الأصلية (reviewed v2)
# - #new: سكالب من هدفين مع تكيّف ATR وتضييق فلاتر الدخول
# - إدارة الصفقة موحدة: بيع جزئي عند TP1 (50%)، نقل SL للتعادل، وتريلينغ بعد TP1

import os, json, requests
from datetime import datetime, timedelta, timezone
import pandas as pd

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import (
    TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
)

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

# إدارة الصفقة: ثوابت عامة (قد تُخصّص لكل نسخة عبر get_cfg)
TP1_FRACTION = 0.5
MIN_NOTIONAL_USDT = 10.0
TRAIL_MIN_STEP_RATIO = 0.001

# مخاطر عامة
MAX_TRADES_PER_DAY       = 20
MAX_CONSEC_LOSSES        = 3
DAILY_LOSS_LIMIT_USDT    = 200.0

# تتبُّع
DEBUG_LOG_SIGNALS = False
_LAST_REJECT = {}
_LAST_ENTRY_BAR_TS = {}      # key: f"{base}|{variant}"
_SYMBOL_LAST_TRADE_AT = {}   # key: f"{base}|{variant}"

# ================== إعدادات النسختين ==================
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
    "RSI_GATE_POLICY": None,  # لا بوابة في #old

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

    "RVOL_MIN": 1.6,
    "ATR_MIN_FOR_TREND": 0.003,

    "USE_FIB": True,
    "BREAKOUT_BUFFER_LTF": 0.0018,
    "RSI_GATE_POLICY": "balanced",

    # إدارة عبر ATR
    "USE_ATR_SL_TP": True,
    "SL_ATR_MULT": 0.9,
    "TP1_ATR_MULT": 1.2,
    "TP2_ATR_MULT": 2.2,

    # تريلينغ/وقت/تبريد
    "TRAIL_AFTER_TP1": True,
    "TRAIL_ATR_MULT": 1.0,
    "LOCK_MIN_PROFIT_PCT": 0.004,  # 0.4%
    "MAX_HOLD_HOURS": 6,
    "SYMBOL_COOLDOWN_MIN": 15,
}

# نطاقات RSI حسب النمط (عامّة)
RSI_MIN_PULLBACK, RSI_MAX_PULLBACK = 45, 65
RSI_MIN_BREAKOUT, RSI_MAX_BREAKOUT = 50, 80

# ======= تحكم اختياري بالفلترة متعددة الفريمات (يحافظ على الجوهر) =======
ENABLE_MTF_STRICT = True    # اجعله False للعودة للسلوك الأصلي سريعًا
MTF_UP_TFS = ("4h", "1h", "15m")  # الفريمات المطلوبة لاعتبار الاتجاه Bullish متوافق
SCORE_THRESHOLD = 70        # حد أدنى لتنفيذ الإشارة بعد نجاح شروطك (يمكن تغييره)

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
    except: pass
    return default

def _df(data):
    return pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])

# تقسيم الرمز إلى أساس/نسخة (#old/#new)

def _split_symbol_variant(symbol: str):
    if "#" in symbol:
        base, variant = symbol.split("#", 1)
        variant = (variant or "new").lower()
        if variant not in ("old","new"): variant = "new"
        return base, variant
    return symbol, "new"

# دمج إعدادات النسخة

def get_cfg(variant: str):
    cfg = dict(BASE_CFG)
    if variant == "new":
        cfg.update(NEW_SCALP_OVERRIDES)
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
    except: pass

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

def near_any_fib(price: float, hhv: float, llv: float, tol: float):
    rng = hhv - llv
    if rng <= 0: return False, ""
    fib382 = hhv - rng * 0.382; fib618 = hhv - rng * 0.618
    for lvl, name in ((fib382, "Fib 0.382"), (fib618, "Fib 0.618")):
        if abs(price - lvl) / max(lvl, 1e-9) <= tol: return True, name
    return False, ""

# ===== MACD/RSI Gate =====

def macd_rsi_gate(prev_row, closed_row, policy):
    if not policy:  # معناه بوابة متوقفة (#old)
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

# ================== سياق HTF ==================

def _get_htf_context(symbol):
    base, _ = _split_symbol_variant(symbol)

    # 15m كما هو (السلوك القديم)
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

    if not ENABLE_MTF_STRICT:
        return ctx

    # فلتر اتجاهي إضافي من 1h و4h (لا يغيّر مخرجاتك وإنما يضيف حقول معلوماتية)
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

    mtf = {}
    for tf in MTF_UP_TFS:
        if tf == "15m":
            mtf["15m"] = {"tf": "15m", "price": ctx["close"], "ema": ctx["ema50_now"],
                          "trend_up": bool(ctx["close"] > ctx["ema50_now"])}
        else:
            info = _tf_info(tf)
            if info: mtf[tf] = info
    ctx["mtf"] = mtf
    return ctx

# ================== منطق الدخول ==================

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
    if pd.isna(closed.get("rvol")) or closed["rvol"] < cfg["RVOL_MIN"]: return None
    if closed["close"] <= closed["open"]: return None

    if ctx.get("resistance") and (ctx["resistance"] - price) < 1.2 * atr_ltf: return None
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

    if not mode_ok: return None

    rsi_val = float(closed["rsi"])
    if chosen_mode == "pullback" and not (RSI_MIN_PULLBACK < rsi_val < RSI_MAX_PULLBACK): return None
    if chosen_mode == "breakout" and not (RSI_MIN_BREAKOUT < rsi_val < RSI_MAX_BREAKOUT): return None

    _LAST_ENTRY_BAR_TS[key] = last_ts_closed
    return "buy"

# ================== تقييم الإشارة (إضافة غير مغيرة للجوهر) ==================

def _opportunity_score(df, prev, closed):
    """تقييم بسيط وشفاف يُستخدم بعد نجاح شروطك الحالية لرفع جودة الفرص."""
    score, why, pattern = 0, [], ""

    # إغلاق إيجابي
    if closed["close"] > closed["open"]:
        score += 10; why.append("BullishClose")

    # فوق EMA21/EMA50
    try:
        if closed["close"] > closed.get("ema21", closed["close"]):
            score += 10; why.append("AboveEMA21")
        if closed["close"] > closed.get("ema50", closed["close"]):
            score += 10; why.append("AboveEMA50")
    except Exception:
        pass

    # RVOL قوي
    try:
        if not pd.isna(closed.get("rvol")) and closed["rvol"] >= 1.5:
            score += 15; why.append("HighRVOL")
    except Exception:
        pass

    # NR-Breakout تأكيدي
    try:
        nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
        hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
        if nr_recent and (closed["close"] > hi_range):
            score += 20; why.append("NR_Breakout"); pattern = "NR_Breakout"
    except Exception:
        pass

    # ابتلاع شرائي
    try:
        if _bullish_engulf(prev, closed):
            score += 20; why.append("BullishEngulf"); pattern = pattern or "BullishEngulf"
    except Exception:
        pass

    return score, ", ".join(why), (pattern or "Generic")

# ================== فحص الإشارة — NEW (سكالب محسّن) ==================

def check_signal_new(symbol):
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

    # فلتر اتجاه MTF اختياري (يمكن تعطيله من الأعلى)
    if ENABLE_MTF_STRICT:
        mtf = ctx.get("mtf") or {}
        if any(tf not in mtf for tf in MTF_UP_TFS): return None
        if not all(mtf[tf]["trend_up"] for tf in MTF_UP_TFS): return None

    # شرطك الأصلي (EMA50 15m صاعد + السعر فوقه)
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
    if pd.isna(closed.get("rvol")) or closed["rvol"] < cfg["RVOL_MIN"]: return None
    if closed["close"] <= closed["open"]: return None

    if ctx.get("resistance") and (ctx["resistance"] - price) < 1.2 * atr_ltf: return None
    if ctx.get("support") and price <= ctx["support"] * (1 + SUPPORT_BUFFER): return None

    # بوابة MACD/RSI كما هي لديك
    if not macd_rsi_gate(prev, closed, policy=cfg["RSI_GATE_POLICY"]): return None

    # اختيار نمط الدخول كما عندك
    chosen_mode = None; mode_ok = False
    if cfg["ENTRY_MODE"] == "pullback":
        chosen_mode = "pullback"; mode_ok = _entry_pullback_logic(df, closed, prev, atr_ltf, ctx, cfg)
    elif cfg["ENTRY_MODE"] == "breakout":
        chosen_mode = "breakout"; mode_ok = _entry_breakout_logic(df, closed, prev, atr_ltf, ctx, cfg)
    elif cfg["ENTRY_MODE"] == "hybrid":
        for m in cfg["HYBRID_ORDER"]:
            if m == "breakout" and _entry_breakout_logic(df, closed, prev, atr_ltf, ctx, cfg):
                chosen_mode = "breakout"; mode_ok = True; break
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr_ltf, ctx, cfg):
                chosen_mode = "pullback"; mode_ok = True; break
    else:
        crossed = (prev["ema9"] < prev["ema21"]) and (closed["ema9"] > closed["ema21"]) 
        macd_ok = float(df["macd"].iloc[-2]) > float(df["macd_signal"].iloc[-2])
        chosen_mode = "crossover"; mode_ok = crossed and macd_ok

    # بدائلك المحسنة كما هي
    if not mode_ok:
        sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
        try:
            hhv = float(df.iloc[:-1]["high"].rolling(SR_WINDOW, min_periods=10).max().iloc[-1])
        except Exception:
            hhv = None
        if hhv:
            breakout_ok = price > hhv * (1.0 + cfg["BREAKOUT_BUFFER_LTF"])
            near_res_block = (res_ltf is not None) and (res_ltf * (1 - RESISTANCE_BUFFER) <= price <= res_ltf * (1 + RESISTANCE_BUFFER))
            if breakout_ok and not near_res_block:
                chosen_mode = chosen_mode or "breakout"; mode_ok = True
        if not mode_ok and cfg["USE_FIB"]:
            hhv2, llv2 = recent_swing(df, cfg["SWING_LOOKBACK"])
            if hhv2 and llv2:
                near_fib, _ = near_any_fib(price, hhv2, llv2, cfg["FIB_TOL"])
                sup_block = (sup_ltf is not None) and (price <= sup_ltf * (1 + SUPPORT_BUFFER))
                momentum_up = (float(closed["rsi"]) > float(prev["rsi"])) or (float(closed.get("macd_hist",0)) > float(prev.get("macd_hist",0)))
                if near_fib and (not sup_block) and momentum_up:
                    chosen_mode = chosen_mode or "pullback"; mode_ok = True

    if not mode_ok: return None

    rsi_val = float(closed["rsi"])
    if chosen_mode == "pullback" and not (RSI_MIN_PULLBACK < rsi_val < RSI_MAX_PULLBACK): return None
    if chosen_mode == "breakout" and not (RSI_MIN_BREAKOUT < rsi_val < RSI_MAX_BREAKOUT): return None

    # تقييم بعد نجاح الشروط لزيادة التركيز
    score, why, patt = _opportunity_score(df, prev, closed)
    if score < SCORE_THRESHOLD:  # يمنع الفرص الضعيفة
        return None

    _LAST_ENTRY_BAR_TS[key] = last_ts_closed
    # نحافظ على قابلية التوافق: Truthy = buy
    return {"decision": "buy", "score": score, "reason": why, "pattern": patt, "ts": last_ts_closed}

# ================== Router ==================

def check_signal(symbol):
    base, variant = _split_symbol_variant(symbol)
    return check_signal_old(symbol) if variant == "old" else check_signal_new(symbol)

# ================== SL/TP ==================

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

# ================== تنفيذ الشراء ==================

def execute_buy(symbol):
    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)

    if count_open_positions() >= MAX_OPEN_POSITIONS: return None, "🚫 الحد الأقصى للصفقات المفتوحة."
    if _is_blocked(): return None, "🚫 ممنوع فتح صفقات الآن (حظر مخاطرة)."
    if load_position(symbol): return None, "🚫 لديك صفقة مفتوحة على هذا الرمز/الاستراتيجية."

    try:
        price_raw = fetch_price(base); price = float(price_raw)
    except Exception:
        return None, "⚠️ فشل جلب السعر."
    if price <= 0: return None, "⚠️ سعر غير صالح."

    # ATR على إطار التنفيذ
    data_for_atr = fetch_ohlcv(base, LTF_TIMEFRAME, 140)
    atr_val = _atr_from_df(_df(data_for_atr)) if data_for_atr else None

    sl_tmp, tp1_tmp, tp2_tmp = _compute_sl_tp(price, atr_val, cfg)

    usdt = float(fetch_balance("USDT") or 0)
    if usdt < MIN_NOTIONAL_USDT: return None, "🚫 رصيد USDT غير كافٍ."

    # حجم بسيط ثابت حسب TRADE_AMOUNT_USDT
    if usdt < TRADE_AMOUNT_USDT: return None, "🚫 رصيد USDT غير كافٍ."
    amount = TRADE_AMOUNT_USDT / price
    if amount * price < MIN_NOTIONAL_USDT: return None, "🚫 قيمة الصفقة أقل من الحد الأدنى."

    order = place_market_order(base, "buy", amount)
    if not order: return None, "⚠️ فشل تنفيذ الصفقة."

    try:
        fill_px = float(order.get("average") or order.get("price") or price)
        price = fill_px if fill_px > 0 else price
    except Exception:
        pass

    sl, tp1, tp2 = _compute_sl_tp(price, atr_val, cfg)

    # محاولة التقاط تفاصيل الإشارة (لا تؤثر على شيء إن لم تتوفر)
    details = None
    try:
        sig = check_signal_new(symbol)
        if isinstance(sig, dict) and sig.get("decision") == "buy":
            details = {"score": sig.get("score"), "reason": sig.get("reason"), "pattern": sig.get("pattern")}
    except Exception:
        details = None

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(price),
        "stop_loss": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "partial_done": False,
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "atr_on_entry": float(atr_val or 0.0),
        "variant": variant,
        "cfg": cfg,  # لأغراض التشخيص
    }
    if details:
        pos.update(details)

    save_position(symbol, pos)
    _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()

    extra = ""
    if details:
        extra = f"\nScore: <b>{details.get('score')}</b> • {details.get('pattern')} • {details.get('reason')}"
    _tg(
        f"✅ <b>دخول BUY</b> {symbol}\n"
        f"قيمة الشراء: <b>{amount*price:,.2f}$</b>\n"
        f"الكمية: <code>{amount:.6f}</code>\n"
        f"الدخول: <code>{price:.6f}</code>\n"
        f"SL: <code>{sl:.6f}</code>\n"
        f"TP1: <code>{tp1:.6f}</code>\n"
        f"TP2: <code>{tp2:.6f}</code>{extra}"
    )

    register_trade_opened()
    return order, f"✅ شراء {symbol} | SL: {sl:.6f} | TP1: {tp1:.6f} | TP2: {tp2:.6f}"

# ================== إدارة الصفقة ==================

def manage_position(symbol):
    base, variant = _split_symbol_variant(symbol); cfg = get_cfg(variant)

    pos = load_position(symbol)
    if not pos: return False

    current = float(fetch_price(base))
    entry   = float(pos["entry_price"]) 
    sl      = float(pos["stop_loss"]) 
    tp1     = float(pos["tp1"]) 
    tp2     = float(pos["tp2"]) 
    amount  = float(pos["amount"]) 

    if amount <= 0:
        clear_position(symbol); return False

    base_asset = base.split("/")[0]
    wallet_balance = float(fetch_balance(base_asset) or 0)
    if wallet_balance <= 0:
        _tg(f"⚠️ لا يوجد رصيد {base_asset} للبيع في {symbol} — سيتم إغلاق المركز محليًا.")
        clear_position(symbol); return False

    sellable = min(amount, wallet_balance)

    # إغلاق زمني للسكالب
    try:
        opened_at = datetime.fromisoformat(pos.get("opened_at"))
        if now_riyadh() - opened_at > timedelta(hours=cfg["MAX_HOLD_HOURS"]):
            order = place_market_order(base, "sell", sellable)
            if order:
                exit_px = float(order.get("average") or order.get("price") or current)
                pnl_gross = (exit_px - entry) * sellable
                fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
                pnl_net = pnl_gross - fees
                _tg(f"⌛️ <b>إغلاق زمني</b> {symbol} @ <code>{exit_px:.6f}</code> • P/L: <b>{pnl_net:.2f}$</b>")
                close_trade(symbol, exit_px, pnl_net, reason="TIME")
                _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()
                return True
    except Exception:
        pass

    # TP1: بيع جزئي + نقل SL للتعادل
    if (not pos.get("partial_done")) and current >= tp1 and sellable > 0:
        part_qty = sellable * TP1_FRACTION
        if part_qty * current < MIN_NOTIONAL_USDT: part_qty = sellable
        order = place_market_order(base, "sell", part_qty)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * part_qty
            fees = (entry + exit_px) * part_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees

            pos["amount"] = float(max(0.0, amount - part_qty))
            pos["partial_done"] = True
            pos["stop_loss"] = float(max(entry, sl))  # التعادل
            save_position(symbol, pos)
            register_trade_result(pnl_net)
            _tg(
                f"🎯 <b>TP1 تحقق</b> {symbol}\n"
                f"تم بيع: <code>{part_qty:.6f} {base_asset}</code>\n"
                f"السعر: <code>{exit_px:.6f}</code>\n"
                f"SL ← التعادل."
            )

    # تحديث مرجعي بعد TP1
    pos_ref = load_position(symbol)
    if not pos_ref: return True
    amount = float(pos_ref.get("amount", 0.0))
    wallet_balance = float(fetch_balance(base_asset) or 0)
    sellable = min(amount, wallet_balance)
    sl = float(pos_ref.get("stop_loss", sl))

    # تريلينغ بعد TP1 (حسب cfg)
    if cfg["TRAIL_AFTER_TP1"] and pos_ref.get("partial_done") and sellable > 0:
        data_for_atr = fetch_ohlcv(base, LTF_TIMEFRAME, 140)
        if data_for_atr:
            df_atr = _df(data_for_atr); atr_val = _atr_from_df(df_atr)
            if atr_val and atr_val > 0:
                new_sl_atr = current - cfg["TRAIL_ATR_MULT"] * atr_val
                new_sl = max(sl, new_sl_atr, entry * (1 + cfg["LOCK_MIN_PROFIT_PCT"]))
                if new_sl > sl * (1 + TRAIL_MIN_STEP_RATIO):
                    pos_ref["stop_loss"] = float(new_sl); save_position(symbol, pos_ref)
                    _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")

    # TP2: إغلاق كامل
    if sellable > 0 and current >= tp2:
        order = place_market_order(base, "sell", sellable)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sellable
            fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            _tg(
                f"🏁 <b>TP2</b> {symbol} — <b>إغلاق كامل</b>\n"
                f"كمية: <code>{sellable:.6f} {base_asset}</code>\n"
                f"السعر: <code>{exit_px:.6f}</code>\n"
                f"P/L: <b>{pnl_net:.2f}$</b>"
            )
            close_trade(symbol, exit_px, pnl_net, reason="TP2")
            _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()
            return True

    # SL: إغلاق كامل
    pos_ref = load_position(symbol)
    if not pos_ref: return True
    amount = float(pos_ref.get("amount", 0.0))
    wallet_balance = float(fetch_balance(base_asset) or 0)
    sellable = min(amount, wallet_balance)
    sl = float(pos_ref.get("stop_loss", sl))

    if sellable > 0 and current <= sl:
        order = place_market_order(base, "sell", sellable)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sellable
            fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            _tg(
                f"🛑 <b>SL</b> {symbol} — <b>إغلاق كامل</b>\n"
                f"كمية: <code>{sellable:.6f} {base_asset}</code>\n"
                f"السعر: <code>{exit_px:.6f}</code>\n"
                f"P/L: <b>{pnl_net:.2f}$</b>"
            )
            close_trade(symbol, exit_px, pnl_net, reason="SL")
            _SYMBOL_LAST_TRADE_AT[f"{base}|{variant}"] = now_riyadh()
            return True

    return False

# ================== إغلاق وتسجيل ==================

def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos: return
    closed = load_closed_positions()

    entry = float(pos["entry_price"]); amount = float(pos["amount"]) 
    pnl_pct = ((exit_price / entry) - 1.0) if entry else 0.0

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

    headers = ["الرمز#النسخة", "الكمية", "دخول", "خروج", "P/L$", "P/L%", "Score", "نمط", "سبب"]
    rows = []
    for t in todays:
        rows.append([
            t.get("symbol","-"),
            f"{float(t.get('amount',0)):,.6f}",
            f"{float(t.get('entry_price',0)):,.6f}",
            f"{float(t.get('exit_price',0)):,.6f}",
            f"{float(t.get('profit',0)):,.2f}",
            f"{round(float(t.get('pnl_pct',0))*100,2)}%",
            str(t.get("score","-")),
            t.get("pattern","-"),
            (t.get("entry_reason", t.get('reason','-'))[:40] + ("…" if len(t.get("entry_reason", t.get('reason','')))>40 else "")),
        ])
    table = _fmt_table(rows, headers)

    risk_line = f"وضع المخاطر: {'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'} • "\
                f"اليومي: <b>{s.get('daily_pnl',0.0):.2f}$</b> • "\
                f"متتالية خسائر: <b>{s.get('consecutive_losses',0)}</b> • "\
                f"صفقات اليوم: <b>{s.get('trades_today',0)}</b>"

    summary = (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b>\n"
        f"{risk_line}\n"
    )
    return summary + table
'''

out_path = Path("/mnt/data/strategy_dual_variants_scalp_applied_FINAL_patched.py")
out_path.write_text(patched_code, encoding="utf-8")
out_path.name
