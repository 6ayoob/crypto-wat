# strategy.py — MTF 15m/5m + Pullback/Breakout/Hybrid + ATR/VWAP/RVol/NR + SCALP+ gates (applied)
# ملاحظة مهمة: طُبِّق منطق SCALP+ داخل check_signal فقط كما طلبت.
# - لا إضافات قناة/رسائل جديدة هنا، ولا منطق قفل نسختين.
# - لم نغيّر إدارة الصفقات لديك (هدفان + تريلينغ بعد TP1)، ولا التواقيع العامة للدوال.

import os, json, requests, math
from datetime import datetime, timedelta, timezone
import pandas as pd

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import (
    TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
)

# ================== إعدادات عامة ==================
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# أطر زمنية
HTF_TIMEFRAME = "15m"   # الإطار الأعلى (سياق/اتجاه)
LTF_TIMEFRAME = "5m"    # الإطار الأدنى (تنفيذ)

# أنماط الدخول
ENTRY_MODE    = "hybrid"  # "pullback" أو "breakout" أو "hybrid"
HYBRID_ORDER  = ["pullback", "breakout"]

# مؤشرات أساسية
EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 50
# نطاق RSI العام (يبقى مرجعًا فقط)
RSI_MIN, RSI_MAX = 45, 80
VOL_MA, SR_WINDOW = 20, 50
RESISTANCE_BUFFER, SUPPORT_BUFFER = 0.005, 0.002
ATR_PERIOD = 14

# إعدادات HTF إضافية
HTF_EMA_TREND_PERIOD = 50
HTF_SR_WINDOW = 50
HTF_MIN_RES_BUFFER_ATR_LTF = 1.2  # مسافة أمان للمقاومة (بالـ ATR من LTF)

# سيولة/تذبذب/فلترة جودة على LTF
MIN_DOLLAR_VOLUME_LTF = 60000   # close*volume على الشمعة
RVOL_WINDOW = 20
RVOL_MIN = 1.2
NR_WINDOW = 10
NR_FACTOR = 0.75
PULLBACK_VALUE_REF = "ema21"        # "ema21" أو "vwap"
PULLBACK_CONFIRM   = "bullish_engulf"  # أو "bos"
ATR_MIN_FOR_TREND  = 0.002        # نِسبي/السعر

# ===== نسب ثابتة للأهداف والوقف (افتراضي) =====
STOP_LOSS_PCT = 0.02
TP1_PCT       = 0.03
TP2_PCT       = 0.06
TP1_FRACTION  = 0.5

# ===== بدائل: ATR-Based SL/TP (اختياري) =====
USE_ATR_SL_TP   = False   # فعّل للحصول على SL/TP ديناميكيين
SL_ATR_MULT     = 1.6
TP1_ATR_MULT    = 1.6
TP2_ATR_MULT    = 3.2

# ===== تريلينغ بعد TP1 =====
TRAIL_AFTER_TP1       = True
TRAIL_ATR_MULT        = 1.0   # SL = max(SL, current - ATR*mult, entry*(1+LOCK_MIN_PROFIT_PCT))
LOCK_MIN_PROFIT_PCT   = 0.01
TRAIL_MIN_STEP_RATIO  = 0.001  # لا نحدّث SL إن كانت الزيادة أقل من 0.1%

# ===== حد أقصى لمدة الصفقة =====
USE_MAX_HOLD_HOURS    = True
MAX_HOLD_HOURS        = 12

# ===== Position sizing بالريسك (اختياري) =====
USE_RISK_POSITION_SIZING = False
RISK_PER_TRADE_PCT       = 0.01  # 1% من رصيد USDT
MIN_NOTIONAL_USDT        = 10.0

# ===== حُرّاس المخاطر العامة =====
MAX_TRADES_PER_DAY       = 20
MAX_CONSEC_LOSSES        = 3
DAILY_LOSS_LIMIT_USDT    = 200.0
BLOCK_AFTER_LOSSES_MIN   = 90
SYMBOL_COOLDOWN_MIN      = 30

# نطاقات RSI حسب النمط
RSI_MIN_PULLBACK, RSI_MAX_PULLBACK = 45, 65
RSI_MIN_BREAKOUT, RSI_MAX_BREAKOUT = 50, 80

# تتبُّع أسباب الرفض + وضع تصحيح
DEBUG_LOG_SIGNALS = False
_LAST_REJECT = {}

# ===== إضافات SCALP+ للـ Gate/الدخول =====
RSI_GATE_POLICY = "balanced"   # "lenient" | "balanced" | "strict"
USE_FIB = True
SWING_LOOKBACK = 60
FIB_TOL = 0.004
BREAKOUT_BUFFER_LTF = 0.0015

# ================== Telegram helper (كما هو) ==================
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

# ================== Utils ==================
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

def _df(data):  # OHLCV -> DataFrame
    return pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])

# (مراجَع) أدوات تصحيح

def _reject(symbol, reason):
    _LAST_REJECT[symbol] = reason
    if DEBUG_LOG_SIGNALS:
        _tg(f"🧪 <b>{symbol}</b> رفضت الإشارة: {reason}")
    return None

def get_last_reject_reason(symbol):
    return _LAST_REJECT.get(symbol)

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
        s = _default_risk_state()
        save_risk_state(s)
    return s

def save_risk_state(s): _atomic_write(RISK_STATE_FILE, s)

def register_trade_opened():
    s = load_risk_state()
    s["trades_today"] = int(s.get("trades_today", 0)) + 1
    save_risk_state(s)

def _set_block(minutes, reason="risk"):
    s = load_risk_state()
    until = now_riyadh() + timedelta(minutes=minutes)
    s["blocked_until"] = until.isoformat(timespec="seconds")
    save_risk_state(s)
    _tg(f"⛔️ <b>تم تفعيل حظر مؤقت</b> ({reason}) حتى <code>{until.strftime('%H:%M')}</code>.")

def _clear_block():
    s = load_risk_state()
    s["blocked_until"] = None
    save_risk_state(s)

def _is_blocked():
    s = load_risk_state()
    bu = s.get("blocked_until")
    if not bu: return False
    try:
        t = datetime.fromisoformat(bu)
    except Exception:
        return False
    return now_riyadh() < t

def register_trade_result(pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1

    if s["consecutive_losses"] >= MAX_CONSEC_LOSSES:
        save_risk_state(s)
        _set_block(BLOCK_AFTER_LOSSES_MIN, reason="خسائر متتالية")
        return
    if s["daily_pnl"] <= -abs(DAILY_LOSS_LIMIT_USDT):
        end_of_day = now_riyadh().replace(hour=23, minute=59, second=0, microsecond=0)
        minutes = max(1, int((end_of_day - now_riyadh()).total_seconds() // 60))
        save_risk_state(s)
        _set_block(minutes, reason="تجاوز حد الخسارة اليومي")
        return

    save_risk_state(s)

def _risk_precheck_allow_new_entry():
    if _is_blocked():  return False, "blocked"
    s = load_risk_state()
    if MAX_TRADES_PER_DAY and s.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
        return False, "max_trades"
    if s.get("daily_pnl", 0.0) <= -abs(DAILY_LOSS_LIMIT_USDT):
        return False, "daily_loss_limit"
    if s.get("consecutive_losses", 0) >= MAX_CONSEC_LOSSES:
        return False, "consec_losses"
    return True, ""

# ================== مؤشرات ==================

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_cols(df, fast=12, slow=26, signal=9):
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]  # إضافة للهستوجرام
    return df

def add_indicators(df):
    df["ema9"]  = ema(df["close"], EMA_FAST)
    df["ema21"] = ema(df["close"], EMA_SLOW)
    df["ema50"] = ema(df["close"], EMA_TREND)
    df["rsi"]   = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA).mean()
    df = macd_cols(df)
    return df

# VWAP/RVol/NR على LTF

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
    rng = df["high"] - df["low"]
    rng_ma = rng.rolling(NR_WINDOW).mean()
    df["is_nr"] = rng < (NR_FACTOR * rng_ma)
    df["body"] = (df["close"] - df["open"]).abs()
    df["avg_body20"] = df["body"].rolling(20).mean()
    return df

# ===== ATR =====

def _atr_from_df(df, period=ATR_PERIOD):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-2])

def _get_atr(symbol, tf, lookback=140, period=ATR_PERIOD):
    data = fetch_ohlcv(symbol, tf, lookback)
    if not data: return None
    df = _df(data)
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    if len(atr) < 5: return None
    return float(atr.iloc[-2])

# ===== Swing/SR/Fib (جديد) =====

def _swing_points(df, left=2, right=2):
    highs = df["high"]; lows = df["low"]
    idx = len(df) - 3
    swing_high = None; swing_low = None
    for i in range(max(0, idx-10), idx+1):
        if i-left < 0 or i+right >= len(df): continue
        if highs[i] == max(highs[i-left:i+right+1]):
            swing_high = float(highs[i])
        if lows[i] == min(lows[i-left:i+right+1]):
            swing_low = float(lows[i])
    return swing_high, swing_low

def _bullish_engulf(prev, cur):
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and \
           (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])

def get_sr_on_closed(df, window=SR_WINDOW):
    if len(df) < window + 3:
        return None, None
    df_prev = df.iloc[:-1]
    w = min(window, len(df_prev))
    resistance = df_prev["high"].rolling(w, min_periods=max(5, w//3)).max().iloc[-1]
    support    = df_prev["low"].rolling(w,  min_periods=max(5, w//3)).min().iloc[-1]
    if pd.isna(resistance) or pd.isna(support):
        return None, None
    return float(support), float(resistance)

def recent_swing(df, lookback=SWING_LOOKBACK):
    if len(df) < lookback + 5:
        return None, None
    seg = df.iloc[-(lookback+1):-1]
    hhv = seg["high"].max()
    llv = seg["low"].min()
    if pd.isna(hhv) or pd.isna(llv) or hhv <= llv:
        return None, None
    return float(hhv), float(llv)

def near_any_fib(price: float, hhv: float, llv: float, tol: float):
    rng = hhv - llv
    if rng <= 0: return False, ""
    fib382 = hhv - rng * 0.382
    fib618 = hhv - rng * 0.618
    for lvl, name in ((fib382, "Fib 0.382"), (fib618, "Fib 0.618")):
        if abs(price - lvl) / max(lvl, 1e-9) <= tol:
            return True, name
    return False, ""

# ===== MACD/RSI Gate (جديد) =====

def macd_rsi_gate(prev_row, closed_row, policy=RSI_GATE_POLICY):
    rsi_now = float(closed_row.get("rsi", 50.0))
    rsi_up  = rsi_now > float(prev_row.get("rsi", rsi_now))
    macd_h_now  = float(closed_row.get("macd_hist", 0.0))
    macd_h_prev = float(prev_row.get("macd_hist", 0.0))
    macd_pos    = macd_h_now > 0
    macd_up     = macd_h_now > macd_h_prev

    ok_flags = []
    if rsi_now > 50: ok_flags.append("RSI>50")
    if rsi_up:       ok_flags.append("RSI↑")
    if macd_pos:     ok_flags.append("MACD_hist>0")
    if macd_up:      ok_flags.append("MACD_hist↑")

    k = len(ok_flags)
    if policy == "lenient":
        ok = k >= 1
    elif policy == "strict":
        ok = ("RSI>50" in ok_flags) and ("MACD_hist>0" in ok_flags) and ("MACD_hist↑" in ok_flags)
    else:  # balanced
        ok = k >= 2

    return ok

# ================== سياق HTF ==================

def _get_htf_context(symbol):
    data = fetch_ohlcv(symbol, HTF_TIMEFRAME, 200)
    if not data: return None
    df = _df(data)
    df["ema50_htf"] = ema(df["close"], HTF_EMA_TREND_PERIOD)
    if len(df) < HTF_SR_WINDOW + 3: return None
    df_prev = df.iloc[:-2]
    w = min(HTF_SR_WINDOW, len(df_prev))
    resistance = df_prev["high"].rolling(w).max().iloc[-1]
    support    = df_prev["low"].rolling(w).min().iloc[-1]
    closed = df.iloc[-2]
    ema_now  = float(closed["ema50_htf"])
    ema_prev = float(df["ema50_htf"].iloc[-7]) if len(df) >= 7 else ema_now
    return {"close": float(closed["close"]),
            "ema50_now": ema_now, "ema50_prev": ema_prev,
            "support": float(support), "resistance": float(resistance)}

_LAST_ENTRY_BAR_TS = {}
_SYMBOL_LAST_TRADE_AT = {}

# ================== أنماط الدخول (كما هي) ==================

def _entry_pullback_logic(df, closed, prev, atr_ltf, htf_ctx):
    ref_val = closed["ema21"] if PULLBACK_VALUE_REF=="ema21" else closed.get("vwap", closed["ema21"])
    if pd.isna(ref_val): ref_val = closed["ema21"]
    near_val = (closed["close"] >= ref_val) and (closed["low"] <= ref_val)
    if not near_val: return False
    if PULLBACK_CONFIRM == "bullish_engulf":
        return _bullish_engulf(prev, closed)
    elif PULLBACK_CONFIRM == "bos":
        swing_high, _ = _swing_points(df)
        return bool(swing_high and closed["close"] > swing_high)
    return True


def _entry_breakout_logic(df, closed, prev, atr_ltf, htf_ctx):
    hi_range = float(df["high"].iloc[-NR_WINDOW-2:-2].max())
    is_nr_recent = bool(df["is_nr"].iloc[-3:-1].all())
    vwap_ok = closed["close"] > float(closed.get("vwap", closed["ema21"]))
    return (closed["close"] > hi_range) and is_nr_recent and vwap_ok

# ================== فحص الإشارة (مطوَّر) ==================

def check_signal(symbol):
    ok, _ = _risk_precheck_allow_new_entry()
    if not ok:
        return _reject(symbol, "حظر/قيود مخاطرة")

    # تبريد لكل رمز
    last_t = _SYMBOL_LAST_TRADE_AT.get(symbol)
    if last_t and (now_riyadh() - last_t) < timedelta(minutes=SYMBOL_COOLDOWN_MIN):
        return _reject(symbol, "Cooldown")

    # لا نعيد الإشارة إن كانت هناك صفقة مفتوحة بالفعل
    if load_position(symbol):
        return _reject(symbol, "مركز مفتوح")

    # سياق HTF
    ctx = _get_htf_context(symbol)
    if not ctx: return _reject(symbol, "HTF غير متاح")
    ema50_slope_up_htf = (ctx["ema50_now"] - ctx["ema50_prev"]) > 0
    if not (ema50_slope_up_htf and ctx["close"] > ctx["ema50_now"]):
        return _reject(symbol, "اتجاه HTF غير مؤكد")

    # بيانات LTF
    data = fetch_ohlcv(symbol, LTF_TIMEFRAME, 260)
    if not data: return _reject(symbol, "LTF OHLCV غير متاح")
    df = _df(data); df = _ensure_ltf_indicators(df)
    if len(df) < 120: return _reject(symbol, "بيانات قليلة")

    prev, closed = df.iloc[-3], df.iloc[-2]
    last_ts_closed = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(symbol) == last_ts_closed:
        return _reject(symbol, "نفس الشمعة")

    price = float(closed["close"])
    atr_ltf = _atr_from_df(df)
    if not atr_ltf or atr_ltf <= 0: return _reject(symbol, "ATR غير صالح")
    if (atr_ltf / max(1e-9, price)) < ATR_MIN_FOR_TREND:
        return _reject(symbol, "ATR نسبي ضعيف")

    # مسافة من EMA50 LTF — بين 0.5 و 3 ATR
    dist = price - float(closed["ema50"])
    if dist < 0.5 * atr_ltf:
        return _reject(symbol, "قرب شديد من EMA50 (<0.5 ATR)")
    if dist > 3.0 * atr_ltf:
        return _reject(symbol, "تمدد مبالغ فيه (>3 ATR)")

    # جودة: سيولة/RVOL/شمعة خضراء
    if (closed["close"] * closed["volume"]) < MIN_DOLLAR_VOLUME_LTF: 
        return _reject(symbol, "سيولة ضعيفة")
    if pd.isna(closed.get("rvol")) or closed["rvol"] < RVOL_MIN:
        return _reject(symbol, "RVol < الحد")
    if closed["close"] <= closed["open"]:
        return _reject(symbol, "شمعة غير صاعدة")

    # قُرب HTF SR
    if ctx.get("resistance") and (ctx["resistance"] - price) < HTF_MIN_RES_BUFFER_ATR_LTF * atr_ltf:
        return _reject(symbol, "قرب مقاومة HTF")
    if ctx.get("support") and price <= ctx["support"] * (1 + SUPPORT_BUFFER):
        return _reject(symbol, "قرب دعم HTF (اختراق كاذب)")

    # ===== بوابة MACD/RSI (SCALP+) =====
    if not macd_rsi_gate(prev, closed, policy=RSI_GATE_POLICY):
        return _reject(symbol, "بوابة MACD/RSI")

    # ===== اختيار نمط الدخول الأساسي =====
    chosen_mode = None
    mode_ok = False
    if ENTRY_MODE == "pullback":
        chosen_mode = "pullback"
        mode_ok = _entry_pullback_logic(df, closed, prev, atr_ltf, ctx)
    elif ENTRY_MODE == "breakout":
        chosen_mode = "breakout"
        mode_ok = _entry_breakout_logic(df, closed, prev, atr_ltf, ctx)
    elif ENTRY_MODE == "hybrid":
        for m in HYBRID_ORDER:
            if m == "pullback" and _entry_pullback_logic(df, closed, prev, atr_ltf, ctx):
                chosen_mode = "pullback"; mode_ok = True; break
            if m == "breakout" and _entry_breakout_logic(df, closed, prev, atr_ltf, ctx):
                chosen_mode = "breakout"; mode_ok = True; break
    else:
        crossed = (prev["ema9"] < prev["ema21"]) and (closed["ema9"] > closed["ema21"])
        macd_ok = float(df["macd"].iloc[-2]) > float(df["macd_signal"].iloc[-2])
        chosen_mode = "crossover"
        mode_ok = crossed and macd_ok

    # ===== تحسينات دخول SCALP+ (بدائل دقيقة) =====
    if not mode_ok:
        # (A) Breakout SR على LTF بنطاق SR_WINDOW + هامش صغير وتجنّب الدخول داخل كتلة المقاومة
        sup_ltf, res_ltf = get_sr_on_closed(df, SR_WINDOW)
        try:
            hhv = float(df.iloc[:-1]["high"].rolling(SR_WINDOW, min_periods=10).max().iloc[-1])
        except Exception:
            hhv = None
        if hhv:
            breakout_ok = price > hhv * (1.0 + BREAKOUT_BUFFER_LTF)
            near_res_block = (res_ltf is not None) and (res_ltf * (1 - RESISTANCE_BUFFER) <= price <= res_ltf * (1 + RESISTANCE_BUFFER))
            if breakout_ok and not near_res_block:
                chosen_mode = chosen_mode or "breakout"
                mode_ok = True
        # (B) Pullback Fib 0.382–0.618 بشرط تحسّن RSI أو MACD_hist وعدم وجود ضغط دعم مباشر
        if not mode_ok and USE_FIB:
            hhv2, llv2 = recent_swing(df, SWING_LOOKBACK)
            if hhv2 and llv2:
                near_fib, _ = near_any_fib(price, hhv2, llv2, FIB_TOL)
                sup_block = (sup_ltf is not None) and (price <= sup_ltf * (1 + SUPPORT_BUFFER))
                momentum_up = (float(closed["rsi"]) > float(prev["rsi"])) or (float(closed.get("macd_hist",0)) > float(prev.get("macd_hist",0)))
                if near_fib and (not sup_block) and momentum_up:
                    chosen_mode = chosen_mode or "pullback"
                    mode_ok = True

    if not mode_ok:
        return _reject(symbol, "شروط نمط الدخول")

    # فلتر RSI بحسب النمط (كما هو)
    rsi_val = float(closed["rsi"])
    if chosen_mode == "pullback" and not (RSI_MIN_PULLBACK < rsi_val < RSI_MAX_PULLBACK):
        return _reject(symbol, f"RSI({rsi_val:.1f}) خارج نطاق pullback")
    if chosen_mode == "breakout" and not (RSI_MIN_BREAKOUT < rsi_val < RSI_MAX_BREAKOUT):
        return _reject(symbol, f"RSI({rsi_val:.1f}) خارج نطاق breakout")

    _LAST_ENTRY_BAR_TS[symbol] = last_ts_closed
    return "buy"

# ================== حساب SL/TP (كما هو) ==================

def _compute_sl_tp(entry, atr_val=None):
    if USE_ATR_SL_TP and atr_val and atr_val > 0:
        sl  = entry - SL_ATR_MULT  * atr_val
        tp1 = entry + TP1_ATR_MULT * atr_val
        tp2 = entry + TP2_ATR_MULT * atr_val
    else:
        sl  = entry * (1 - STOP_LOSS_PCT)
        tp1 = entry * (1 + TP1_PCT)
        tp2 = entry * (1 + TP2_PCT)
    return float(sl), float(tp1), float(tp2)

# ================== تنفيذ الشراء (كما هو) ==================

def execute_buy(symbol):
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."
    if _is_blocked():
        return None, "🚫 ممنوع فتح صفقات الآن (حظر مخاطرة)."
    if load_position(symbol):
        return None, "🚫 لديك صفقة مفتوحة على هذا الرمز."

    try:
        price_raw = fetch_price(symbol)
        price = float(price_raw)
    except Exception:
        return None, "⚠️ فشل جلب السعر."
    if price <= 0: return None, "⚠️ سعر غير صالح."

    atr_val = _get_atr(symbol, LTF_TIMEFRAME)
    sl_tmp, tp1_tmp, tp2_tmp = _compute_sl_tp(price, atr_val=atr_val)

    usdt = float(fetch_balance("USDT") or 0)
    if usdt < MIN_NOTIONAL_USDT:
        return None, "🚫 رصيد USDT غير كافٍ."

    if USE_RISK_POSITION_SIZING and atr_val:
        risk_usdt = float(fetch_balance("USDT") or 0) * RISK_PER_TRADE_PCT
        per_unit_risk = max(1e-9, price - sl_tmp)
        qty_by_risk = risk_usdt / per_unit_risk
        cost = qty_by_risk * price
        if cost > usdt:
            qty_by_risk = (usdt / price) * 0.98
        amount = max(0.0, qty_by_risk)
    else:
        if usdt < TRADE_AMOUNT_USDT:
            return None, "🚫 رصيد USDT غير كافٍ."
        amount = TRADE_AMOUNT_USDT / price

    if amount * price < MIN_NOTIONAL_USDT:
        return None, "🚫 قيمة الصفقة أقل من الحد الأدنى."

    order = place_market_order(symbol, "buy", amount)
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    try:
        fill_px = float(order.get("average") or order.get("price") or price)
        price = fill_px if fill_px > 0 else price
    except Exception:
        pass

    sl, tp1, tp2 = _compute_sl_tp(price, atr_val=atr_val)

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(price),
        "stop_loss": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "partial_done": False,
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "atr_on_entry": float(atr_val or 0.0)
    }
    save_position(symbol, pos)
    _SYMBOL_LAST_TRADE_AT[symbol] = now_riyadh()

    _tg(
        f"✅ <b>دخول BUY</b> {symbol}\n"
        f"قيمة الشراء: <b>{amount*price:,.2f}$</b>\n"
        f"الكمية: <code>{amount:.6f}</code>\n"
        f"الدخول: <code>{price:.6f}</code>\n"
        f"SL: <code>{sl:.6f}</code>\n"
        f"TP1 ({int((tp1/price-1)*100)}%): <code>{tp1:.6f}</code>\n"
        f"TP2 ({int((tp2/price-1)*100)}%): <code>{tp2:.6f}</code>"
    )

    register_trade_opened()
    return order, f"✅ شراء {symbol} | SL: {sl:.6f} | TP1: {tp1:.6f} | TP2: {tp2:.6f}"

# ================== إدارة الصفقة (كما هي) ==================

def manage_position(symbol):
    pos = load_position(symbol)
    if not pos: return False

    current = float(fetch_price(symbol))
    entry   = float(pos["entry_price"])
    sl      = float(pos["stop_loss"])
    tp1     = float(pos["tp1"])
    tp2     = float(pos["tp2"])
    amount  = float(pos["amount"])

    if amount <= 0:
        clear_position(symbol)
        return False

    base_asset = symbol.split("/")[0]
    wallet_balance = float(fetch_balance(base_asset) or 0)
    if wallet_balance <= 0:
        _tg(f"⚠️ لا يوجد رصيد {base_asset} للبيع في {symbol} — سيتم إغلاق المركز محليًا.")
        clear_position(symbol)
        return False

    sellable = min(amount, wallet_balance)

    # إغلاق زمني اختياري
    if USE_MAX_HOLD_HOURS:
        try:
            opened_at = datetime.fromisoformat(pos.get("opened_at"))
            if now_riyadh() - opened_at > timedelta(hours=MAX_HOLD_HOURS):
                order = place_market_order(symbol, "sell", sellable)
                if order:
                    exit_px = float(order.get("average") or order.get("price") or current)
                    pnl_gross = (exit_px - entry) * sellable
                    fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
                    pnl_net = pnl_gross - fees
                    _tg(f"⌛️ <b>إغلاق زمني</b> {symbol} @ <code>{exit_px:.6f}</code> • P/L: <b>{pnl_net:.2f}$</b>")
                    close_trade(symbol, exit_px, pnl_net, reason="TIME")
                    _SYMBOL_LAST_TRADE_AT[symbol] = now_riyadh()
                    return True
        except Exception:
            pass

    # --- TP1: بيع 50% + SL = التعادل ---
    if (not pos.get("partial_done")) and current >= tp1 and sellable > 0:
        part_qty = sellable * TP1_FRACTION
        if part_qty * current < MIN_NOTIONAL_USDT:
            part_qty = sellable
        order = place_market_order(symbol, "sell", part_qty)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * part_qty
            fees = (entry + exit_px) * part_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees

            pos["amount"] = float(max(0.0, amount - part_qty))
            pos["partial_done"] = True
            pos["stop_loss"] = float(max(entry, sl))  # نقل SL للتعادل
            save_position(symbol, pos)
            register_trade_result(pnl_net)

            _tg(
                f"🎯 <b>TP1 تحقق</b> {symbol}\n"
                f"تم بيع: <code>{part_qty:.6f} {base_asset}</code>\n"
                f"السعر: <code>{exit_px:.6f}</code>\n"
                f"SL ← التعادل."
            )

    # تحديث بعد TP1
    pos_ref = load_position(symbol)
    if not pos_ref:
        return True
    amount = float(pos_ref.get("amount", 0.0))
    wallet_balance = float(fetch_balance(base_asset) or 0)
    sellable = min(amount, wallet_balance)
    sl = float(pos_ref.get("stop_loss", sl))

    # --- تريلينغ بعد TP1 بالـ ATR ---
    if TRAIL_AFTER_TP1 and pos_ref.get("partial_done") and sellable > 0:
        atr_val = _get_atr(symbol, LTF_TIMEFRAME)
        if atr_val and atr_val > 0:
            new_sl_atr = current - TRAIL_ATR_MULT * atr_val
            new_sl = max(sl, new_sl_atr, entry * (1 + LOCK_MIN_PROFIT_PCT))
            if new_sl > sl * (1 + TRAIL_MIN_STEP_RATIO):
                pos_ref["stop_loss"] = float(new_sl)
                save_position(symbol, pos_ref)
                _tg(f"🧭 <b>Trailing SL</b> {symbol} → <code>{new_sl:.6f}</code>")

    # --- TP2: إغلاق كامل ---
    if sellable > 0 and current >= tp2:
        order = place_market_order(symbol, "sell", sellable)
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
            _SYMBOL_LAST_TRADE_AT[symbol] = now_riyadh()
            return True

    # --- SL: إغلاق كامل ---
    pos_ref = load_position(symbol)
    if not pos_ref:
        return True
    amount = float(pos_ref.get("amount", 0.0))
    wallet_balance = float(fetch_balance(base_asset) or 0)
    sellable = min(amount, wallet_balance)
    sl = float(pos_ref.get("stop_loss", sl))

    if sellable > 0 and current <= sl:
        order = place_market_order(symbol, "sell", sellable)
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
            _SYMBOL_LAST_TRADE_AT[symbol] = now_riyadh()
            return True

    return False

# ================== إغلاق وتسجيل (كما هو) ==================

def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos: return
    closed = load_closed_positions()

    entry = float(pos["entry_price"])
    amount = float(pos["amount"])
    pnl_pct = ((exit_price / entry) - 1.0) if entry else 0.0

    closed.append({
        "symbol": symbol,
        "entry_price": entry,
        "exit_price": float(exit_price),
        "amount": amount,
        "profit": float(pnl_net),
        "pnl_pct": round(pnl_pct, 6),
        "reason": reason,
        "opened_at": pos.get("opened_at"),
        "closed_at": now_riyadh().isoformat(timespec="seconds"),
    })
    save_closed_positions(closed)
    register_trade_result(pnl_net)
    clear_position(symbol)

# ================== تقرير يومي (كما هو) ==================

def _fmt_table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "<pre>" + fmt_row(headers) + "\n" + "\n".join(fmt_row(r) for r in rows) + "</pre>"

def build_daily_report_text():
    closed = load_closed_positions()
    today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    s = load_risk_state()

    if not todays:
        extra = f"\nوضع المخاطر: {'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'} • صفقات اليوم: {s.get('trades_today',0)} • PnL اليومي: {s.get('daily_pnl',0.0):.2f}$"
        return f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم.{extra}"

    total_pnl = sum(float(t.get("profit", 0.0)) for t in todays)
    wins = [t for t in todays if float(t.get("profit", 0.0)) > 0]
    losses = [t for t in todays if float(t.get("profit", 0.0)) <= 0]
    win_rate = round(100 * len(wins) / max(1, len(todays)), 2)

    best = max(todays, key=lambda t: float(t.get("profit", 0.0)))
    worst = min(todays, key=lambda t: float(t.get("profit", 0.0)))

    headers = ["الرمز", "الكمية", "دخول", "خروج", "P/L$", "P/L%"]
    rows = []
    for t in todays:
        rows.append([
            t.get("symbol","-"),
            f"{float(t.get('amount',0)):,.6f}",
            f"{float(t.get('entry_price',0)):,.6f}",
            f"{float(t.get('exit_price',0)):,.6f}",
            f"{float(t.get('profit',0)):,.2f}",
            f"{round(float(t.get('pnl_pct',0))*100,2)}%",
        ])
    table = _fmt_table(rows, headers)

    risk_line = f"وضع المخاطر: {'محظور حتى ' + s.get('blocked_until') if s.get('blocked_until') else 'سماح'} • "\
                f"اليومي: <b>{s.get('daily_pnl',0.0):.2f}$</b> • "\
                f"متتالية خسائر: <b>{s.get('consecutive_losses',0)}</b> • "\
                f"صفقات اليوم: <b>{s.get('trades_today',0)}</b>"

    summary = (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b> • الرابحة: <b>{len(wins)}</b> • الخاسرة: <b>{len(losses)}</b>\n"
        f"{risk_line}\n"
    )
    return summary + table
