# strategy.py — 80/100 edition
import os, json, time, math, threading
from datetime import datetime, timedelta, timezone

import pandas as pd

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS

# ========= إعدادات عامة =========
RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# ========= إعدادات الإشارة/المخاطر =========
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
RSI_MIN, RSI_MAX = 50, 70
VOL_MA = 20
SR_WINDOW = 50
RESISTANCE_BUFFER = 0.005  # 0.5%
SUPPORT_BUFFER    = 0.002  # 0.2%

# تعدد الأطر
REQUIRE_MTF = True  # شرط السعر فوق EMA50 على 15m أيضاً

# ATR
ATR_PERIOD = 14
ATR_SL_MULT = 1.5
ATR_TRAIL_MULT = 1.0
R_MULT_TP = 2.0       # الهدف = R×2 (حيث R = المخاطرة = entry - SL)
PARTIAL_FRACTION = 0.5

# الرسوم (جولة كاملة round-trip) بالـ bps
FEE_BPS_ROUNDTRIP = 8  # 0.08% تقريباً

# حماية يومية (كما كانت)
DAILY_MAX_LOSS_USDT = 50
MAX_CONSECUTIVE_LOSSES = 3
MAX_TRADES_PER_DAY = 10
COOLDOWN_MINUTES_AFTER_HALT = 120

# منع التكرار على نفس الشمعة المغلقة
_LAST_ENTRY_BAR_TS = {}  # {symbol: last_closed_ts_used}

# ========= أدوات عامة / IO =========
def now_riyadh():
    return datetime.now(RIYADH_TZ)

def ensure_dirs():
    os.makedirs(POSITIONS_DIR, exist_ok=True)

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

def _today_str():
    return now_riyadh().strftime("%Y-%m-%d")

# ========= تخزين الصفقات =========
def _pos_path(symbol):
    ensure_dirs()
    return f"{POSITIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol):
    return _read_json(_pos_path(symbol), None)

def save_position(symbol, position):
    _atomic_write(_pos_path(symbol), position)

def clear_position(symbol):
    try:
        p = _pos_path(symbol)
        if os.path.exists(p):
            os.remove(p)
    except:
        pass

def count_open_positions():
    ensure_dirs()
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions():
    return _read_json(CLOSED_POSITIONS_FILE, [])

def save_closed_positions(lst):
    _atomic_write(CLOSED_POSITIONS_FILE, lst)

# ========= حالة المخاطر اليومية =========
def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0, "trades_today": 0, "blocked_until": None}

def load_risk_state():
    s = _read_json(RISK_STATE_FILE, _default_risk_state())
    if s.get("date") != _today_str():
        s = _default_risk_state()
        save_risk_state(s)
    return s

def save_risk_state(s):
    _atomic_write(RISK_STATE_FILE, s)

def register_trade_opened():
    s = load_risk_state()
    s["trades_today"] = int(s.get("trades_today", 0)) + 1
    save_risk_state(s)

def register_trade_result(pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1
    save_risk_state(s)

def is_trading_blocked():
    s = load_risk_state()
    if s.get("blocked_until"):
        try:
            until = datetime.fromisoformat(s["blocked_until"])
            if until.tzinfo is None:
                until = until.replace(tzinfo=RIYADH_TZ)
            if now_riyadh() < until:
                return True, f"⏸️ التداول موقوف حتى {until.isoformat()}."
        except:
            pass
    if s.get("daily_pnl", 0.0) <= -abs(DAILY_MAX_LOSS_USDT):
        return True, "⛔ حد الخسارة اليومية متجاوز."
    if s.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
        return True, "⛔ خسائر متتالية متجاوزة."
    if s.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
        return True, "⛔ حد الصفقات اليومية متجاوز."
    return False, ""

def trigger_cooldown():
    s = load_risk_state()
    until = now_riyadh() + timedelta(minutes=COOLDOWN_MINUTES_AFTER_HALT)
    s["blocked_until"] = until.isoformat()
    save_risk_state(s)

# ========= مؤشرات =========
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

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
    return df

def atr_series(df, period=14):
    c = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - c).abs(),
        (df["low"] - c).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def add_indicators(df):
    df["ema9"]  = ema(df["close"], EMA_FAST)
    df["ema21"] = ema(df["close"], EMA_SLOW)
    df["ema50"] = ema(df["close"], EMA_TREND)
    df["rsi"]   = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA).mean()
    df = macd_cols(df)
    df["atr"] = atr_series(df, ATR_PERIOD)
    return df

# ========= دعم/مقاومة =========
def get_support_resistance_on_closed(df, window=50):
    # استخدم البيانات حتى الشمعة المغلقة الأخيرة (استبعد الحالية)
    if len(df) < window + 3:
        return None, None
    df_prev = df.iloc[:-2]
    w = min(window, len(df_prev))
    resistance = df_prev["high"].rolling(w).max().iloc[-1]
    support    = df_prev["low"].rolling(w).min().iloc[-1]
    return support, resistance

# ========= مُساعدات DF =========
def _df_from_ohlcv(data):
    # توقع الأعمدة: [ts, open, high, low, close, volume]
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    return df

def _get_df(interval, limit):
    data = fetch_ohlcv(None, interval, limit)  # بعض رَوابِطك ربما تتجاهل الرمز هنا؛ سنستخدم لكل رمز لاحقاً
    # ملاحظة: سنستدعي fetch_ohlcv(symbol, ...) مباشرة حيث نحتاج الرمز
    return _df_from_ohlcv(data)

# ========= فحص الإشارة (5m + 15m) =========
def check_signal(symbol):
    blocked, msg = is_trading_blocked()
    if blocked:
        print(msg)
        return None

    data5 = fetch_ohlcv(symbol, "5m", 200)
    if not data5:
        return None
    df5 = _df_from_ohlcv(data5)
    df5 = add_indicators(df5)
    if len(df5) < 60:
        return None

    # نستخدم الشمعة المغلقة الأخيرة وما قبلها
    prev = df5.iloc[-3]
    closed = df5.iloc[-2]   # الشمعة المكتملة
    last_ts_closed = int(df5.iloc[-2]["timestamp"])

    # امنع التكرار على نفس الشمعة المغلقة
    if _LAST_ENTRY_BAR_TS.get(symbol) == last_ts_closed:
        return None

    # فلاتر الحجم والشمعة صاعدة
    if not pd.isna(closed["vol_ma20"]) and closed["volume"] < closed["vol_ma20"]:
        return None
    if closed["close"] <= closed["open"]:
        return None

    # اتجاه: فوق EMA50 على 5m
    if closed["close"] < closed["ema50"]:
        return None

    # RSI معتدل
    if not (RSI_MIN < closed["rsi"] < RSI_MAX):
        return None

    # تقاطع EMA9/21 تأكيدي + MACD
    crossed = prev["ema9"] < prev["ema21"] and closed["ema9"] > closed["ema21"]
    macd_ok = closed["macd"] > closed["macd_signal"]
    if not (crossed and macd_ok):
        return None

    # دعم/مقاومة (على بيانات مغلقة فقط)
    support, resistance = get_support_resistance_on_closed(df5, SR_WINDOW)
    price = float(closed["close"])
    if support and resistance:
        if price >= resistance * (1 - RESISTANCE_BUFFER):
            return None
        if price <= support * (1 + SUPPORT_BUFFER):
            return None

    # فلتر متعدد الأطر (15m): أيضاً فوق EMA50 للشمعة المغلقة على 15m
    if REQUIRE_MTF:
        data15 = fetch_ohlcv(symbol, "15m", 150)
        if not data15:
            return None
        df15 = _df_from_ohlcv(data15)
        df15["ema50"] = ema(df15["close"], EMA_TREND)
        if len(df15) < 60:
            return None
        closed15 = df15.iloc[-2]
        if closed15["close"] < closed15["ema50"]:
            return None

    # مرّت كل الفلاتر
    _LAST_ENTRY_BAR_TS[symbol] = last_ts_closed
    return "buy"

# ========= تنفيذ الشراء =========
def execute_buy(symbol):
    blocked, msg = is_trading_blocked()
    if blocked:
        return None, msg

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."

    price = float(fetch_price(symbol))
    usdt = float(fetch_balance("USDT"))
    if usdt < TRADE_AMOUNT_USDT:
        return None, "🚫 رصيد USDT غير كافٍ."

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, "buy", amount)
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    # نحسب ATR للوقف/الهدف على 5m (شمعة مغلقة)
    data5 = fetch_ohlcv(symbol, "5m", 100)
    df5 = _df_from_ohlcv(data5)
    df5 = add_indicators(df5)
    atr = float(df5["atr"].iloc[-2])  # ATR للشمعة المغلقة

    sl = price - ATR_SL_MULT * atr
    r = price - sl
    tp = price + R_MULT_TP * r

    position = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(price),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "trailing_stop": float(price - ATR_TRAIL_MULT * atr),
        "atr": float(atr),
        "atr_period": ATR_PERIOD,
        "atr_sl_mult": ATR_SL_MULT,
        "atr_trail_mult": ATR_TRAIL_MULT,
        "r_mult_tp": R_MULT_TP,
        "partial_done": False,
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
        "entry_bar_ts": int(df5.iloc[-2]["timestamp"]) if len(df5) >= 2 else None,
    }
    save_position(symbol, position)
    register_trade_opened()
    return order, f"✅ تم شراء {symbol} | SL(ATR): {sl:.6f} | TP: {tp:.6f}"

# ========= إدارة الصفقة =========
def manage_position(symbol):
    pos = load_position(symbol)
    if not pos:
        return False

    current = float(fetch_price(symbol))
    entry = float(pos["entry_price"])
    amount = float(pos["amount"])

    # ATR حديث للشمعة المغلقة
    data5 = fetch_ohlcv(symbol, "5m", 50)
    df5 = _df_from_ohlcv(data5)
    df5["atr"] = atr_series(df5, ATR_PERIOD)
    atr = float(df5["atr"].iloc[-2])

    # تحديث trailing الديناميكي (رفع فقط)
    trail_level = current - pos.get("atr_trail_mult", ATR_TRAIL_MULT) * atr
    new_trailing = max(float(pos["trailing_stop"]), float(trail_level))
    if new_trailing > pos["trailing_stop"]:
        pos["trailing_stop"] = float(new_trailing)
        save_position(symbol, pos)

    # جني جزئي عند 1×R (منتصف الطريق نحو TP لأن TP=2R)
    half_target = entry + (pos["take_profit"] - entry) / 2
    closed_any = False

    if (not pos.get("partial_done")) and current >= half_target and amount > 0:
        sell_amount = amount * PARTIAL_FRACTION
        order = place_market_order(symbol, "sell", sell_amount)
        if order:
            # PnL الصافي بعد الرسوم (للنصف المباع)
            pnl_gross = (current - entry) * sell_amount
            fees = (entry + current) * sell_amount * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees

            pos["amount"] = float(max(0.0, amount - sell_amount))
            pos["partial_done"] = True
            save_position(symbol, pos)
            register_trade_result(pnl_net)

    # إغلاق كامل: TP أو SL أو Trailing
    amount = float(pos["amount"])
    if amount <= 0:
        return False

    reason = None
    if current >= pos["take_profit"]:
        reason = "TP"
    elif current <= pos["stop_loss"]:
        reason = "SL"
    elif current <= pos["trailing_stop"]:
        reason = "TRAIL"

    if reason:
        order = place_market_order(symbol, "sell", amount)
        if order:
            pnl_gross = (current - entry) * amount
            fees = (entry + current) * amount * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            close_trade(symbol, exit_price=current, pnl_net=pnl_net, reason=reason)
            closed_any = True

            # تهدئة إذا كَثُرت الخسائر المتتالية
            s = load_risk_state()
            if s.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
                # نوقف فتح صفقات لفترة
                until = now_riyadh() + timedelta(minutes=COOLDOWN_MINUTES_AFTER_HALT)
                s["blocked_until"] = until.isoformat()
                save_risk_state(s)

    return closed_any

# ========= إغلاق وتسجيل =========
def close_trade(symbol, exit_price, pnl_net, reason="MANUAL"):
    pos = load_position(symbol)
    if not pos:
        return
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
        "closed_at": now_riyadh().isoformat(timespec="seconds")
    })
    save_closed_positions(closed)
    register_trade_result(pnl_net)
    clear_position(symbol)
