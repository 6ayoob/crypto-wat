# strategy.py — 80/100 + Daily Report Builder (Updated TP ladder & fill price)
import os, json
from datetime import datetime, timedelta, timezone
import pandas as pd

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP

RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# إعدادات المؤشرات
EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 50
RSI_MIN, RSI_MAX = 50, 70
VOL_MA, SR_WINDOW = 20, 50
RESISTANCE_BUFFER, SUPPORT_BUFFER = 0.005, 0.002

# تعدد الأطر
REQUIRE_MTF = True

# ATR / إدارة المخاطرة
ATR_PERIOD = 14
ATR_SL_MULT = 1.5
ATR_TRAIL_MULT = 1.0
R_MULT_TP = 2.0

# --- سُلّم جني أرباح أقرب ---
P1_R = 0.6          # الهدف الأول كنسبة من R
P2_R = 1.0          # الهدف الثاني كنسبة من R
P1_FRAC = 0.25      # نسبة البيع عند TP1
P2_FRAC = 0.25      # نسبة البيع عند TP2
TRAIL_TIGHT_MULT = 0.8   # تشديد التريلينغ بعد TP2
BE_ADD_FEE = 0.0008      # نقل الوقف للتعادل + رسوم تقريبية (≈8bps)

def now_riyadh():
    return datetime.now(RIYADH_TZ)

def _today_str():
    return now_riyadh().strftime("%Y-%m-%d")

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

# ============== تخزين الصفقات ==============
def _pos_path(symbol):
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return f"{POSITIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol):
    return _read_json(_pos_path(symbol), None)

def save_position(symbol, position):
    _atomic_write(_pos_path(symbol), position)

def clear_position(symbol):
    try:
        p = _pos_path(symbol)
        if os.path.exists(p): os.remove(p)
    except: pass

def count_open_positions():
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions():
    return _read_json(CLOSED_POSITIONS_FILE, [])

def save_closed_positions(lst):
    _atomic_write(CLOSED_POSITIONS_FILE, lst)

# ============== حالة المخاطر اليومية (مختصرة) ==============
def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0, "trades_today": 0, "blocked_until": None}

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

def register_trade_result(pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(pnl_usdt)
    s["consecutive_losses"] = 0 if pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1
    save_risk_state(s)

# ============== مؤشرات ==============
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
    return df

def atr_series(df, period=14):
    c = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-c).abs(), (df["low"]-c).abs()], axis=1).max(axis=1)
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

def _df(data):  # OHLCV -> DataFrame
    return pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])

def get_sr_on_closed(df, window=50):
    if len(df) < window + 3: return (None, None)
    df_prev = df.iloc[:-2]
    w = min(window, len(df_prev))
    resistance = df_prev["high"].rolling(w).max().iloc[-1]
    support    = df_prev["low"].rolling(w).min().iloc[-1]
    return support, resistance

_LAST_ENTRY_BAR_TS = {}  # لمنع التكرار على نفس الشمعة

# ============== فحص الإشارة ==============
def check_signal(symbol):
    data5 = fetch_ohlcv(symbol, "5m", 200)
    if not data5: return None
    df5 = add_indicators(_df(data5))
    if len(df5) < 60: return None

    prev   = df5.iloc[-3]
    closed = df5.iloc[-2]   # الشمعة المكتملة
    last_ts_closed = int(closed["timestamp"])
    if _LAST_ENTRY_BAR_TS.get(symbol) == last_ts_closed:
        return None

    if not pd.isna(closed["vol_ma20"]) and closed["volume"] < closed["vol_ma20"]:
        return None
    if closed["close"] <= closed["open"]:
        return None
    if closed["close"] < closed["ema50"]:
        return None
    if not (RSI_MIN < closed["rsi"] < RSI_MAX):
        return None
    crossed = (prev["ema9"] < prev["ema21"]) and (closed["ema9"] > closed["ema21"])
    macd_ok = closed["macd"] > closed["macd_signal"]
    if not (crossed and macd_ok):
        return None

    support, resistance = get_sr_on_closed(df5, SR_WINDOW)
    price = float(closed["close"])
    if support and resistance:
        if price >= resistance * (1 - RESISTANCE_BUFFER): return None
        if price <= support    * (1 + SUPPORT_BUFFER):    return None

    if REQUIRE_MTF:
        data15 = fetch_ohlcv(symbol, "15m", 150)
        if not data15: return None
        df15 = _df(data15)
        df15["ema50"] = ema(df15["close"], EMA_TREND)
        if len(df15) < 60: return None
        closed15 = df15.iloc[-2]
        if closed15["close"] < closed15["ema50"]:
            return None

    _LAST_ENTRY_BAR_TS[symbol] = last_ts_closed
    return "buy"

# ============== تنفيذ الشراء ==============
def execute_buy(symbol):
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."

    # مبدئيًا نقرأ السعر ثم نستبدله بسعر التنفيذ الفعلي بعد الأمر
    price = float(fetch_price(symbol))
    usdt  = float(fetch_balance("USDT"))
    if price <= 0: return None, "⚠️ سعر غير صالح."
    if usdt  < TRADE_AMOUNT_USDT: return None, "🚫 رصيد USDT غير كافٍ."

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, "buy", amount)
    if not order: return None, "⚠️ فشل تنفيذ الصفقة."

    # 🔹 استخدم سعر التنفيذ الحقيقي إن توفر
    try:
        fill_px = float(order.get("average") or order.get("price") or price)
        price = fill_px
    except Exception:
        pass

    data5 = fetch_ohlcv(symbol, "5m", 100)
    df5 = add_indicators(_df(data5))
    atr = float(df5["atr"].iloc[-2])

    sl = price - ATR_SL_MULT * atr
    r  = price - sl
    tp = price + R_MULT_TP * r

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(price),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "trailing_stop": float(price - ATR_TRAIL_MULT * atr),
        "atr": float(atr),
        # سُلّم الجني
        "p1_done": False,
        "p2_done": False,
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
    }
    save_position(symbol, pos)
    register_trade_opened()
    return order, f"✅ شراء {symbol} | SL(ATR): {sl:.6f} | TP: {tp:.6f}"

# ============== إدارة الصفقة ==============
def manage_position(symbol):
    pos = load_position(symbol)
    if not pos: return False

    current = float(fetch_price(symbol))
    entry   = float(pos["entry_price"])
    amount  = float(pos["amount"])
    if amount <= 0: return False

    data5 = fetch_ohlcv(symbol, "5m", 50)
    df5 = _df(data5)
    df5["atr"] = atr_series(df5, ATR_PERIOD)
    atr = float(df5["atr"].iloc[-2])

    # تحديث trailing (رفع فقط)
    trail_level = current - ATR_TRAIL_MULT * atr
    if trail_level > pos["trailing_stop"]:
        pos["trailing_stop"] = float(trail_level)
        save_position(symbol, pos)

    # ---- R والأهداف المرحلية من TP النهائي (ثابت حتى لو تغيّر SL لاحقًا) ----
    r = (pos["take_profit"] - entry) / R_MULT_TP
    tp1 = entry + P1_R * r
    tp2 = entry + P2_R * r
    tp3 = pos["take_profit"]  # 2R

    # === TP1: بيع 25% عند 0.6R + رفع SL إلى Entry - 0.25R ===
    if (not pos.get("p1_done")) and current >= tp1 and amount > 0:
        sell_amount = amount * P1_FRAC
        order = place_market_order(symbol, "sell", sell_amount)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sell_amount
            fees = (entry + exit_px) * sell_amount * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            pos["amount"] = float(max(0.0, amount - sell_amount))
            pos["p1_done"] = True
            # حماية: SL = Entry - 0.25R (لا نُنزله لو كان أعلى)
            pos["stop_loss"] = max(pos["stop_loss"], entry - 0.25 * r)
            save_position(symbol, pos)
            register_trade_result(pnl_net)

    # === TP2: بيع 25% عند 1.0R + SL = BE+fees + تشديد التريلينغ ===
    elif pos.get("p1_done") and (not pos.get("p2_done")) and current >= tp2 and pos["amount"] > 0:
        # لجعلها 25% من الأصل (وليس من الباقي): اقسم على (1 - P1_FRAC)
        sell_amount = pos["amount"] * (P2_FRAC / max(1e-9, (1.0 - P1_FRAC)))
        order = place_market_order(symbol, "sell", sell_amount)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sell_amount
            fees = (entry + exit_px) * sell_amount * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            pos["amount"] = float(max(0.0, pos["amount"] - sell_amount))
            pos["p2_done"] = True
            # SL = التعادل + رسوم تقريبية
            pos["stop_loss"] = max(pos["stop_loss"], entry * (1 + BE_ADD_FEE))
            # تشديد التريلينغ بعد TP2
            pos["trailing_stop"] = max(pos["trailing_stop"], current - TRAIL_TIGHT_MULT * atr)
            save_position(symbol, pos)
            register_trade_result(pnl_net)

    # إغلاقات نهائية
    reason = None
    if current >= tp3:                    reason = "TP"
    elif current <= pos["stop_loss"]:     reason = "SL"
    elif current <= pos["trailing_stop"]: reason = "TRAIL"

    if reason:
        order = place_market_order(symbol, "sell", pos["amount"])
        if order:
            amount_left = float(pos["amount"])
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * amount_left
            fees = (entry + exit_px) * amount_left * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            close_trade(symbol, exit_price=exit_px, pnl_net=pnl_net, reason=reason)
            return True
    return False

# ============== إغلاق وتسجيل ==============
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

# ============== ✨ تقرير يومي: يبني النص فقط (الإرسال يتم في run.py) ==============
def _fmt_table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "<pre>" + fmt_row(headers) + "\n" + "\n".join(fmt_row(r) for r in rows) + "</pre>"

def build_daily_report_text():
    """يرجع نص التقرير اليومي (HTML) لليوم الحالي بتوقيت الرياض، أو None إن لا توجد صفقات."""
    closed = load_closed_positions()
    today = _today_str()
    todays = [t for t in closed if str(t.get("closed_at", "")).startswith(today)]
    if not todays:
        return f"📊 <b>تقرير اليوم {today}</b>\nلا توجد صفقات اليوم."

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

    summary = (
        f"📊 <b>تقرير اليوم {today}</b>\n"
        f"عدد الصفقات: <b>{len(todays)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b> • الرابحة: <b>{len(wins)}</b> • الخاسرة: <b>{len(losses)}</b>\n"
        f"أفضل صفقة: <b>{best.get('symbol','-')}</b> ({float(best.get('profit',0)):,.2f}$) • "
        f"أسوأ صفقة: <b>{worst.get('symbol','-')}</b> ({float(worst.get('profit',0)):,.2f}$)\n"
    )
    return summary + table
