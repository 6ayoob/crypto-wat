# strategy.py — 80/100 + Daily Report Builder (Fixed %: SL 2% • TP1 3% • TP2 6% + EMA50 slope & S/R)
import os, json
from datetime import datetime, timedelta, timezone
import pandas as pd

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS, FEE_BPS_ROUNDTRIP

RIYADH_TZ = timezone(timedelta(hours=3))
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

# ===== إعدادات المؤشرات =====
EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 50
RSI_MIN, RSI_MAX = 50, 70
VOL_MA, SR_WINDOW = 20, 50
RESISTANCE_BUFFER, SUPPORT_BUFFER = 0.005, 0.002

# تأكيد تعدد الأطر (15m فوق EMA50)
REQUIRE_MTF = True

# ===== نسب ثابتة للأهداف والوقف =====
STOP_LOSS_PCT = 0.02   # 2% أسفل الدخول
TP1_PCT       = 0.03   # 3% فوق الدخول (بيع 50% + نقل SL للتعادل)
TP2_PCT       = 0.06   # 6% فوق الدخول (إغلاق كامل)
TP1_FRACTION  = 0.5    # نسبة البيع عند TP1

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

def add_indicators(df):
    df["ema9"]  = ema(df["close"], EMA_FAST)
    df["ema21"] = ema(df["close"], EMA_SLOW)
    df["ema50"] = ema(df["close"], EMA_TREND)
    df["rsi"]   = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(VOL_MA).mean()
    df = macd_cols(df)
    return df

def _df(data):  # OHLCV -> DataFrame
    return pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])

def get_sr_on_closed(df, window=50):
    if len(df) < window + 3: return (None, None)
    df_prev = df.iloc[:-2]  # حتى الشمعة المغلقة
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

    price = float(closed["close"])
    ema50_now  = float(closed["ema50"])
    ema50_prev = float(df5["ema50"].iloc[-7]) if len(df5) >= 7 else ema50_now
    ema50_slope_up = (ema50_now - ema50_prev) > 0

    # اتجاه أقوى حول EMA50
    if not ema50_slope_up:
        return None
    if price < ema50_now * 1.001:  # هامش فوق EMA50 (~0.1%)
        return None
    if price > ema50_now * 1.03:   # منع التمدد >3%
        return None

    # حجم/شمعة/RSI
    if not pd.isna(closed["vol_ma20"]) and closed["volume"] < closed["vol_ma20"]:
        return None
    if closed["close"] <= closed["open"]:
        return None
    if not (RSI_MIN < closed["rsi"] < RSI_MAX):
        return None

    # دعم/مقاومة
    support, resistance = get_sr_on_closed(df5, SR_WINDOW)
    if support and resistance:
        if price >= resistance * (1 - RESISTANCE_BUFFER): return None
        if price <= support    * (1 + SUPPORT_BUFFER):    return None

    # تقاطع EMA9/21 + MACD
    crossed = (prev["ema9"] < prev["ema21"]) and (closed["ema9"] > closed["ema21"])
    macd_ok = closed["macd"] > closed["macd_signal"]
    if not (crossed and macd_ok):
        return None

    # تعدد الأطر (15m فوق EMA50)
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

    # السعر الحالي (سنستبدله بسعر التنفيذ الفعلي إن توفر)
    price = float(fetch_price(symbol))
    usdt  = float(fetch_balance("USDT"))
    if price <= 0: return None, "⚠️ سعر غير صالح."
    if usdt  < TRADE_AMOUNT_USDT: return None, "🚫 رصيد USDT غير كافٍ."

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, "buy", amount)
    if not order: return None, "⚠️ فشل تنفيذ الصفقة."

    # سعر التنفيذ الفعلي إن توفر
    try:
        fill_px = float(order.get("average") or order.get("price") or price)
        price = fill_px if fill_px > 0 else price
    except Exception:
        pass

    sl  = price * (1 - STOP_LOSS_PCT)
    tp1 = price * (1 + TP1_PCT)
    tp2 = price * (1 + TP2_PCT)

    pos = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": float(price),
        "stop_loss": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "partial_done": False,  # TP1 لم يُنفذ بعد
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
    }
    save_position(symbol, pos)
    register_trade_opened()
    return order, f"✅ شراء {symbol} | SL: {sl:.6f} | TP1: {tp1:.6f} | TP2: {tp2:.6f}"

# ============== إدارة الصفقة ==============
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
        print(f"⚠️ لا يوجد رصيد {base_asset} للبيع — إغلاق محلي.")
        clear_position(symbol)
        return False

    sellable = min(amount, wallet_balance)

    # --- TP1: بيع 50% + SL = التعادل ---
    if (not pos.get("partial_done")) and current >= tp1 and sellable > 0:
        part_qty = sellable * TP1_FRACTION
        order = place_market_order(symbol, "sell", part_qty)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * part_qty
            fees = (entry + exit_px) * part_qty * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees

            pos["amount"] = float(max(0.0, amount - part_qty))
            pos["partial_done"] = True
            pos["stop_loss"] = float(entry)  # نقل الوقف للتعادل
            save_position(symbol, pos)
            register_trade_result(pnl_net)

    # تحديث القيم بعد TP1 المحتمل
    pos_ref = load_position(symbol)
    if not pos_ref: 
        return True
    amount = float(pos_ref.get("amount", 0.0))
    wallet_balance = float(fetch_balance(base_asset) or 0)
    sellable = min(amount, wallet_balance)

    # --- TP2: إغلاق كامل ---
    if sellable > 0 and current >= tp2:
        order = place_market_order(symbol, "sell", sellable)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sellable
            fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            close_trade(symbol, exit_price=exit_px, pnl_net=pnl_net, reason="TP2")
            return True

    # --- SL: إغلاق كامل ---
    pos_ref = load_position(symbol)
    if not pos_ref:
        return True
    amount = float(pos_ref.get("amount", 0.0))
    wallet_balance = float(fetch_balance(base_asset) or 0)
    sellable = min(amount, wallet_balance)

    if sellable > 0 and current <= sl:
        order = place_market_order(symbol, "sell", sellable)
        if order:
            exit_px = float(order.get("average") or order.get("price") or current)
            pnl_gross = (exit_px - entry) * sellable
            fees = (entry + exit_px) * sellable * (FEE_BPS_ROUNDTRIP / 10000.0)
            pnl_net = pnl_gross - fees
            close_trade(symbol, exit_price=exit_px, pnl_net=pnl_net, reason="SL")
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
