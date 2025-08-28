import os
import json
import time
import math
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS

# ===============================
# 📂 ملفات الصفقات والإشعارات
# ===============================
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"
DAILY_REPORT_FILE = "daily_report.json"  # غير مستخدم حالياً لكن تُرك للتمدّد

# توقيت الرياض UTC+3
RIYADH_TZ = timezone(timedelta(hours=3))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# ===============================
# ⚙️ إعدادات الاستراتيجية والحماية
# ===============================
SR_WINDOW = 50
RESISTANCE_BUFFER = 0.005
SUPPORT_BUFFER = 0.002
TRAILING_DISTANCE = 0.01
PARTIAL_FRACTION = 0.5  # نسبة جني ربح جزئي

DAILY_MAX_LOSS_USDT = 50
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_MINUTES_AFTER_HALT = 120
MAX_TRADES_PER_DAY = 10

# ===============================
# 🧰 أدوات عامة
# ===============================
def now_riyadh():
    return datetime.now(RIYADH_TZ)

def _today_str():
    return now_riyadh().strftime("%Y-%m-%d")

def ensure_dirs():
    os.makedirs(POSITIONS_DIR, exist_ok=True)

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def safe_read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ JSON read error: {path} -> {e}")
    return default

def retry(times=3, delay=0.6):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for _ in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay)
            raise last
        return wrapper
    return deco

# ===============================
# ✉️ مرسل تيليجرام (فوري + مجمّع)
# ===============================
class Notifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self._buf = []
        self._lock = threading.Lock()

    @retry()
    def _post(self, text, parse_mode="HTML"):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
        requests.post(url, data=payload, timeout=10)

    def send(self, msg):
        try:
            self._post(msg)
        except Exception as e:
            print("⚠️ فشل إرسال Telegram:", e)

    def push(self, msg):
        with self._lock:
            self._buf.append(msg)

    def flush(self, title="📬 ملخص الإشعارات"):
        with self._lock:
            if not self._buf:
                return
            joined = "\n— " + "\n— ".join(self._buf)
            self._buf.clear()
        self.send(f"<b>{title}</b>{joined}")

NOTIFIER = Notifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

def send_telegram(msg):
    NOTIFIER.send(msg)

# ===============================
# 💾 تخزين الصفقات
# ===============================
def get_position_filename(symbol):
    ensure_dirs()
    return f"{POSITIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol):
    return safe_read_json(get_position_filename(symbol), None)

def save_position(symbol, position):
    try:
        atomic_write_json(get_position_filename(symbol), position)
    except Exception as e:
        print(f"⚠️ خطأ في حفظ الصفقة: {e}")

def clear_position(symbol):
    try:
        file = get_position_filename(symbol)
        if os.path.exists(file):
            os.remove(file)
    except Exception as e:
        print(f"⚠️ خطأ في حذف الصفقة: {e}")

def count_open_positions():
    ensure_dirs()
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions():
    return safe_read_json(CLOSED_POSITIONS_FILE, [])

def save_closed_positions(closed_positions):
    try:
        atomic_write_json(CLOSED_POSITIONS_FILE, closed_positions)
    except Exception as e:
        print(f"⚠️ خطأ في حفظ الصفقات المغلقة: {e}")

# ===============================
# 📊 المؤشرات الفنية
# ===============================
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

def macd(df, fast=12, slow=26, signal=9):
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    return df

def calculate_indicators(df):
    df['ema9']  = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    df['ema50'] = ema(df['close'], 50)
    df['rsi']   = rsi(df['close'], 14)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df = macd(df)
    return df

# ===============================
# 🔎 دعم ومقاومة
# ===============================
def get_support_resistance(df, window=50):
    if len(df) < 5:
        return None, None
    df_prev = df.iloc[:-1]
    use_window = min(window, len(df_prev))
    resistance = df_prev['high'].rolling(use_window).max().iloc[-1]
    support    = df_prev['low'].rolling(use_window).min().iloc[-1]
    return support, resistance

# ===============================
# 🛡️ إدارة حالة المخاطر اليومية
# ===============================
def _default_risk_state():
    return {"date": _today_str(), "daily_pnl": 0.0, "consecutive_losses": 0, "trades_today": 0, "blocked_until": None}

def load_risk_state():
    s = safe_read_json(RISK_STATE_FILE, _default_risk_state())
    # إعادة ضبط لو تغيّر اليوم
    if s.get("date") != _today_str():
        s = _default_risk_state()
        save_risk_state(s)
    return s

def save_risk_state(s):
    try:
        atomic_write_json(RISK_STATE_FILE, s)
    except Exception:
        pass

def is_trading_blocked():
    s = load_risk_state()
    if s.get("blocked_until"):
        try:
            until = datetime.fromisoformat(s["blocked_until"])
            if until.tzinfo is None:
                until = until.replace(tzinfo=RIYADH_TZ)
            if now_riyadh() < until:
                return True, f"⏸️ التداول موقوف حتى {until.astimezone(RIYADH_TZ).isoformat()}."
        except Exception:
            pass
    if s.get("daily_pnl", 0.0) <= -abs(DAILY_MAX_LOSS_USDT):
        return True, "⛔ تم بلوغ حد الخسارة اليومية."
    if s.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
        return True, "⛔ تم بلوغ حد الخسائر المتتالية."
    if s.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
        return True, f"⛔ تم بلوغ الحد الأقصى للصفقات اليوم ({MAX_TRADES_PER_DAY})."
    return False, ""

def trigger_cooldown(reason="risk_halt"):
    s = load_risk_state()
    until = now_riyadh() + timedelta(minutes=COOLDOWN_MINUTES_AFTER_HALT)
    s["blocked_until"] = until.isoformat()
    save_risk_state(s)
    send_telegram(f"⏸️ تفعيل تهدئة حتى {until.isoformat()} ({reason})")

def register_trade_opened():
    s = load_risk_state()
    s["trades_today"] = int(s.get("trades_today", 0)) + 1  # ✅ إصلاح: كانت +2
    save_risk_state(s)

def register_trade_result(total_pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] = float(s.get("daily_pnl", 0.0)) + float(total_pnl_usdt)
    s["consecutive_losses"] = 0 if total_pnl_usdt > 0 else int(s.get("consecutive_losses", 0)) + 1
    save_risk_state(s)

# ===============================
# 🔎 فحص الإشارة
# ===============================
def check_signal(symbol):
    blocked, msg = is_trading_blocked()
    if blocked:
        print(msg)
        return None

    data = fetch_ohlcv(symbol, '5m', 150)
    if not data:
        return None

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = calculate_indicators(df)
    if len(df) < 50:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # فلتر الحجم والشمعة صاعدة
    if len(df['volume']) >= 20:
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] < avg_vol or last['close'] <= last['open']:
            return None

    # الاتجاه EMA50
    if last['close'] < last['ema50']:
        return None

    # RSI
    if not (50 < last['rsi'] < 70):
        return None

    # الدعم والمقاومة
    support, resistance = get_support_resistance(df, SR_WINDOW)
    price = float(last['close'])
    if support and resistance:
        if price >= resistance * (1 - RESISTANCE_BUFFER) or price <= support * (1 + SUPPORT_BUFFER):
            return None

    # تقاطع EMA9 و EMA21 + MACD تأكيدي
    if prev['ema9'] < prev['ema21'] and last['ema9'] > last['ema21'] and last['macd'] > last['macd_signal']:
        return "buy"

    return None

# ===============================
# 🛒 تنفيذ الشراء
# ===============================
def execute_buy(symbol):
    blocked, msg = is_trading_blocked()
    if blocked:
        return None, msg

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, "🚫 الحد الأقصى للصفقات المفتوحة."

    price = float(fetch_price(symbol))
    usdt_balance = float(fetch_balance('USDT'))
    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, "🚫 رصيد USDT غير كافٍ."

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, 'buy', amount)
    if not order:
        return None, "⚠️ فشل تنفيذ الصفقة."

    # وقف خسارة من آخر 10 شمعات
    data = fetch_ohlcv(symbol, '5m', 20)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    swing_low = float(df['low'].rolling(10).min().iloc[-2])
    stop_loss = swing_low
    take_profit = price + (price - stop_loss) * 2  # RR 1:2

    position = {
        "symbol": symbol,
        "amount": float(amount),
        "entry_price": price,
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "trailing_stop": price * (1 - TRAILING_DISTANCE),
        "partial_done": False,
        "opened_at": now_riyadh().isoformat(timespec="seconds"),
    }
    save_position(symbol, position)

    # ✅ إصلاح: زيادة صفقة واحدة فقط
    register_trade_opened()

    send_telegram(
        f"✅ <b>شراء</b> {symbol}\n"
        f"سعر الدخول: <b>{price:.8f}</b>\nTP: <b>{take_profit:.8f}</b> • SL: <b>{stop_loss:.8f}</b>"
    )
    return order, f"✅ تم شراء {symbol}"

# ===============================
# 🔧 إدارة الصفقات تلقائيًا
# ===============================
def manage_position(symbol):
    closed = False
    try:
        position = load_position(symbol)
        if not position:
            return False

        current_price = float(fetch_price(symbol))
        amount = float(position['amount'])
        entry_price = float(position['entry_price'])

        base_asset = symbol.split('/')[0]
        actual_balance = float(fetch_balance(base_asset))
        sell_amount = min(amount, actual_balance)

        # Partial TP (عند نصف المسافة نحو TP)
        half_target = entry_price + (position['take_profit'] - entry_price) / 2
        if not position.get("partial_done") and current_price >= half_target:
            partial_amount = sell_amount * PARTIAL_FRACTION
            order = place_market_order(symbol, 'sell', partial_amount)
            if order:
                position['amount'] = max(0.0, position['amount'] - partial_amount)
                position['partial_done'] = True
                save_position(symbol, position)
                realized_pnl = (current_price - entry_price) * partial_amount
                register_trade_result(realized_pnl)
                NOTIFIER.push(f"📌 Partial TP {symbol}: {realized_pnl:.2f} USDT")

        # Take Profit كامل
        if current_price >= position['take_profit'] and sell_amount > 0:
            order = place_market_order(symbol, 'sell', sell_amount)
            if order:
                realized_pnl = (current_price - entry_price) * sell_amount
                close_trade(symbol, exit_price=current_price, pnl=realized_pnl, reason="TP")
                closed = True
                send_telegram(f"🎯 <b>Take Profit</b> {symbol}: {realized_pnl:.2f} USDT")

        # Stop Loss
        elif current_price <= position['stop_loss'] and sell_amount > 0:
            order = place_market_order(symbol, 'sell', sell_amount)
            if order:
                realized_pnl = (current_price - entry_price) * sell_amount
                close_trade(symbol, exit_price=current_price, pnl=realized_pnl, reason="SL")
                closed = True
                send_telegram(f"🛑 <b>Stop Loss</b> {symbol}: {realized_pnl:.2f} USDT")
                s = load_risk_state()
                if s.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
                    trigger_cooldown("max_consecutive_losses")

        # Trailing Stop (رفع الوقف المتحرك مع الصعود)
        elif current_price > position['trailing_stop'] / (1 - TRAILING_DISTANCE):
            position['trailing_stop'] = current_price * (1 - TRAILING_DISTANCE)
            save_position(symbol, position)

        # Flush الإشعارات المجمّعة كل إدارة
        NOTIFIER.flush()

    except Exception as e:
        print(f"⚠️ خطأ في إدارة الصفقة لـ {symbol}: {e}")
    return closed

# ===============================
# 🔒 إغلاق الصفقة وتسجيلها
# ===============================
def close_trade(symbol, exit_price, pnl, reason="MANUAL"):
    position = load_position(symbol)
    if not position:
        return
    closed_positions = load_closed_positions()

    amount = float(position['amount'])
    entry_price = float(position['entry_price'])
    pnl_pct = ((exit_price / entry_price) - 1.0) if entry_price else 0.0

    closed_positions.append({
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "amount": amount,
        "profit": float(pnl),
        "pnl_pct": round(pnl_pct, 6),
        "reason": reason,
        "opened_at": position.get("opened_at"),
        "closed_at": now_riyadh().isoformat(timespec="seconds"),
    })
    save_closed_positions(closed_positions)
    register_trade_result(pnl)
    clear_position(symbol)

# ===============================
# 📈 تقرير يومي على Telegram
# ===============================
def _fmt_table(rows, headers):
    # تنسيق بسيط بمحاذاة باستخدام <pre>
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "<pre>" + fmt_row(headers) + "\n" + "\n".join(fmt_row(r) for r in rows) + "</pre>"

def send_daily_report():
    closed_positions = load_closed_positions()
    today_str = _today_str()
    daily_trades = [c for c in closed_positions if str(c.get('closed_at', '')).startswith(today_str)]

    if not daily_trades:
        send_telegram(f"📊 <b>تقرير اليوم {today_str}</b>\nلا توجد صفقات اليوم.")
        return

    total_pnl = sum(float(t['profit']) for t in daily_trades)
    win_count = sum(1 for t in daily_trades if float(t['profit']) > 0)
    loss_count = len(daily_trades) - win_count
    win_rate = round(100 * win_count / len(daily_trades), 2)

    best = max(daily_trades, key=lambda t: float(t['profit']))
    worst = min(daily_trades, key=lambda t: float(t['profit']))

    headers = ["الرمز", "الكمية", "دخول", "خروج", "P/L$", "P/L%"]
    rows = []
    for t in daily_trades:
        rows.append([
            t['symbol'],
            f"{float(t['amount']):.6f}",
            f"{float(t['entry_price']):.6f}",
            f"{float(t['exit_price']):.6f}",
            f"{float(t['profit']):.2f}",
            f"{round(float(t.get('pnl_pct',0.0))*100,2)}%"
        ])
    table = _fmt_table(rows, headers)

    summary = (
        f"📊 <b>تقرير اليوم {today_str}</b>\n"
        f"عدد الصفقات: <b>{len(daily_trades)}</b> • ربح/خسارة: <b>{total_pnl:.2f}$</b>\n"
        f"نسبة الفوز: <b>{win_rate}%</b> • الرابحة: <b>{win_count}</b> • الخاسرة: <b>{loss_count}</b>\n"
        f"أفضل صفقة: <b>{best['symbol']}</b> ({float(best['profit']):.2f}$) • "
        f"أسوأ صفقة: <b>{worst['symbol']}</b> ({float(worst['profit']):.2f}$)\n"
    )
    send_telegram(summary + table)

# ===============================
# 🔄 حلقة التشغيل التلقائي
# ===============================
def run_bot(loop_delay_sec=60):
    while True:
        blocked, msg = is_trading_blocked()
        if blocked:
            print(msg)
            time.sleep(loop_delay_sec)
            continue

        for symbol in SYMBOLS:
            try:
                signal = check_signal(symbol)
                if signal == "buy":
                    _, m = execute_buy(symbol)
                    print(m)
                manage_position(symbol)
            except Exception as e:
                print(f"⚠️ خطأ في الرمز {symbol}: {e}")

        time.sleep(loop_delay_sec)

# ===============================
# ⏰ جدولة التقرير اليومي (بتوقيت الرياض)
# ===============================
def schedule_daily_report(hour=9, minute=0):
    def report_loop():
        while True:
            now = now_riyadh()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            sleep_seconds = (target - now).total_seconds()
            time.sleep(max(1, sleep_seconds))
            try:
                send_daily_report()
            except Exception as e:
                print("Report error:", e)
            # تجنّب التكرار بنفس الدقيقة
            time.sleep(61)
    t = threading.Thread(target=report_loop, daemon=True)
    t.start()

# ===============================
# 🔹 بدء البوت
# ===============================
if __name__ == "__main__":
    print("🚀 بدء تشغيل البوت Spot OKX")
    send_telegram("🚀 بدأ تشغيل بوت Spot OKX")
    schedule_daily_report(hour=9, minute=0)  # 09:00 بتوقيت الرياض
    run_bot(loop_delay_sec=60)
