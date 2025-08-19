import os
import json
import time
import pandas as pd
from datetime import datetime
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS

# ===============================
# 📂 ملفات الصفقات
# ===============================
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"

def ensure_dirs():
    os.makedirs(POSITIONS_DIR, exist_ok=True)

def get_position_filename(symbol):
    ensure_dirs()
    symbol = symbol.replace("/", "_")
    return f"{POSITIONS_DIR}/{symbol}.json"

def load_position(symbol):
    try:
        file = get_position_filename(symbol)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ خطأ في قراءة الصفقة: {e}")
    return None

def save_position(symbol, position):
    try:
        ensure_dirs()
        file = get_position_filename(symbol)
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(position, f, indent=2, ensure_ascii=False)
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
    try:
        if os.path.exists(CLOSED_POSITIONS_FILE):
            with open(CLOSED_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ خطأ في قراءة الصفقات المغلقة: {e}")
    return []

def save_closed_positions(closed_positions):
    try:
        with open(CLOSED_POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(closed_positions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ خطأ في حفظ الصفقات المغلقة: {e}")

# ===============================
# 📊 المؤشرات الفنية (EMA / RSI / MACD)
# ===============================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_cols(df, fast=12, slow=26, signal=9):
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    return df

def calculate_indicators(df):
    df['ema9']  = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    df['rsi']   = rsi(df['close'], 14)
    df['ema50'] = ema(df['close'], 50)  # تأكيد الاتجاه
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df = macd_cols(df)                  # إضافة MACD
    return df

# ===============================
# 🔎 دعم ومقاومة
# ===============================
def get_support_resistance(df, window=50):
    try:
        n = len(df)
        if n < 5:
            return None, None
        df_prev = df.iloc[:-1].copy()
        if len(df_prev) < 1:
            return None, None
        use_window = min(window, len(df_prev))
        resistance = df_prev['high'].rolling(use_window).max().iloc[-1]
        support = df_prev['low'].rolling(use_window).min().iloc[-1]
        if pd.isna(support) or pd.isna(resistance):
            return None, None
        return support, resistance
    except Exception as e:
        print(f"⚠️ خطأ في حساب الدعم/المقاومة: {e}")
        return None, None

# ===============================
# ⚙️ إعدادات فلتر SR و Trailing
# ===============================
SR_WINDOW = 50
RESISTANCE_BUFFER = 0.005  # 0.5%
SUPPORT_BUFFER    = 0.002  # 0.2%
TRAILING_DISTANCE = 0.01   # 1% مسافة وقف متحرك بعد TP1
PARTIAL_FRACTION  = 0.5    # إغلاق 50% عند TP1

# ===============================
# 🎯 منطق الإشارة مع MACD + فلاتر
# ===============================
def check_signal(symbol):
    try:
        data_5m = fetch_ohlcv(symbol, '5m', 200)
        if not data_5m:
            return None
        df = pd.DataFrame(data_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_indicators(df)
        if len(df) < 60:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # فلتر الحجم + شمعة صاعدة
        if len(df['volume']) >= 20:
            avg_vol = df['vol_ma20'].iloc[-1]
            if pd.notna(avg_vol) and (last['volume'] < avg_vol or last['close'] <= last['open']):
                return None

        # اتجاه: فوق EMA50
        if last['close'] < last['ema50']:
            return None

        # RSI بين 50 و 70
        if not (50 < last['rsi'] < 70):
            return None

        # MACD صاعد
        if not (last['macd'] > last['macd_signal']):
            return None

        # دعم/مقاومة
        support, resistance = get_support_resistance(df, window=SR_WINDOW)
        last_price = float(last['close'])
        if support is not None and resistance is not None:
            if last_price >= resistance * (1 - RESISTANCE_BUFFER):
                return None
            if last_price <= support * (1 + SUPPORT_BUFFER):
                return None

        # إشارة الدخول: تقاطع EMA9/EMA21 صعودي
        if (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']):
            return "buy"

    except Exception as e:
        print(f"⚠️ خطأ في فحص الإشارة لـ {symbol}: {e}")
    return None

# ===============================
# 🛒 تنفيذ الشراء (Partial TP + Trailing Stop)
# ===============================
def execute_buy(symbol):
    try:
        if count_open_positions() >= MAX_OPEN_POSITIONS:
            return None, f"🚫 وصلت للحد الأقصى للصفقات المفتوحة ({MAX_OPEN_POSITIONS})."

        price = fetch_price(symbol)
        usdt_balance = fetch_balance('USDT')
        if usdt_balance < TRADE_AMOUNT_USDT:
            return None, f"🚫 رصيد USDT غير كافٍ لشراء {symbol}."

        amount_total = TRADE_AMOUNT_USDT / price
        order = place_market_order(symbol, 'buy', amount_total)
        if not order:
            return None, f"⚠️ فشل تنفيذ أمر الشراء لـ {symbol}."

        # وقف: أدنى قاع آخر 10 شموع (قبل الشمعة الحالية)
        data_5m = fetch_ohlcv(symbol, '5m', 20)
        df = pd.DataFrame(data_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        swing_low = df['low'].rolling(10).min().iloc[-2]

        stop_loss = float(swing_low)
        risk = price - stop_loss
        if risk <= 0:
            return None, f"⚠️ لم يتم فتح الصفقة لأن المخاطرة سالبة {symbol}."

        tp1 = price + risk * 1.0    # الهدف الأول 1:1
        tp2 = price + risk * 2.0    # الهدف الثاني 1:2

        position = {
            "symbol": symbol,
            "amount_total": float(amount_total),
            "amount_open": float(amount_total),   # المتبقي المفتوح
            "entry_price": float(price),
            "stop_loss": float(stop_loss),
            "take_profit_1": float(tp1),
            "take_profit_2": float(tp2),
            "partial_closed": False,
            "trailing_stop": None,               # يُفعّل بعد TP1
            "highest_price": float(price),       # لأجل التريلينغ
            "opened_at": datetime.utcnow().isoformat()
        }
        save_position(symbol, position)
        msg = (
            f"✅ تم شراء {symbol} بسعر {price:.8f}\n"
            f"🎯 TP1: {tp1:.8f} | 🎯 TP2: {tp2:.8f} | 🛑 SL: {stop_loss:.8f}"
        )
        return order, msg
    except Exception as e:
        return None, f"⚠️ خطأ أثناء تنفيذ الشراء لـ {symbol}: {e}"

# ===============================
# 📈 إدارة الصفقات (Partial TP + Trailing Stop)
# ===============================
def manage_position(symbol):
    try:
        position = load_position(symbol)
        if not position:
            return False

        current_price = fetch_price(symbol)
        entry_price   = position['entry_price']
        amount_open   = float(position.get('amount_open', 0.0))
        if amount_open <= 0:
            clear_position(symbol)
            return False

        # تحديث أعلى سعر
        position['highest_price'] = max(position.get('highest_price', entry_price), current_price)

        base_asset = symbol.split('/')[0]
        actual_balance = fetch_balance(base_asset)
        # تأكد ألا نبيع أكثر من الرصيد المتاح
        sellable_amount = min(amount_open, actual_balance)
        if sellable_amount <= 0:
            return False

        def close_part(exit_price, qty, reason):
            # تنفيذ بيع
            order = place_market_order(symbol, 'sell', qty)
            if not order:
                return False
            # تحديث الحالة
            position['amount_open'] = float(max(0.0, position['amount_open'] - qty))
            save_position(symbol, position)
            # حفظ في السجل
            profit = (exit_price - entry_price) * qty
            closed = load_closed_positions()
            closed.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "amount": float(qty),
                "profit": float(profit),
                "reason": reason,
                "closed_at": datetime.utcnow().isoformat()
            })
            save_closed_positions(closed)
            return True

        def close_all(exit_price, reason):
            qty = min(position['amount_open'], fetch_balance(base_asset))
            if qty <= 0:
                return False
            ok = close_part(exit_price, qty, reason)
            if ok:
                clear_position(symbol)
            return ok

        # 🎯 الوصول إلى TP1 (إغلاق جزئي + تفعيل التريلينغ)
        if not position.get("partial_closed", False) and current_price >= position['take_profit_1']:
            qty_half = sellable_amount * PARTIAL_FRACTION
            if qty_half > 0:
                if close_part(current_price, qty_half, "TP1_PARTIAL"):
                    position['partial_closed'] = True
                    # قفل الصفقة على الأقل على نقطة الدخول
                    position['stop_loss'] = max(position['stop_loss'], position['entry_price'])
                    # تفعيل Trailing
                    position['trailing_stop'] = position['highest_price'] * (1 - TRAILING_DISTANCE)
                    save_position(symbol, position)
                    print(f"ℹ️ {symbol}: أخذ ربح جزئي وتفعيل Trailing Stop.")

        # 🎯 الهدف الثاني TP2 (إغلاق كامل)
        if current_price >= position['take_profit_2']:
            if close_all(current_price, "TP2_FULL"):
                print(f"✅ {symbol}: تحقق الهدف الثاني وإغلاق كامل.")
                return True

        # 🔁 تحديث Trailing Stop (لو مفعّل)
        if position.get("trailing_stop"):
            new_trailing = position['highest_price'] * (1 - TRAILING_DISTANCE)
            if new_trailing > position['trailing_stop']:
                position['trailing_stop'] = new_trailing
                save_position(symbol, position)

            if current_price <= position['trailing_stop']:
                if close_all(current_price, "TRAILING_STOP"):
                    print(f"🛑 {symbol}: تم الخروج بواسطة Trailing Stop.")
                    return True

        # 🛑 وقف الخسارة
        if current_price <= position['stop_loss']:
            if close_all(current_price, "STOP_LOSS"):
                print(f"🛑 {symbol}: تم الخروج بواسطة وقف الخسارة.")
                return True

    except Exception as e:
        print(f"⚠️ خطأ في إدارة الصفقة لـ {symbol}: {e}")

    return False

# ===============================
# 🚀 حلقة التشغيل الآلي
# ===============================
SCAN_INTERVAL_SEC = 15     # كل كم ثانية يفحص الإشارات
MANAGE_INTERVAL_SEC = 5    # كل كم ثانية يدير الصفقات المفتوحة
PER_SYMBOL_PAUSE = 0.4     # لتخفيف الضغط على API

def run_live():
    print("✅ بدء التشغيل الآلي (OKX Spot) مع MACD + Partial TP + Trailing Stop")
    print(f"الرموز: {', '.join(SYMBOLS)}")
    last_scan = 0
    while True:
        now = time.time()

        # إدارة الصفقات أولاً (أسرع)
        if True:
            for sym in SYMBOLS:
                try:
                    manage_position(sym)
                    time.sleep(PER_SYMBOL_PAUSE)
                except Exception as e:
                    print(f"⚠️ manage_position({sym}) -> {e}")

        # فحص الإشارات وفتح صفقات جديدة
        if now - last_scan >= SCAN_INTERVAL_SEC:
            for sym in SYMBOLS:
                try:
                    # لا تفتح جديدة إذا فيه صفقة مفتوحة على نفس الرمز
                    if load_position(sym):
                        continue
                    signal = check_signal(sym)
                    if signal == "buy":
                        if count_open_positions() < MAX_OPEN_POSITIONS:
                            order, msg = execute_buy(sym)
                            print(msg)
                        else:
                            print(f"🚫 حد الصفقات المفتوحة ممتلئ ({MAX_OPEN_POSITIONS}).")
                    time.sleep(PER_SYMBOL_PAUSE)
                except Exception as e:
                    print(f"⚠️ check/execute({sym}) -> {e}")
            last_scan = now

        time.sleep(MANAGE_INTERVAL_SEC)

if __name__ == "__main__":
    ensure_dirs()
    try:
        run_live()
    except KeyboardInterrupt:
        print("\n👋 تم الإيقاف اليدوي.")
