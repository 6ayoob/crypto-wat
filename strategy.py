import pandas as pd
import json
import os
from datetime import datetime
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS

POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"

# ===============================
# 📂 التعامل مع ملفات الصفقات
# ===============================

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
            with open(file, 'r') as f:
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
# 📊 المؤشرات الفنية
# ===============================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df['ema9'] = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    return df

# ===============================
# 🔎 دعم ومقاومة (Support & Resistance)
# ===============================

def get_support_resistance(df, window=50):
    """
    يحسب مستوى الدعم والمقاومة من بيانات الشموع.
    نستخدم نافذة (window) من الشموع السابقة (نستبعد الشمعة الحالية عند الحساب).
    يعيد (support, resistance) أو (None, None) إذا البيانات غير كافية.
    """
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
# 🎯 منطق الإشارة مع فلتر SR
# ===============================

SR_WINDOW = 50
RESISTANCE_BUFFER = 0.005  # 0.5% لا نشترِي إذا السعر قريب جدًا من المقاومة
SUPPORT_BUFFER = 0.002     # 0.2% نطلب أن السعر يكون فوق الدعم بقليل

def check_signal(symbol):
    try:
        data_5m = fetch_ohlcv(symbol, '5m', 100)
        if not data_5m:
            return None

        df = pd.DataFrame(data_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_indicators(df)

        if len(df) < 5:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # فلتر الحجم: الحجم الأخير يجب أن يكون أعلى من متوسط 20 شمعة (إن أمكن)
        if len(df['volume']) >= 20:
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            if pd.notna(avg_vol) and last['volume'] < avg_vol:
                return None

        support, resistance = get_support_resistance(df, window=SR_WINDOW)
        last_price = float(last['close'])

        if support is not None and resistance is not None:
            if last_price >= resistance * (1 - RESISTANCE_BUFFER):
                return None
            if last_price <= support * (1 + SUPPORT_BUFFER):
                return None

        if (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']) and (last['rsi'] > 50):
            return "buy"
    except Exception as e:
        print(f"⚠️ خطأ في فحص الإشارة لـ {symbol}: {e}")
    return None

# ===============================
# 🛒 تنفيذ الشراء (هدف ربح 4%)
# ===============================

def execute_buy(symbol):
    try:
        if count_open_positions() >= MAX_OPEN_POSITIONS:
            return None, f"🚫 وصلت للحد الأقصى للصفقات المفتوحة ({MAX_OPEN_POSITIONS})."

        price = fetch_price(symbol)
        usdt_balance = fetch_balance('USDT')

        if usdt_balance < TRADE_AMOUNT_USDT:
            return None, f"🚫 رصيد USDT غير كافٍ لشراء {symbol}."

        amount = TRADE_AMOUNT_USDT / price
        order = place_market_order(symbol, 'buy', amount)

        stop_loss = price * 0.98  # 2% وقف خسارة
        take_profit = price * 1.04  # 4% هدف ربح

        position = {
            "symbol": symbol,
            "amount": amount,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        save_position(symbol, position)
        return order, f"✅ تم شراء {symbol} بسعر {price:.8f}\n🎯 هدف الربح: {take_profit:.8f} (+4%) | 🛑 وقف الخسارة: {stop_loss:.8f} (-2%)"
    except Exception as e:
        return None, f"⚠️ خطأ أثناء تنفيذ الشراء لـ {symbol}: {e}"

# ===============================
# 📈 إدارة الصفقات
# ===============================

def manage_position(symbol):
    try:
        position = load_position(symbol)
        if not position:
            return False

        current_price = fetch_price(symbol)
        amount = position['amount']
        entry_price = position['entry_price']

        base_asset = symbol.split('/')[0]
        actual_balance = fetch_balance(base_asset)
        sell_amount = round(min(amount, actual_balance), 6)

        def close_trade(exit_price):
            profit = (exit_price - entry_price) * sell_amount
            closed_positions = load_closed_positions()
            closed_positions.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "amount": sell_amount,
                "profit": profit,
                "closed_at": datetime.utcnow().isoformat()
            })
            save_closed_positions(closed_positions)
            clear_position(symbol)
            return True

        if current_price >= position['take_profit']:
            order = place_market_order(symbol, 'sell', sell_amount)
            if order:
                return close_trade(current_price)

        if current_price <= position['stop_loss']:
            order = place_market_order(symbol, 'sell', sell_amount)
            if order:
                return close_trade(current_price)

    except Exception as e:
        print(f"⚠️ خطأ في إدارة الصفقة لـ {symbol}: {e}")

    return False
