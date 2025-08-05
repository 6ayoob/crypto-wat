import pandas as pd
import json
import os
from okx_api import fetch_ohlcv, fetch_price, place_market_order, get_balance
from config import TRADE_AMOUNT_USDT, STOP_LOSS_PCT, TAKE_PROFIT_PCT

MAX_OPEN_POSITIONS = 4  # الحد الأقصى للصفقات المفتوحة

def get_position_filename(symbol):
    """
    يعيد اسم ملف تخزين مركز التداول لكل رمز (زوج عملات) مع استبدال '/' بـ '_'
    """
    symbol = symbol.replace("/", "_")
    return f"positions/{symbol}.json"

def load_position(symbol):
    """
    تحميل بيانات مركز التداول من ملف JSON إذا كان موجودًا
    """
    file = get_position_filename(symbol)
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_position(symbol, position):
    """
    حفظ بيانات مركز التداول في ملف JSON
    """
    os.makedirs("positions", exist_ok=True)
    file = get_position_filename(symbol)
    with open(file, 'w') as f:
        json.dump(position, f)

def clear_position(symbol):
    """
    حذف ملف مركز التداول عند إغلاق الصفقة
    """
    file = get_position_filename(symbol)
    if os.path.exists(file):
        os.remove(file)

def count_open_positions():
    """
    حساب عدد الصفقات المفتوحة حالياً (عدد الملفات في مجلد positions)
    """
    os.makedirs("positions", exist_ok=True)
    return len([f for f in os.listdir("positions") if f.endswith(".json")])

def check_ma_crossover(df):
    """
    يتحقق من إشارة شراء بناءً على تقاطع المتوسط المتحرك لـ 20 يوم مع 50 يوم صعودًا
    """
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    # تحقق هل تقاطع الما20 فوق الما50 بين آخر نقطتين
    return df['ma20'].iloc[-2] < df['ma50'].iloc[-2] and df['ma20'].iloc[-1] > df['ma50'].iloc[-1]

def check_signal(symbol):
    """
    يفحص إشارة الشراء من خلال المتوسطات المتحركة على إطارين زمنيّين: 5 دقائق و15 دقيقة
    الشرط: وجود إشارة شراء في كلا الإطارين معًا
    """
    data_5m = fetch_ohlcv(symbol, '5m', 100)
    data_15m = fetch_ohlcv(symbol, '15m', 100)

    if not data_5m or not data_15m:
        return None

    df_5m = pd.DataFrame(data_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_15m = pd.DataFrame(data_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    signal_5m = check_ma_crossover(df_5m)
    signal_15m = check_ma_crossover(df_15m)

    if signal_5m and signal_15m:
        return "buy"

    return None

def execute_buy(symbol):
    """
    تنفيذ أمر شراء إذا لم يتم تجاوز الحد الأقصى للصفقات المفتوحة ولديك رصيد كاف
    - يحسب كمية الشراء بناءً على مبلغ ثابت بالدولار (TRADE_AMOUNT_USDT)
    - يحدد وقف الخسارة والهدف بالنسبة المئوية المحددة
    - يخزن مركز التداول
    """
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, f"🚫 الحد الأقصى للصفقات ({MAX_OPEN_POSITIONS}) مفتوح بالفعل."

    price = fetch_price(symbol)
    usdt_balance = get_balance('USDT')

    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, f"🚫 لا يوجد رصيد كافي لشراء {symbol}"

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, 'buy', amount)

    stop_loss = price * (1 - STOP_LOSS_PCT)
    take_profit = price * (1 + TAKE_PROFIT_PCT)

    position = {
        "symbol": symbol,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

    save_position(symbol, position)
    return order, f"✅ شراء {symbol} @ {price:.2f}\n🎯 هدف: {take_profit:.2f} | ❌ وقف: {stop_loss:.2f}"

def manage_position(symbol, send_message):
    """
    مراقبة المركز المفتوح:
    - إذا وصل السعر لوقف الخسارة، يتم البيع وإغلاق المركز مع إرسال رسالة
    - إذا وصل السعر لهدف الربح، يتم البيع وإغلاق المركز مع إرسال رسالة
    """
    position = load_position(symbol)
    if not position:
        return

    current_price = fetch_price(symbol)

    if current_price <= position['stop_loss']:
        place_market_order(symbol, 'sell', position['amount'])
        clear_position(symbol)
        send_message(f"❌ تم وقف الخسارة لـ {symbol} عند {current_price:.2f}")

    elif current_price >= position['take_profit']:
        place_market_order(symbol, 'sell', position['amount'])
        clear_position(symbol)
        send_message(f"🎯 تم تحقيق هدف الربح لـ {symbol} عند {current_price:.2f}")
