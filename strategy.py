import pandas as pd
import numpy as np
import json
import os
import talib
from okx_api import fetch_ohlcv, fetch_price, place_market_order, get_balance
from config import TRADE_AMOUNT_USDT  # لم نعد نستخدم STOP_LOSS_PCT وTAKE_PROFIT_PCT لأنها أصبحت ديناميكية

MAX_OPEN_POSITIONS = 4  # الحد الأقصى للصفقات المفتوحة

# ========================== إدارة مراكز التداول ==========================

def get_position_filename(symbol):
    """اسم ملف تخزين مركز التداول لكل زوج"""
    symbol = symbol.replace("/", "_")
    return f"positions/{symbol}.json"

def load_position(symbol):
    """تحميل بيانات المركز من ملف JSON"""
    file = get_position_filename(symbol)
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_position(symbol, position):
    """حفظ بيانات المركز"""
    os.makedirs("positions", exist_ok=True)
    file = get_position_filename(symbol)
    with open(file, 'w') as f:
        json.dump(position, f)

def clear_position(symbol):
    """حذف ملف المركز عند الإغلاق"""
    file = get_position_filename(symbol)
    if os.path.exists(file):
        os.remove(file)

def count_open_positions():
    """عدد الصفقات المفتوحة حالياً"""
    os.makedirs("positions", exist_ok=True)
    return len([f for f in os.listdir("positions") if f.endswith(".json")])

# ========================== المؤشرات الفنية ==========================

def calculate_indicators(df):
    """حساب EMA9, EMA21, RSI, ATR"""
    df['ema9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema21'] = talib.EMA(df['close'], timeperiod=21)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return df

# ========================== فحص الإشارة اللحظية ==========================

def check_signal(symbol):
    """
    استراتيجية لحظية:
    - EMA9 يتقاطع صعودًا فوق EMA21
    - RSI > 50
    - حجم تداول أعلى من متوسط آخر 20 شمعة
    - تأكيد الاتجاه على إطار 5m
    """
    data_1m = fetch_ohlcv(symbol, '1m', 200)
    data_5m = fetch_ohlcv(symbol, '5m', 200)
    if not data_1m or not data_5m:
        return None

    df1 = pd.DataFrame(data_1m, columns=['timestamp','open','high','low','close','volume'])
    df5 = pd.DataFrame(data_5m, columns=['timestamp','open','high','low','close','volume'])
    df1 = calculate_indicators(df1)
    df5 = calculate_indicators(df5)

    last = df1.iloc[-1]
    prev = df1.iloc[-2]

    cond_buy = (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']) \
               and (last['rsi'] > 50) \
               and (last['volume'] > df1['volume'].rolling(20).mean().iloc[-1]) \
               and (df5['ema9'].iloc[-1] > df5['ema21'].iloc[-1])  # تأكيد من 5m

    return "buy" if cond_buy else None

# ========================== تنفيذ الشراء ==========================

def execute_buy(symbol):
    """
    تنفيذ شراء مع وقف خسارة وجني أرباح ديناميكيين باستخدام ATR
    وإدارة مخاطرة بنسبة 1% من الرصيد لكل صفقة
    """
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, f"🚫 الحد الأقصى للصفقات ({MAX_OPEN_POSITIONS}) مفتوح بالفعل."

    price = fetch_price(symbol)
    usdt_balance = get_balance('USDT')
    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, f"🚫 لا يوجد رصيد كافي لشراء {symbol}"

    # جلب ATR لتحديد SL/TP
    data = fetch_ohlcv(symbol, '1m', 200)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df = calculate_indicators(df)
    atr = df['atr'].iloc[-1]

    stop_loss = price - (1.5 * atr)
    take_profit_1 = price + (1.0 * atr)
    take_profit_2 = price + (2.0 * atr)

    # إدارة رأس المال: المخاطرة بنسبة 1% من الرصيد
    risk_pct = 0.01
    risk_amount = usdt_balance * risk_pct
    amount = risk_amount / (price - stop_loss)

    order = place_market_order(symbol, 'buy', amount)

    position = {
        "symbol": symbol,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "tp1": take_profit_1,
        "tp2": take_profit_2
    }

    save_position(symbol, position)
    return order, f"✅ شراء {symbol} @ {price:.4f}\n🎯 TP1: {take_profit_1:.4f} | TP2: {take_profit_2:.4f} | ❌ SL: {stop_loss:.4f}"

# ========================== إدارة الصفقات ==========================

def manage_position(symbol, send_message):
    """
    - إذا وصل السعر لوقف الخسارة يتم البيع وإغلاق المركز
    - إذا وصل السعر لهدف أول (TP1) يتم بيع نصف الكمية وتحريك وقف الخسارة لنقطة الدخول
    - إذا وصل السعر لهدف ثاني (TP2) يتم بيع الباقي وإغلاق المركز
    """
    position = load_position(symbol)
    if not position:
        return

    current_price = fetch_price(symbol)
    amount = position['amount']

    # TP1
    if current_price >= position['tp1'] and 'tp1_hit' not in position:
        place_market_order(symbol, 'sell', amount * 0.5)
        position['tp1_hit'] = True
        position['stop_loss'] = position['entry_price']  # نقل وقف الخسارة لنقطة الدخول
        save_position(symbol, position)
        send_message(f"🎯 تم تحقيق الهدف الأول TP1 لـ {symbol} عند {current_price:.4f} وتم بيع نصف الكمية")

    # TP2
    elif current_price >= position['tp2']:
        place_market_order(symbol, 'sell', amount * 0.5)
        clear_position(symbol)
        send_message(f"✅ تم تحقيق الهدف الثاني TP2 لـ {symbol} عند {current_price:.4f} وتم إغلاق الصفقة بالكامل")

    # Stop Loss
    elif current_price <= position['stop_loss']:
        place_market_order(symbol, 'sell', amount)
        clear_position(symbol)
        send_message(f"❌ تم وقف الخسارة لـ {symbol} عند {current_price:.4f}")
