import pandas as pd
import numpy as np
import json
import os
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS
from datetime import datetime, timedelta

POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"

def get_position_filename(symbol):
    symbol = symbol.replace("/", "_")
    return f"{POSITIONS_DIR}/{symbol}.json"

def load_position(symbol):
    file = get_position_filename(symbol)
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_position(symbol, position):
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    file = get_position_filename(symbol)
    with open(file, 'w') as f:
        json.dump(position, f)

def clear_position(symbol):
    file = get_position_filename(symbol)
    if os.path.exists(file):
        os.remove(file)

def count_open_positions():
    os.makedirs(POSITIONS_DIR, exist_ok=True)
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions():
    if os.path.exists(CLOSED_POSITIONS_FILE):
        with open(CLOSED_POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_closed_positions(closed_positions):
    with open(CLOSED_POSITIONS_FILE, 'w') as f:
        json.dump(closed_positions, f, indent=2)

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_indicators(df):
    df['ema9'] = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    df['atr'] = atr(df, 14)
    return df

def check_signal(symbol):
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
               and (df5['ema9'].iloc[-1] > df5['ema21'].iloc[-1])

    return "buy" if cond_buy else None

def execute_buy(symbol):
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, f"🚫 الحد الأقصى للصفقات ({MAX_OPEN_POSITIONS}) مفتوح بالفعل."

    price = fetch_price(symbol)
    usdt_balance = fetch_balance('USDT')

    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, f"🚫 لا يوجد رصيد كافي لشراء {symbol}"

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, 'buy', amount)

    stop_loss = price * 0.97  # وقف خسارة 3% تحت سعر الدخول
    tp1 = price * 1.03        # هدف أول: +3%
    tp2 = price * 1.06        # هدف ثاني: +6%

    position = {
        "symbol": symbol,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp1_hit": False,
        "trailing_active": False
    }

    save_position(symbol, position)
    return order, f"✅ شراء {symbol} @ {price:.4f}\n🎯 TP1: {tp1:.4f} (+3%) | 🏆 TP2: {tp2:.4f} (+6%) | ❌ SL: {stop_loss:.4f} (-3%)"

def manage_position(symbol, send_message):
    position = load_position(symbol)
    if not position or 'amount' not in position or position['amount'] <= 0:
        send_message(f"❌ بيانات الصفقة غير صالحة لـ {symbol}")
        clear_position(symbol)
        return

    current_price = fetch_price(symbol)
    amount = position['amount']
    entry_price = position['entry_price']

    base_asset = symbol.split('/')[0]
    actual_balance = fetch_balance(base_asset)

    sell_amount = min(amount, actual_balance)
    sell_amount = round(sell_amount, 6)

    # ✅ تحقق من TP1
    if current_price >= position['tp1'] and not position.get('tp1_hit'):
        sell_amount_half = round(sell_amount * 0.5, 6)
        if sell_amount_half <= 0 or sell_amount_half > actual_balance:
            send_message(f"❌ الكمية غير كافية للبيع الجزئي لـ {symbol}: رصيد متاح = {actual_balance}, مطلوب = {sell_amount_half}")
            return
        order = place_market_order(symbol, 'sell', sell_amount_half)
        if order:
            position['amount'] -= sell_amount_half
            position['tp1_hit'] = True
            position['stop_loss'] = entry_price
            position['trailing_active'] = True
            save_position(symbol, position)
            send_message(f"🎯 تم تحقيق TP1 لـ {symbol} عند {current_price:.4f} | بيع نصف الكمية ✅ وتحريك وقف الخسارة لنقطة الدخول")
        else:
            send_message(f"❌ فشل تنفيذ أمر البيع الجزئي لـ {symbol} عند TP1")
            return

    # ✅ Trailing Stop
    if position.get('trailing_active'):
        new_sl = round(current_price * 0.99, 4)
        if new_sl > position['stop_loss']:
            position['stop_loss'] = new_sl
            save_position(symbol, position)

    # ✅ تحقق من TP2
    if current_price >= position['tp2']:
        order = place_market_order(symbol, 'sell', sell_amount)
        if order:
            profit = (current_price - entry_price) * sell_amount
            closed_positions = load_closed_positions()
            closed_positions.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": current_price,
                "amount": sell_amount,
                "profit": profit,
                "closed_at": datetime.utcnow().isoformat()
            })
            save_closed_positions(closed_positions)
            clear_position(symbol)
            send_message(f"🏆 تم تحقيق TP2 لـ {symbol} عند {current_price:.4f} | الصفقة مغلقة بالكامل ✅")
        else:
            send_message(f"❌ فشل تنفيذ أمر البيع الكامل لـ {symbol} عند TP2")
        return

    # ✅ تحقق من وقف الخسارة
    if current_price <= position['stop_loss']:
        order = place_market_order(symbol, 'sell', sell_amount)
        if order:
            profit = (current_price - entry_price) * sell_amount
            closed_positions = load_closed_positions()
            closed_positions.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": current_price,
                "amount": sell_amount,
                "profit": profit,
                "closed_at": datetime.utcnow().isoformat()
            })
            save_closed_positions(closed_positions)
            clear_position(symbol)
            send_message(f"❌ تم ضرب وقف الخسارة لـ {symbol} عند {current_price:.4f} | الصفقة مغلقة 🚫")
        else:
            send_message(f"❌ فشل تنفيذ أمر البيع الكامل لـ {symbol} عند وقف الخسارة")
