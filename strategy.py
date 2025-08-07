import pandas as pd
import json
import os
import time
import logging
from datetime import datetime
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS

logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"

def get_position_filename(symbol):
    symbol = symbol.replace("/", "_")
    return f"{POSITIONS_DIR}/{symbol}.json"

def load_position(symbol):
    file = get_position_filename(symbol)
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"ملف صفقة تالف: {file}")
            return None
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
        try:
            with open(CLOSED_POSITIONS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("ملف الصفقات المغلقة تالف")
            return []
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

def calculate_indicators(df):
    df['ema9'] = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    return df

def check_signal(symbol):
    data_1m = fetch_ohlcv(symbol, '1m', 200)
    data_5m = fetch_ohlcv(symbol, '5m', 200)
    if not data_1m or not data_5m:
        return None

    df1 = pd.DataFrame(data_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df5 = pd.DataFrame(data_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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
    if price is None:
        return None, f"🚫 فشل جلب سعر {symbol}"

    usdt_balance = fetch_balance('USDT')
    if usdt_balance is None or usdt_balance < TRADE_AMOUNT_USDT:
        return None, f"🚫 لا يوجد رصيد كافٍ لشراء {symbol}: متاح = {usdt_balance}, مطلوب = {TRADE_AMOUNT_USDT}"

    amount = TRADE_AMOUNT_USDT / price
    amount = round(amount, 6)

    max_retries = 3
    for attempt in range(max_retries):
        order = place_market_order(symbol, 'buy', amount)
        if order:
            break
        time.sleep(1)
    else:
        return None, f"❌ فشل تنفيذ أمر الشراء لـ {symbol} بعد {max_retries} محاولات"

    stop_loss = round(price * 0.97, 6)
    tp1 = round(price * 1.03, 6)
    tp2 = round(price * 1.06, 6)

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
    message = f"✅ شراء {symbol} @ {price:.4f}\n🎯 TP1: {tp1:.4f} (+3%) | 🏆 TP2: {tp2:.4f} (+6%) | ❌ SL: {stop_loss:.4f} (-3%)"
    return order, message

def manage_position(symbol, send_message):
    position = load_position(symbol)
    if not position or 'amount' not in position or position['amount'] <= 0:
        send_message(f"❌ بيانات الصفقة غير صالحة لـ {symbol}")
        clear_position(symbol)
        return

    current_price = fetch_price(symbol)
    if current_price is None:
        send_message(f"❌ فشل جلب السعر الحالي لـ {symbol}")
        return

    amount = position['amount']
    entry_price = position['entry_price']

    base_asset = symbol.replace("-", "/").split('/')[0]
    actual_balance = fetch_balance(base_asset)

    sell_amount = min(amount, actual_balance)
    sell_amount = round(sell_amount, 6)

    # تحقق من TP1
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
            time.sleep(3)
        else:
            send_message(f"❌ فشل تنفيذ أمر البيع الجزئي لـ {symbol} عند TP1")
            return

    # Trailing Stop
    if position.get('trailing_active'):
        new_sl = round(current_price * 0.99, 4)
        if new_sl > position['stop_loss']:
            position['stop_loss'] = new_sl
            save_position(symbol, position)

    # تحقق من TP2
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
            time.sleep(3)
        else:
            send_message(f"❌ فشل تنفيذ أمر البيع الكامل لـ {symbol} عند TP2")
        return

    # تحقق من وقف الخسارة
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
            time.sleep(3)
        else:
            send_message(f"❌ فشل تنفيذ أمر البيع الكامل لـ {symbol} عند وقف الخسارة")
