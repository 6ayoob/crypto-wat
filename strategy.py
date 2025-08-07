import pandas as pd
import numpy as np
import json
import os
import time
import logging
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance, get_instrument_info
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS
from datetime import datetime, timedelta

# إعداد التسجيل
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

def get_instrument_constraints(symbol):
    """جلب قيود التداول للزوج من OKX"""
    try:
        info = get_instrument_info(symbol)
        return {
            'min_order_size': float(info.get('minSz', 0.0001)),
            'quantity_precision': int(-np.log10(float(info.get('lotSz', 0.0001))))
        }
    except Exception as e:
        logging.error(f"فشل جلب قيود الزوج {symbol}: {str(e)}")
        return {'min_order_size': 0.0001, 'quantity_precision': 6}

def check_api_status():
    """التحقق من حالة اتصال OKX API"""
    try:
        # افتراضي: استدعاء API للتحقق من الحالة
        response = okx_api.get_system_status()  # يجب أن تكون متوفرة في okx_api
        return response.get('status') == 'ok'
    except Exception as e:
        logging.error(f"فشل التحقق من حالة API: {str(e)}")
        return False

def check_signal(symbol):
    if not check_api_status():
        logging.warning(f"لا يمكن التحقق من إشارة لـ {symbol}: مشكلة اتصال API")
        return None

    data_1m = fetch_ohlcv(symbol, '1m', 200)
    data_5m = fetch_ohlcv(symbol, '5m', 200)
    if not data_1m or not data_5m:
        logging.error(f"فشل جلب بيانات OHLCV لـ {symbol}")
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
        message = f"🚫 الحد الأقصى للصفقات ({MAX_OPEN_POSITIONS}) مفتوح بالفعل."
        logging.info(message)
        return None, message

    if not check_api_status():
        message = f"🚫 فشل الشراء لـ {symbol}: مشكلة اتصال API"
        logging.error(message)
        return None, message

    price = fetch_price(symbol)
    if price is None:
        message = f"🚫 فشل جلب سعر {symbol}"
        logging.error(message)
        return None, message

    usdt_balance = fetch_balance('USDT')
    if usdt_balance is None or usdt_balance < TRADE_AMOUNT_USDT:
        message = f"🚫 لا يوجد رصيد كافٍ لشراء {symbol}: متاح = {usdt_balance}, مطلوب = {TRADE_AMOUNT_USDT}"
        logging.error(message)
        return None, message

    constraints = get_instrument_constraints(symbol)
    amount = TRADE_AMOUNT_USDT / price
    amount = max(constraints['min_order_size'], round(amount, constraints['quantity_precision']))

    max_retries = 3
    for attempt in range(max_retries):
        order, error = place_market_order(symbol, 'buy', amount)
        if order:
            break
        logging.error(f"محاولة {attempt+1} فشلت لشراء {symbol}: {error}")
        time.sleep(1)
    else:
        message = f"❌ فشل تنفيذ أمر الشراء لـ {symbol} بعد {max_retries} محاولات"
        logging.error(message)
        return None, message

    stop_loss = round(price * 0.97, constraints['quantity_precision'])
    tp1 = round(price * 1.03, constraints['quantity_precision'])
    tp2 = round(price * 1.06, constraints['quantity_precision'])

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
    logging.info(message)
    return order, message

def manage_position(symbol, send_message):
    position = load_position(symbol)
    required_keys = ['amount', 'entry_price', 'stop_loss', 'tp1', 'tp2', 'tp1_hit', 'trailing_active']
    if not position or not all(key in position for key in required_keys) or position['amount'] <= 0:
        message = f"❌ بيانات الصفقة غير صالحة لـ {symbol}"
        send_message(message)
        logging.error(message)
        clear_position(symbol)
        return

    if not check_api_status():
        message = f"🚫 فشل إدارة الصفقة لـ {symbol}: مشكلة اتصال API"
        send_message(message)
        logging.error(message)
        return

    current_price = fetch_price(symbol)
    if current_price is None:
        message = f"❌ فشل جلب السعر الحالي لـ {symbol}"
        send_message(message)
        logging.error(message)
        return

    amount = position['amount']
    entry_price = position['entry_price']
    base_asset = symbol.split('/')[0]
    actual_balance = fetch_balance(base_asset)
    if actual_balance is None:
        message = f"❌ فشل جلب رصيد {base_asset} لـ {symbol}"
        send_message(message)
        logging.error(message)
        return

    constraints = get_instrument_constraints(symbol)
    sell_amount = max(constraints['min_order_size'], round(min(amount, actual_balance), constraints['quantity_precision']))

    # ✅ تحقق من TP1
    if current_price >= position['tp1'] and not position.get('tp1_hit'):
        sell_amount_half = max(constraints['min_order_size'], round(sell_amount * 0.5, constraints['quantity_precision']))
        if sell_amount_half <= 0 or sell_amount_half > actual_balance:
            message = f"❌ الكمية غير كافية للبيع الجزئي لـ {symbol}: رصيد متاح = {actual_balance}, مطلوب = {sell_amount_half}"
            send_message(message)
            logging.error(message)
            return

        max_retries = 3
        for attempt in range(max_retries):
            # استخدام أمر محدود بدلاً من سوقي لتحسين التنفيذ
            order, error = place_limit_order(symbol, 'sell', sell_amount_half, position['tp1'])
            if order:
                break
            logging.error(f"محاولة {attempt+1} فشلت لبيع جزئي لـ {symbol} عند TP1: {error}")
            send_message(f"❌ محاولة {attempt+1} فشلت لبيع جزئي لـ {symbol} عند TP1: {error}")
            time.sleep(1)
        else:
            message = f"❌ فشل تنفيذ أمر البيع الجزئي لـ {symbol} عند TP1 بعد {max_retries} محاولات"
            send_message(message)
            logging.error(message)
            return

        position['amount'] -= sell_amount_half
        position['tp1_hit'] = True
        position['stop_loss'] = entry_price
        position['trailing_active'] = True
        save_position(symbol, position)
        message = f"🎯 تم تحقيق TP1 لـ {symbol} عند {current_price:.4f} | بيع نصف الكمية ✅ وتحريك وقف الخسارة لنقطة الدخول"
        send_message(message)
        logging.info(message)
        return

    # ✅ Trailing Stop
    if position.get('trailing_active'):
        new_sl = round(current_price * 0.99, constraints['quantity_precision'])
        if new_sl > position['stop_loss']:
            position['stop_loss'] = new_sl
            save_position(symbol, position)

    # ✅ تحقق من TP2
    if current_price >= position['tp2']:
        max_retries = 3
        for attempt in range(max_retries):
            order, error = place_market_order(symbol, 'sell', sell_amount)
            if order:
                break
            logging.error(f"محاولة {attempt+1} فشلت لبيع كامل لـ {symbol} عند TP2: {error}")
            send_message(f"❌ محاولة {attempt+1} فشلت لبيع كامل لـ {symbol} عند TP2: {error}")
            time.sleep(1)
        else:
            message = f"❌ فشل تنفيذ أمر البيع الكامل لـ {symbol} عند TP2 بعد {max_retries} محاولات"
            send_message(message)
            logging.error(message)
            return

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
        message = f"🏆 تم تحقيق TP2 لـ {symbol} عند {current_price:.4f} | الصفقة مغلقة بالكامل ✅"
        send_message(message)
        logging.info(message)
        return

    # ✅ تحقق من وقف الخسارة
    if current_price <= position['stop_loss']:
        max_retries = 3
        for attempt in range(max_retries):
            order, error = place_market_order(symbol, 'sell', sell_amount)
            if order:
                break
            logging.error(f"محاولة {attempt+1} فشلت لبيع كامل لـ {symbol} عند SL: {error}")
            send_message(f"❌ محاولة {attempt+1} فشلت لبيع كامل لـ {symbol} عند SL: {error}")
            time.sleep(1)
        else:
            message = f"❌ فشل تنفيذ أمر البيع الكامل لـ {symbol} عند وقف الخسارة بعد {max_retries} محاولات"
            send_message(message)
            logging.error(message)
            return

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
        message = f"❌ تم ضرب وقف الخسارة لـ {symbol} عند {current_price:.4f} | الصفقة مغلقة 🚫"
        send_message(message)
        logging.info(message)

# اختياري: إشعارات Telegram
# def send_telegram_message(message, chat_id, bot_token):
#     import requests
#     url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
#     payload = {'chat_id': chat_id, 'text': message}
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status()
#     except Exception as e:
#         logging.error(f"فشل إرسال إشعار Telegram: {str(e)}")
#
# def send_message_wrapper(message):
#     send_message(message)  # الدالة الأصلية الممررة إلى manage_position
#     send_telegram_message(message, "YOUR_CHAT_ID", "YOUR_BOT_TOKEN")
