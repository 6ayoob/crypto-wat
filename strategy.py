import pandas as pd
import numpy as np
import json
import os
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from datetime import datetime, timedelta
import requests
import time

POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

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

def calculate_indicators(df):
    df['ema9'] = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    return df

def check_signal(symbol):
    data_5m = fetch_ohlcv(symbol, '5m', 100)
    if not data_5m:
        return None

    df = pd.DataFrame(data_5m, columns=['timestamp','open','high','low','close','volume'])
    df = calculate_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # شرط شراء مبسط: EMA9 تعبر EMA21 من الأسفل للأعلى
    if (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']):
        return "buy"
    return None

def execute_buy(symbol):
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return None, f"🚫 وصلت للحد الأقصى للصفقات المفتوحة ({MAX_OPEN_POSITIONS})."

    price = fetch_price(symbol)
    usdt_balance = fetch_balance('USDT')

    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, f"🚫 رصيد USDT غير كافٍ لشراء {symbol}."

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, 'buy', amount)

    stop_loss = price * 0.98  # وقف خسارة 2% تحت سعر الدخول (أكثر أمان)
    take_profit = price * 1.04  # هدف ربح 4%

    position = {
        "symbol": symbol,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }

    save_position(symbol, position)
    return order, f"✅ تم شراء {symbol} بسعر {price:.4f}\n🎯 هدف الربح: {take_profit:.4f} (+4%) | 🛑 وقف الخسارة: {stop_loss:.4f} (-2%)"

def manage_position(symbol):
    position = load_position(symbol)
    if not position:
        return

    current_price = fetch_price(symbol)
    amount = position['amount']
    entry_price = position['entry_price']

    base_asset = symbol.split('/')[0]
    actual_balance = fetch_balance(base_asset)
    sell_amount = min(amount, actual_balance)
    sell_amount = round(sell_amount, 6)

    if current_price >= position['take_profit']:
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
            send_telegram_message(f"🏆 تم تحقيق هدف الربح لـ {symbol} عند {current_price:.4f} | الصفقة مغلقة ✅")
        else:
            send_telegram_message(f"❌ فشل تنفيذ أمر البيع لـ {symbol} عند هدف الربح")
        return

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
            send_telegram_message(f"❌ تم ضرب وقف الخسارة لـ {symbol} عند {current_price:.4f} | الصفقة مغلقة 🚫")
        else:
            send_telegram_message(f"❌ فشل تنفيذ أمر البيع لـ {symbol} عند وقف الخسارة")
        return

if __name__ == "__main__":
    send_telegram_message("🚀 بدأ البوت بمراقبة الأسواق باستخدام استراتيجية أبسط وأكثر أمانًا ✅")
    last_report_date = None

    while True:
        try:
            for symbol in SYMBOLS:
                position = load_position(symbol)

                if position is None:
                    signal = check_signal(symbol)
                    if signal == "buy":
                        order, message = execute_buy(symbol)
                        if message:
                            send_telegram_message(message)
                else:
                    manage_position(symbol)  # استدعاء بديل

        except Exception as e:
            import traceback
            send_telegram_message(f"⚠️ خطأ في main.py:\n{traceback.format_exc()}")

        time.sleep(60)
