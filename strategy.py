# strategy.py

import pandas as pd
import json
import os
from okx_api import fetch_ohlcv, fetch_price, place_market_order, get_balance
from config import SYMBOL, TIMEFRAME, TRADE_AMOUNT_USDT, STOP_LOSS_PCT, TAKE_PROFIT_PCT

POSITIONS_FILE = "positions.json"

def load_position():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return None

def save_position(position):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(position, f)

def clear_position():
    if os.path.exists(POSITIONS_FILE):
        os.remove(POSITIONS_FILE)

def check_signal():
    data = fetch_ohlcv(SYMBOL, TIMEFRAME, 100)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()

    if df['ma20'].iloc[-2] < df['ma50'].iloc[-2] and df['ma20'].iloc[-1] > df['ma50'].iloc[-1]:
        return "buy"
    return None

def execute_buy():
    price = fetch_price(SYMBOL)
    usdt_balance = get_balance('USDT')

    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, "🚫 لا يوجد رصيد كافي"

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(SYMBOL, 'buy', amount)

    stop_loss = price * (1 - STOP_LOSS_PCT)
    take_profit = price * (1 + TAKE_PROFIT_PCT)

    position = {
        "symbol": SYMBOL,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

    save_position(position)
    return order, f"✅ صفقة شراء مفتوحة @ {price:.2f}\n🎯 هدف: {take_profit:.2f}\n❌ وقف: {stop_loss:.2f}"

def manage_position(send_message):
    position = load_position()
    if not position:
        return

    current_price = fetch_price(SYMBOL)

    if current_price <= position['stop_loss']:
        place_market_order(SYMBOL, 'sell', position['amount'])
        clear_position()
        send_message(f"❌ تم ضرب وقف الخسارة عند {current_price:.2f}")
    elif current_price >= position['take_profit']:
        place_market_order(SYMBOL, 'sell', position['amount'])
        clear_position()
        send_message(f"🎯 تم تحقيق هدف الربح عند {current_price:.2f}")
