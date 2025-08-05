import pandas as pd
import json
import os
from okx_api import fetch_ohlcv, fetch_price, place_market_order, get_balance
from config import TRADE_AMOUNT_USDT, STOP_LOSS_PCT, TAKE_PROFIT_PCT

MAX_OPEN_POSITIONS = 4  # الحد الأقصى للصفقات المفتوحة

def get_position_filename(symbol):
    symbol = symbol.replace("/", "_")
    return f"positions/{symbol}.json"

def load_position(symbol):
    file = get_position_filename(symbol)
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_position(symbol, position):
    os.makedirs("positions", exist_ok=True)
    file = get_position_filename(symbol)
    with open(file, 'w') as f:
        json.dump(position, f)

def clear_position(symbol):
    file = get_position_filename(symbol)
    if os.path.exists(file):
        os.remove(file)

def count_open_positions():
    os.makedirs("positions", exist_ok=True)
    return len([f for f in os.listdir("positions") if f.endswith(".json")])

def check_signal(symbol):
    data = fetch_ohlcv(symbol, '5m', 100)  # تعديل الإطار الزمني إلى 5 دقائق
    if not data:
        return None
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()

    # إشارة شراء: تقاطع الما20 مع الما50 صاعدًا
    if df['ma20'].iloc[-2] < df['ma50'].iloc[-2] and df['ma20'].iloc[-1] > df['ma50'].iloc[-1]:
        return "buy"
    return None

def execute_buy(symbol):
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
