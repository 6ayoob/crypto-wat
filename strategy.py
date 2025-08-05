# strategy.py

import pandas as pd
import json
import os
from okx_api import fetch_ohlcv, fetch_price, place_market_order, get_balance
from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT

TRADE_AMOUNT_USDT = 20  # قيمة كل صفقة بالدولار
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

def get_open_positions_count():
    if not os.path.exists("positions"):
        return 0
    return len([f for f in os.listdir("positions") if f.endswith(".json")])

def check_signal(symbol, timeframe='15m'):
    if get_open_positions_count() >= MAX_OPEN_POSITIONS:
        return None  # لا نفتح صفقات جديدة

    data = fetch_ohlcv(symbol, timeframe, 100)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'], df['macd_signal'] = compute_macd(df['close'])

    # إشارات الدخول: تقاطع EMA + RSI خروج من التشبع البيعي + تقاطع MACD
    if (
        df['ema9'].iloc[-2] < df['ema21'].iloc[-2] and
        df['ema9'].iloc[-1] > df['ema21'].iloc[-1] and
        df['rsi'].iloc[-1] > 30 and
        df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
    ):
        return "buy"
    return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def execute_buy(symbol):
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
    return order, f"✅ شراء {symbol} @ {price:.4f}\n🎯 هدف: {take_profit:.4f} | ❌ وقف: {stop_loss:.4f}"

def manage_position(symbol, send_message):
    position = load_position(symbol)
    if not position:
        return

    current_price = fetch_price(symbol)

    if current_price <= position['stop_loss']:
        place_market_order(symbol, 'sell', position['amount'])
        clear_position(symbol)
        send_message(f"❌ تم وقف الخسارة لـ {symbol} عند {current_price:.4f}")

    elif current_price >= position['take_profit']:
        place_market_order(symbol, 'sell', position['amount'])
        clear_position(symbol)
        send_message(f"🎯 تم تحقيق هدف الربح لـ {symbol} عند {current_price:.4f}")
