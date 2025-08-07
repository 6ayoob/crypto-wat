import pandas as pd
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS
from datetime import datetime
import requests

def send_telegram_message(text, token=None, chat_id=None):
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": chat_id, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df['ema9'] = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    return df

def check_signal(symbol):
    data_5m = fetch_ohlcv(symbol, '5m', 100)
    if not data_5m:
        return None

    df = pd.DataFrame(data_5m, columns=['timestamp','open','high','low','close','volume'])
    df = calculate_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # شراء عندما يعبر EMA9 فوق EMA21 و RSI > 50
    if (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']) and (last['rsi'] > 50):
        return "buy"
    return None

def execute_buy(symbol):
    price = fetch_price(symbol)
    usdt_balance = fetch_balance('USDT')

    if usdt_balance < TRADE_AMOUNT_USDT:
        return None, f"🚫 رصيد USDT غير كافٍ لشراء {symbol}."

    amount = TRADE_AMOUNT_USDT / price
    order = place_market_order(symbol, 'buy', amount)

    stop_loss = price * 0.98  # وقف خسارة 2%
    take_profit = price * 1.04  # هدف ربح 4%

    position = {
        "symbol": symbol,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }

    return order, position, f"✅ تم شراء {symbol} بسعر {price:.4f}\n🎯 هدف الربح: {take_profit:.4f} (+4%) | 🛑 وقف الخسارة: {stop_loss:.4f} (-2%)"

def manage_position(position, token=None, chat_id=None):
    current_price = fetch_price(position['symbol'])
    amount = position['amount']
    entry_price = position['entry_price']

    base_asset = position['symbol'].split('/')[0]
    actual_balance = fetch_balance(base_asset)
    sell_amount = min(amount, actual_balance)
    sell_amount = round(sell_amount, 6)

    # تحقق هدف الربح
    if current_price >= position['take_profit']:
        order = place_market_order(position['symbol'], 'sell', sell_amount)
        if order:
            profit = (current_price - entry_price) * sell_amount
            send_telegram_message(f"🏆 تم تحقيق هدف الربح لـ {position['symbol']} عند {current_price:.4f} | الصفقة مغلقة ✅", token, chat_id)
            return None  # الصفقة أغلقت
        else:
            send_telegram_message(f"❌ فشل تنفيذ أمر البيع عند هدف الربح لـ {position['symbol']}", token, chat_id)
        return position  # الصفقة مستمرة

    # تحقق وقف الخسارة
    if current_price <= position['stop_loss']:
        order = place_market_order(position['symbol'], 'sell', sell_amount)
        if order:
            profit = (current_price - entry_price) * sell_amount
            send_telegram_message(f"❌ تم ضرب وقف الخسارة لـ {position['symbol']} عند {current_price:.4f} | الصفقة مغلقة 🚫", token, chat_id)
            return None  # الصفقة أغلقت
        else:
            send_telegram_message(f"❌ فشل تنفيذ أمر البيع عند وقف الخسارة لـ {position['symbol']}", token, chat_id)
        return position  # الصفقة مستمرة

    return position  # الصفقة مستمرة
