import ccxt
import time
import datetime
from config import (
    API_KEY, SECRET_KEY, PASSPHRASE, SYMBOLS, TIMEFRAME,
    TRADE_AMOUNT_USDT, STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_OPEN_POSITIONS
)

# تهيئة الاتصال بـ OKX
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# قائمة الصفقات المفتوحة
open_positions = {}

def get_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']

def get_candles(symbol, timeframe="3m", limit=100):
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return candles
    except Exception as e:
        print(f"❌ خطأ في جلب البيانات لـ {symbol}: {e}")
        return []

def calculate_signal(candles):
    if len(candles) < 20:
        return None
    close_prices = [c[4] for c in candles]  # سعر الإغلاق
    ma5 = sum(close_prices[-5:]) / 5
    ma10 = sum(close_prices[-10:]) / 10
    if ma5 > ma10:
        return "buy"
    return None

def place_order(symbol, side, amount):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol}")
        return order
    except Exception as e:
        print(f"❌ فشل تنفيذ أمر {side} لـ {symbol}: {e}")
        return None

def monitor_positions():
    to_remove = []
    for symbol, pos in open_positions.items():
        current_price = get_price(symbol)
        entry_price = pos['entry_price']
        side = pos['side']
        
        # احتساب الأرباح والخسائر
        if side == 'buy':
            change = (current_price - entry_price) / entry_price
        else:
            change = (entry_price - current_price) / entry_price

        if change >= TAKE_PROFIT_PCT:
            print(f"🎯 جني أرباح {symbol} عند {current_price}")
            place_order(symbol, 'sell', pos['amount'])
            to_remove.append(symbol)

        elif change <= -STOP_LOSS_PCT:
            print(f"🚨 وقف خسارة {symbol} عند {current_price}")
            place_order(symbol, 'sell', pos['amount'])
            to_remove.append(symbol)

    for symbol in to_remove:
        del open_positions[symbol]

def run_strategy():
    while True:
        try:
            if len(open_positions) < MAX_OPEN_POSITIONS:
                for symbol in SYMBOLS:
                    if symbol in open_positions:
                        continue

                    candles = get_candles(symbol, timeframe=TIMEFRAME)
                    signal = calculate_signal(candles)
                    if signal == "buy":
                        price = get_price(symbol)
                        amount = round(TRADE_AMOUNT_USDT / price, 5)
                        order = place_order(symbol, 'buy', amount)
                        if order:
                            open_positions[symbol] = {
                                'entry_price': price,
                                'amount': amount,
                                'side': 'buy',
                                'time': datetime.datetime.now()
                            }

                    if len(open_positions) >= MAX_OPEN_POSITIONS:
                        break

            monitor_positions()
            time.sleep(60)  # راقب كل دقيقة
        except Exception as e:
            print(f"⚠️ خطأ عام: {e}")
            time.sleep(60)

