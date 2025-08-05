# okx_api.py

import ccxt
from config import API_KEY, API_SECRET, API_PASSWORD

# إعداد الاتصال بـ OKX
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSWORD,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

def get_balance(usdt_only=True):
    balance = exchange.fetch_balance()
    if usdt_only:
        return balance['total'].get('USDT', 0)
    return balance['total']

def fetch_ohlcv(symbol, timeframe='15m', limit=100):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {symbol}: {e}")
        return []

def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"❌ خطأ في جلب السعر الحالي لـ {symbol}: {e}")
        return None

def place_market_buy(symbol, usdt_amount):
    try:
        price = get_current_price(symbol)
        if price is None:
            return None

        quantity = round(usdt_amount / price, 6)
        order = exchange.create_market_buy_order(symbol, quantity)
        return order
    except Exception as e:
        print(f"❌ فشل أمر الشراء لـ {symbol}: {e}")
        return None

def place_market_sell(symbol, quantity):
    try:
        order = exchange.create_market_sell_order(symbol, quantity)
        return order
    except Exception as e:
        print(f"❌ فشل أمر البيع لـ {symbol}: {e}")
        return None

def get_position_size(symbol):
    balance = exchange.fetch_balance()
    coin = symbol.split("-")[0]
    free = balance['free'].get(coin, 0)
    return free
