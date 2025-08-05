# okx_api.py

import ccxt
from config import API_KEY, SECRET_KEY, PASSPHRASE

exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

def get_balance(asset='USDT'):
    balances = exchange.fetch_balance()
    return balances.get(asset, {}).get('free', 0)

def place_market_order(symbol, side, amount):
    return exchange.create_market_order(symbol, side, amount)

def fetch_price(symbol):
    return exchange.fetch_ticker(symbol)['last']

def fetch_ohlcv(symbol, timeframe='1m', limit=100):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

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
