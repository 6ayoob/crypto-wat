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
