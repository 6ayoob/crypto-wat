import ccxt
from config import OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE

exchange = ccxt.okx({
    'apiKey': OKX_API_KEY,
    'secret': OKX_SECRET_KEY,
    'password': OKX_PASSPHRASE,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

def get_last_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def place_limit_order(symbol, side, price, amount):
    try:
        order = exchange.create_limit_order(symbol, side, amount, price)
        return order
    except Exception as e:
        print(f"Error placing limit order {side} {symbol}: {e}")
        return None

def place_market_order(symbol, side, amount):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        return order
    except Exception as e:
        print(f"Error placing market order {side} {symbol}: {e}")
        return None

def get_historical_candles(symbol, timeframe='1h', limit=100):
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return candles  # List of [timestamp, open, high, low, close, volume]
    except Exception as e:
        print(f"Error fetching candles for {symbol}: {e}")
        return None
