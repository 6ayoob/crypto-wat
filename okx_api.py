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

def format_symbol(symbol):
    return symbol.replace("-", "/")

def fetch_balance(asset='USDT'):
    try:
        balances = exchange.fetch_balance()
        return balances.get(asset, {}).get('free', 0)
    except Exception as e:
        print(f"❌ خطأ في جلب الرصيد لـ {asset}: {e}")
        return 0

def fetch_price(symbol):
    symbol = format_symbol(symbol)
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"❌ خطأ في جلب السعر الحالي لـ {symbol}: {e}")
        return None

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    symbol = format_symbol(symbol)
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {symbol}: {e}")
        return []

def place_market_order(symbol, side, amount):
    symbol_formatted = format_symbol(symbol)

    if side.lower() == "sell":
        base_asset = symbol.split("-")[0]
        actual_balance = fetch_balance(base_asset)
        if actual_balance == 0:
            error = f"❌ لا يوجد رصيد متاح لـ {base_asset} لتنفيذ أمر البيع لـ {symbol}"
            print(error)
            return None, error
        amount = min(amount, actual_balance * 0.99)
        if amount <= 0:
            error = f"❌ الكمية غير كافية للبيع بعد التحقق من الرصيد لـ {base_asset}"
            print(error)
            return None, error

    try:
        order = exchange.create_market_order(symbol_formatted, side, amount)
        return order, None
    except Exception as e:
        error = f"❌ خطأ في تنفيذ أمر السوق ({side}) لـ {symbol_formatted}: {e}"
        print(error)
        return None, error
