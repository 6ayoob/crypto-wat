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
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return data
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {symbol}: {e}")
        return []

def place_market_order(symbol, side, amount, send_message=None):
    symbol_ccxt = format_symbol(symbol)

    if amount <= 0:
        msg = f"⚠️ الكمية صفر أو غير صالحة عند محاولة تنفيذ أمر {side} لـ {symbol}"
        print(msg)
        if send_message:
            send_message(msg)
        return None

    try:
        order = exchange.create_market_order(symbol_ccxt, side, amount)
        msg = f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol} بنجاح ✅"
        print(msg)
        if send_message:
            send_message(msg)
        return order
    except Exception as e:
        msg = f"❌ فشل تنفيذ أمر السوق ({side}) لـ {symbol}: {str(e)}"
        print(msg)
        if send_message:
            send_message(msg)
        return None
