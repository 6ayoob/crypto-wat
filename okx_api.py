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

def is_symbol_available(symbol):
    try:
        markets = exchange.load_markets()
        return format_symbol(symbol) in markets
    except Exception as e:
        print(f"❌ خطأ في تحميل الأسواق من OKX: {e}")
        return False

def fetch_balance(asset='USDT'):
    try:
        balances = exchange.fetch_balance()
        return balances.get(asset, {}).get('free', 0)
    except Exception as e:
        print(f"❌ خطأ في جلب الرصيد لـ {asset}: {e}")
        return 0

def fetch_price(symbol):
    symbol_formatted = format_symbol(symbol)
    if not is_symbol_available(symbol):
        print(f"❌ السوق غير متوفر على OKX: {symbol_formatted}")
        return None
    try:
        ticker = exchange.fetch_ticker(symbol_formatted)
        return ticker['last']
    except Exception as e:
        print(f"❌ خطأ في جلب السعر الحالي لـ {symbol_formatted}: {e}")
        return None

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    symbol_formatted = format_symbol(symbol)
    if not is_symbol_available(symbol):
        print(f"❌ السوق غير متوفر على OKX: {symbol_formatted}")
        return []
    try:
        data = exchange.fetch_ohlcv(symbol_formatted, timeframe=timeframe, limit=limit)
        return data
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {symbol_formatted}: {e}")
        return []

def place_market_order(symbol, side, amount):
    symbol_formatted = format_symbol(symbol)
    if not is_symbol_available(symbol):
        print(f"❌ لا يمكن تنفيذ الأمر، السوق غير متوفر: {symbol_formatted}")
        return None
    if side.lower() == "sell":
        base_asset = symbol.split("-")[0]
        balance = fetch_balance(base_asset)
        if balance <= 0:
            print(f"❌ لا يوجد رصيد متاح لـ {base_asset} لتنفيذ البيع.")
            return None
        amount = min(amount, balance * 0.99)
        if amount <= 0:
            print(f"❌ الكمية غير صالحة للبيع بعد التحقق من الرصيد لـ {base_asset}.")
            return None
    try:
        order = exchange.create_market_order(symbol_formatted, side, amount)
        print(f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol_formatted} بكمية {amount}")
        return order
    except Exception as e:
        print(f"❌ خطأ في تنفيذ أمر السوق ({side}) لـ {symbol_formatted}: {e}")
        return None
