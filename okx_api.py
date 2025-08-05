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
    """
    جلب رصيد العملة المتاحة في الحساب
    """
    try:
        balances = exchange.fetch_balance()
        return balances.get(asset, {}).get('free', 0)
    except Exception as e:
        print(f"❌ خطأ في جلب الرصيد لـ {asset}: {e}")
        return 0

def fetch_price(symbol):
    """
    جلب السعر الحالي لسوق معين
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"❌ خطأ في جلب السعر الحالي لـ {symbol}: {e}")
        return None

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    """
    جلب بيانات الشموع (OHLCV) لفترة زمنية معينة
    """
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return data
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {symbol}: {e}")
        return []

def place_market_order(symbol, side, amount):
    """
    تنفيذ أمر سوق (شراء أو بيع) بمقدار معين
    """
    try:
        order = exchange.create_market_order(symbol, side, amount)
        return order
    except Exception as e:
        print(f"❌ خطأ في تنفيذ أمر السوق ({side}) لـ {symbol}: {e}")
        return None

def place_market_buy(symbol, usdt_amount):
    """
    تنفيذ أمر شراء بقيمة USDT معينة، بحساب كمية العملة تلقائيًا
    """
    price = fetch_price(symbol)
    if price is None:
        print(f"❌ لا يمكن تحديد السعر للشراء لـ {symbol}")
        return None
    try:
        quantity = round(usdt_amount / price, 6)
        order = exchange.create_market_buy_order(symbol, quantity)
        return order
    except Exception as e:
        print(f"❌ فشل أمر الشراء لـ {symbol}: {e}")
        return None

def place_market_sell(symbol, quantity):
    """
    تنفيذ أمر بيع لكمية معينة من العملة
    """
    try:
        order = exchange.create_market_sell_order(symbol, quantity)
        return order
    except Exception as e:
        print(f"❌ فشل أمر البيع لـ {symbol}: {e}")
        return None

def get_position_size(symbol):
    """
    جلب كمية العملة المتاحة (الرصيد الحر) للمركز المفتوح
    """
    try:
        balance = exchange.fetch_balance()
        coin = symbol.split("-")[0]
        free = balance['free'].get(coin, 0)
        return free
    except Exception as e:
        print(f"❌ خطأ في جلب حجم المركز لـ {symbol}: {e}")
        return 0
