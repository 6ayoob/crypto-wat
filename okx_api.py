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
    # تحويل من CRV-USDT إلى CRV/USDT
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


def place_market_order(symbol, side, amount):
    symbol_formatted = format_symbol(symbol)

    # في حالة البيع، تأكد من الكمية المتاحة أولاً
    if side.lower() == "sell":
        base_asset = symbol.split("-")[0]
        actual_balance = fetch_balance(base_asset)

        if actual_balance == 0:
            print(f"❌ لا يوجد رصيد متاح لـ {base_asset} لتنفيذ أمر البيع لـ {symbol}")
            return None

        # استخدم أقل من الرصيد المتاح لتجنب الأخطاء بسبب الفروقات
        amount = min(amount, actual_balance * 0.99)

        if amount <= 0:
            print(f"❌ الكمية غير كافية للبيع بعد التحقق من الرصيد لـ {base_asset}")
            return None

    try:
        order = exchange.create_market_order(symbol_formatted, side, amount)
        return order
    except Exception as e:
        print(f"❌ خطأ في تنفيذ أمر السوق ({side}) لـ {symbol_formatted}: {e}")
        return None
