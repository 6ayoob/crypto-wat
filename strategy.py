import ccxt
import json
import time
import requests
import datetime
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, API_KEY, API_SECRET, API_PASSWORD

# إعداد بورصة OKX باستخدام CCXT للتداول الحقيقي
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSWORD,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

# إعدادات التداول
POSITION_FILE = "positions.json"
TRADE_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TAKE_PROFIT_PERCENT = 4
STOP_LOSS_PERCENT = 1
TRADE_AMOUNT_USDT = 20

# إرسال إشعار إلى تيليجرام
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except:
        pass

# تحميل الصفقات من ملف JSON
def load_positions():
    try:
        with open(POSITION_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

# حفظ الصفقات إلى ملف JSON
def save_positions(positions):
    with open(POSITION_FILE, 'w') as f:
        json.dump(positions, f, indent=2)

# الحصول على بيانات الشموع (Candle Data)
def get_ohlcv(symbol, timeframe='1m', limit=50):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"خطأ في جلب البيانات لـ {symbol}: {e}")
        return []

# حساب المتوسط المتحرك البسيط
def sma(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

# التحقق من وجود إشارة شراء (تقاطع MA20 فوق MA50)
def check_entry_signal(symbol):
    candles = get_ohlcv(symbol)
    if not candles:
        return False
    closes = [c[4] for c in candles]
    ma20 = sma(closes, 20)
    ma50 = sma(closes, 50)
    if ma20 and ma50 and closes[-1] > ma20 > ma50:
        return True
    return False

# تنفيذ صفقة شراء
def enter_trade(symbol):
    if not check_entry_signal(symbol):
        return False
    price = exchange.fetch_ticker(symbol)['last']
    amount = round(TRADE_AMOUNT_USDT / price, 6)
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        positions = load_positions()
        positions[symbol] = {
            'entry_price': price,
            'amount': amount,
            'timestamp': time.time()
        }
        save_positions(positions)
        send_telegram(f"✅ شراء {symbol} بسعر {price:.2f} كمية {amount}")
        return True
    except Exception as e:
        send_telegram(f"❌ فشل في شراء {symbol}: {e}")
        return False

# فحص الصفقات المفتوحة وتطبيق وقف الخسارة / جني الأرباح

def check_positions():
    positions = load_positions()
    closed = []
    for symbol, data in positions.items():
        try:
            current_price = exchange.fetch_ticker(symbol)['last']
            entry = data['entry_price']
            change = ((current_price - entry) / entry) * 100

            if change >= TAKE_PROFIT_PERCENT:
                exchange.create_market_sell_order(symbol, data['amount'])
                send_telegram(f"💰 تم جني الأرباح في {symbol} عند {current_price:.2f} (+{change:.2f}%)")
                closed.append(symbol)

            elif change <= -STOP_LOSS_PERCENT:
                exchange.create_market_sell_order(symbol, data['amount'])
                send_telegram(f"⚠️ تم وقف الخسارة في {symbol} عند {current_price:.2f} ({change:.2f}%)")
                closed.append(symbol)
        except Exception as e:
            send_telegram(f"خطأ في تحديث {symbol}: {e}")

    for symbol in closed:
        positions.pop(symbol)
    save_positions(positions)

# ملخص الصفقات المفتوحة

def get_positions_summary():
    positions = load_positions()
    if not positions:
        return "📭 لا توجد صفقات حالياً."
    message = "📊 الصفقات الحالية:\n"
    for symbol, data in positions.items():
        current_price = exchange.fetch_ticker(symbol)['last']
        entry = data['entry_price']
        change = ((current_price - entry) / entry) * 100
        message += f"\n{symbol}: {change:.2f}% (شراء بسعر {entry:.2f}, الآن {current_price:.2f})"
    return message

# دالة رئيسية لفحص كل السوق

def scan_market():
    found = 0
    for symbol in TRADE_SYMBOLS:
        success = enter_trade(symbol)
        if success:
            found += 1
    return found
