import ccxt
import pandas as pd
import time
import requests

# مفاتيح OKX (لا تقم بنشرها أو مشاركتها بشكل عام)
API_KEY = "6e2d2b3f-636a-424a-a97e-5154e39e525a"
SECRET_KEY = "D4B9966385BEE5A7B7D8791BA5C0539F"
PASSPHRASE = "Ta123456&"

TELEGRAM_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
TELEGRAM_CHAT_ID = "658712542"

TRADING_AMOUNT_USDT = 15  # مبلغ الدخول لكل صفقة

exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram send error:", e)

def fetch_ohlcv(symbol, timeframe='1m', limit=50):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"Error fetching OHLCV: {e}")
        return None

def calculate_ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def check_trade_signal(symbol):
    df = fetch_ohlcv(symbol, timeframe='1m', limit=100)
    if df is None or len(df) < 50:
        return None
    
    ema9 = calculate_ema(df, 9)
    ema21 = calculate_ema(df, 21)
    ema50 = calculate_ema(df, 50)

    last_ema9 = ema9.iloc[-1]
    last_ema21 = ema21.iloc[-1]
    prev_ema9 = ema9.iloc[-2]
    prev_ema21 = ema21.iloc[-2]
    last_price = df['close'].iloc[-1]
    last_ema50 = ema50.iloc[-1]

    # شرط الدخول شراء
    if prev_ema9 <= prev_ema21 and last_ema9 > last_ema21 and last_price > last_ema50:
        return 'buy'
    # شرط الخروج (بيع) يعتمد على وقف الخسارة وجني الربح بشكل منفصل في الكود الأساسي
    return None

def place_order(symbol, side, amount):
    try:
        # OKX Spot requires symbol like 'BTC/USDT'
        order = exchange.create_market_order(symbol, side, amount)
        send_telegram_message(f"✅ تم تنفيذ أمر {side.upper()} على {symbol}، كمية: {amount}")
        return order
    except Exception as e:
        send_telegram_message(f"❌ خطأ في تنفيذ أمر {side} على {symbol}: {e}")
        return None

def get_balance(asset='USDT'):
    try:
        balances = exchange.fetch_balance()
        free_balance = balances.get(asset, {}).get('free', 0)
        return free_balance
    except Exception as e:
        print("Error fetching balance:", e)
        return 0

def main_loop(symbol='BTC/USDT'):
    position = None  # لتخزين حالة الصفقة الحالية
    
    while True:
        signal = check_trade_signal(symbol)
        balance = get_balance('USDT')
        
        if signal == 'buy' and position is None and balance >= TRADING_AMOUNT_USDT:
            price = exchange.fetch_ticker(symbol)['last']
            amount = TRADING_AMOUNT_USDT / price
            order = place_order(symbol, 'buy', amount)
            if order:
                entry_price = price
                stop_loss = entry_price * 0.99  # 1% وقف خسارة
                take_profit = entry_price * 4  # 4% جني ربح
                position = {
                    'amount': amount,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                send_telegram_message(f"📈 تم فتح صفقة شراء على {symbol} بسعر {entry_price:.2f}")
        
        if position is not None:
            current_price = exchange.fetch_ticker(symbol)['last']
            # تحقق وقف الخسارة
            if current_price <= position['stop_loss']:
                place_order(symbol, 'sell', position['amount'])
                send_telegram_message(f"❌ تم تفعيل وقف الخسارة عند {current_price:.2f} على {symbol}")
                position = None
            # تحقق جني الربح
            elif current_price >= position['take_profit']:
                place_order(symbol, 'sell', position['amount'])
                send_telegram_message(f"🎯 تم الوصول لهدف الربح عند {current_price:.2f} على {symbol}")
                position = None
        
        time.sleep(30)  # تفادي حظر الطلبات عبر انتظار 30 ثانية

if __name__ == "__main__":
    send_telegram_message("🤖 بدأ البوت في العمل!")
    main_loop("BTC/USDT")
