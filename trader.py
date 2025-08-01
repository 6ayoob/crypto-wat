import ccxt
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# إعدادات عامة
symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]  # يمكنك استبدالها بالقائمة التي أرسلتها
investment_amount = 30  # مبلغ الشراء بالدولار

# قاعدة بيانات بسيطة لتتبع الصفقات
TRADES_FILE = 'trades.json'

def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_trades(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)

def calculate_ema(data, period):
    prices = [x[4] for x in data]  # Close prices
    ema = []
    k = 2 / (period + 1)
    for i in range(len(prices)):
        if i < period - 1:
            ema.append(None)
        elif i == period - 1:
            sma = sum(prices[:period]) / period
            ema.append(sma)
        else:
            ema.append(prices[i] * k + ema[i - 1] * (1 - k))
    return ema

def should_buy(symbol):
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe='5m', limit=50)
        ema9 = calculate_ema(ohlcv, 9)
        ema21 = calculate_ema(ohlcv, 21)
        if ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]:
            return True
    except Exception as e:
        print(f"❌ Error fetching or calculating EMA for {symbol}: {e}")
    return False

def place_order(symbol):
    price = binance.fetch_ticker(symbol)["last"]
    amount = investment_amount / price
    order = binance.create_market_buy_order(symbol, amount)
    print(f"✅ Bought {symbol} for ${investment_amount:.2f}")
    return price

def monitor_trades(trades):
    for symbol in list(trades.keys()):
        entry = trades[symbol]['entry_price']
        qty = trades[symbol]['quantity']
        ticker = binance.fetch_ticker(symbol)
        current = ticker['last']
        change = (current - entry) / entry * 100

        if change >= 10:
            binance.create_market_sell_order(symbol, qty)
            print(f"🎯 Sold {symbol} at +10% profit.")
            del trades[symbol]
        elif change >= 5 and not trades[symbol].get("half_sold"):
            binance.create_market_sell_order(symbol, qty / 2)
            trades[symbol]["half_sold"] = True
            print(f"💰 Took 50% profit from {symbol} at +5%")
        elif change <= -2:
            binance.create_market_sell_order(symbol, qty)
            print(f"🛑 Sold {symbol} at -2% loss.")
            del trades[symbol]
    return trades

def run():
    trades = load_trades()
    for symbol in symbols:
        if symbol in trades:
            continue
        if should_buy(symbol):
            entry_price = place_order(symbol)
            qty = investment_amount / entry_price
            trades[symbol] = {
                "entry_price": entry_price,
                "quantity": qty
            }
            time.sleep(1)
    trades = monitor_trades(trades)
    save_trades(trades)

if __name__ == "__main__":
    while True:
        print("🔁 Running strategy loop...")
        try:
            run()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        time.sleep(60 * 5)  # كل 5 دقائق
