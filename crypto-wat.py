import time
import requests
import hmac
import hashlib
import json
from datetime import datetime
from config import *

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except:
        pass

def get_klines(symbol):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "15m", "limit": 50}
    try:
        response = requests.get(url, params=params)
        return response.json()
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return []

def calculate_ema(prices, length):
    weights = [2 / (length + 1) * (1 - 2 / (length + 1)) ** i for i in range(len(prices))]
    weights.reverse()
    ema = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
    return ema

def place_order(symbol, side, quantity):
    timestamp = int(time.time() * 1000)
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": timestamp
    }
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(BINANCE_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_API}
    url = f"https://api.binance.com/api/v3/order?{query_string}&signature={signature}"
    response = requests.post(url, headers=headers)
    return response.json()

def get_balance(asset="USDT"):
    url = f"https://api.binance.com/api/v3/account"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(BINANCE_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_API}
    url = f"{url}?{query_string}&signature={signature}"
    response = requests.get(url, headers=headers)
    balances = response.json().get("balances", [])
    for b in balances:
        if b["asset"] == asset:
            return float(b["free"])
    return 0.0

def trade_logic():
    with open("coins.txt", "r") as f:
        symbols = [line.strip().upper() for line in f.readlines()]
    usdt_balance = get_balance()
    send_telegram(f"📈 فحص السوق بدأ لـ {len(symbols)} عملة، الرصيد: {usdt_balance:.2f} USDT")

    for symbol in symbols:
        try:
            symbol_pair = symbol + "USDT"
            klines = get_klines(symbol_pair)
            closes = [float(k[4]) for k in klines]
            ema9 = calculate_ema(closes[-20:], 9)
            ema21 = calculate_ema(closes[-20:], 21)

            if ema9 > ema21:
                price = closes[-1]
                qty = round(30 / price, 5)
                order = place_order(symbol_pair, "BUY", qty)
                send_telegram(f"✅ تم شراء {symbol} بسعر {price:.4f}, كمية: {qty}")

                target1 = price * 1.05
                target2 = price * 1.10
                stop_loss = price * 0.98
                holding = True
                while holding:
                    current_price = float(get_klines(symbol_pair)[-1][4])
                    if current_price >= target1:
                        place_order(symbol_pair, "SELL", round(qty / 2, 5))
                        send_telegram(f"🎯 بيع 50% من {symbol} عند +5% بسعر {current_price:.4f}")
                        target1 = 999999  # تعطيل البيع الأول
                    if current_price >= target2:
                        place_order(symbol_pair, "SELL", round(qty / 2, 5))
                        send_telegram(f"🏁 بيع 100% من {symbol} عند +10% بسعر {current_price:.4f}")
                        holding = False
                    if current_price <= stop_loss:
                        place_order(symbol_pair, "SELL", qty)
                        send_telegram(f"🚨 وقف خسارة {symbol} عند {current_price:.4f}")
                        holding = False
                    time.sleep(30)
        except Exception as e:
            send_telegram(f"❌ خطأ في {symbol}: {str(e)}")

    send_telegram("✅ فحص السوق اكتمل، سيتم التكرار بعد 5 دقائق.")

while True:
    try:
        trade_logic()
    except Exception as e:
        send_telegram(f"❌ خطأ عام: {e}")
    time.sleep(300)  # كل 5 دقائق
