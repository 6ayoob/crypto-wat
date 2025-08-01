import asyncio
import random
import time
import hmac
import hashlib
import requests
from config import *

BASE_URL = "https://api.bybit.com"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

def get_top_50_symbols():
    url = f"{BASE_URL}/v5/market/tickers?category=spot"
    data = requests.get(url).json()
    tickers = data.get("result", {}).get("list", [])
    sorted_tickers = sorted(tickers, key=lambda x: float(x["volume24h"]), reverse=True)
    return [t["symbol"] for t in sorted_tickers[:TOP_COINS] if "USDT" in t["symbol"]]

def get_klines(symbol, interval="60", limit=200):
    url = f"{BASE_URL}/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}"
    return requests.get(url).json().get("result", {}).get("list", [])

def get_price(symbol):
    url = f"{BASE_URL}/v2/public/tickers?symbol={symbol}"
    return float(requests.get(url).json()["result"][0]["last_price"])

def generate_signature(params, secret):
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def place_order(symbol, side, qty):
    url = f"{BASE_URL}/v5/order/create"
    timestamp = str(int(time.time() * 1000))
    params = {
        "api_key": BYBIT_API_KEY,
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": qty,
        "category": "spot"
    }
    params["sign"] = generate_signature(params, BYBIT_API_SECRET)
    return requests.post(url, data=params).json()

def calculate_signals(klines):
    closes = [float(k[4]) for k in klines]
    if len(closes) < 200:
        return False
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200
    return ma20 > ma50 > ma200

def get_balance():
    url = f"{BASE_URL}/v5/account/wallet-balance?accountType=SPOT"
    timestamp = str(int(time.time() * 1000))
    params = {
        "api_key": BYBIT_API_KEY,
        "timestamp": timestamp,
    }
    params["sign"] = generate_signature(params, BYBIT_API_SECRET)
    response = requests.get(url, params=params).json()
    balance = float(response["result"]["list"][0]["coin"][0]["walletBalance"])
    return balance

async def trading_loop():
    while True:
        try:
            symbols = get_top_50_symbols()
            balance = get_balance()
            usdt_to_trade = balance * TRADE_AMOUNT_PERCENT
            send_telegram(f"📈 فحص السوق بدأ لـ {len(symbols)} عملة، الرصيد: {balance:.2f} USDT")

            for symbol in symbols:
                klines = get_klines(symbol)
                if not klines:
                    continue
                signal = calculate_signals(klines)
                if signal:
                    price = float(klines[-1][4])
                    qty = round(usdt_to_trade / price, 4)
                    place_order(symbol, "Buy", qty)
                    sl_price = price * (1 - STOP_LOSS_PERCENT / 100)
                    tp_price = price * (1 + random.uniform(TAKE_PROFIT_MIN, TAKE_PROFIT_MAX) / 100)
                    send_telegram(f"✅ إشارة شراء {symbol}\nالسعر: {price:.4f}\nالكمية: {qty}\n🚫 وقف خسارة: {sl_price:.4f}\n🎯 جني ربح: {tp_price:.4f}")
                    await asyncio.sleep(1.5)

            send_telegram("✅ فحص السوق اكتمل، سيتم التكرار خلال 25 دقيقة.")
        except Exception as e:
            send_telegram(f"❌ حدث خطأ: {str(e)}")

        await asyncio.sleep(CHECK_INTERVAL_MINUTES * 60)
