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
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("Telegram Error:", response.status_code, response.text)
    except Exception as e:
        print("Telegram Exception:", e)

def get_top_50_symbols():
    url = f"{BASE_URL}/v5/market/tickers?category=spot"
    try:
        response = requests.get(url)
        data = response.json()
        print("🧪 DEBUG - Tick Data:", data)  # ✅ ستطبع البيانات القادمة من Bybit

        tickers = data.get("result", {}).get("list", [])
        sorted_tickers = sorted(tickers, key=lambda x: float(x["volume24h"]), reverse=True)
        return [t["symbol"] for t in sorted_tickers[:TOP_COINS] if "USDT" in t["symbol"]]

    except Exception as e:
        print("❌ Error fetching tickers:", e)
        return []

        data = response.json()
        tickers = data.get("result", {}).get("list", [])
        sorted_tickers = sorted(tickers, key=lambda x: float(x["volume24h"]), reverse=True)
        return [t["symbol"] for t in sorted_tickers[:TOP_COINS] if "USDT" in t["symbol"]]
    except Exception as e:
        print("Exception in get_top_50_symbols:", e)
        return []

def get_klines(symbol, interval="60", limit=200):
    url = f"{BASE_URL}/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching klines for {symbol}:", response.status_code, response.text)
            return []
        return response.json().get("result", {}).get("list", [])
    except Exception as e:
        print(f"Exception in get_klines for {symbol}:", e)
        return []

def get_price(symbol):
    url = f"{BASE_URL}/v2/public/tickers?symbol={symbol}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching price for {symbol}:", response.status_code, response.text)
            return 0.0
        return float(response.json()["result"][0]["last_price"])
    except Exception as e:
        print(f"Exception in get_price for {symbol}:", e)
        return 0.0

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
    try:
        response = requests.post(url, data=params)
        if response.status_code != 200:
            print(f"Order error for {symbol}:", response.status_code, response.text)
        return response.json()
    except Exception as e:
        print(f"Exception in place_order for {symbol}:", e)
        return {}

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

    try:
        response = requests.get(url, params=params)
        data = response.json()
        print("💰 DEBUG - Balance Response:", data)  # ✅ ستطبع البيانات الخاصة برصيد الحساب

        balance = float(data["result"]["list"][0]["coin"][0]["walletBalance"])
        return balance

    except Exception as e:
        print("❌ Error getting balance:", e)
        return 0.0


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
            send_telegram(f"❌ حدث خطأ في الدورة: {str(e)}")

        await asyncio.sleep(CHECK_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    asyncio.run(trading_loop())
