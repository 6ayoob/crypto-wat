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

def generate_signature(params, secret):
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def read_symbols(filename="coins.txt"):
    try:
        with open(filename, "r") as file:
            symbols = [line.strip() for line in file if line.strip()]
        return symbols
    except FileNotFoundError:
        print("coins.txt غير موجود، سيتم جلب الرموز من Bybit.")
        return []

def get_top_50_symbols():
    url = f"{BASE_URL}/v5/market/tickers?category=spot"
    try:
        response = requests.get(url)
        data = response.json()
        tickers = data.get("result", {}).get("list", [])
        sorted_tickers = sorted(tickers, key=lambda x: float(x["volume24h"]), reverse=True)
        return [t["symbol"] for t in sorted_tickers[:50] if "USDT" in t["symbol"]]
    except Exception as e:
        print("❌ Error fetching tickers:", e)
        return []

def get_klines(symbol, interval="60", limit=200):
    url = f"{BASE_URL}/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching klines for {symbol}:", response.status_code, response.text)
            return []
        data = response.json()
        klines = data.get("result", {}).get("list", [])
        if not klines:
            print(f"Warning: No klines data for {symbol}")
        return klines
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
    if not klines or len(klines) < 200:
        return False
    try:
        closes = [float(k[4]) for k in klines]
    except (IndexError, ValueError) as e:
        print("Error parsing klines:", e)
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
        coins = data.get("result", {}).get("list", [])
        usdt_balance = 0.0
        for coin_info in coins:
            if coin_info.get("coin") == "USDT":
                usdt_balance = float(coin_info.get("walletBalance", 0))
                break
        return usdt_balance
    except Exception as e:
        print("❌ Error getting balance:", e)
        return 0.0

async def trading_loop():
    while True:
        try:
            symbols = read_symbols()
            if not symbols:
                symbols = get_top_50_symbols()

            balance = get_balance()
            usdt_to_trade = 30  # شراء دائم بـ 30 دولار كما طلبت

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

                    sl_price = price * (1 - 0.02)  # وقف خسارة 2%
                    tp1_price = price * (1 + 0.05)  # جني ربح 5%
                    tp2_price = price * (1 + 0.10)  # جني ربح 10%

                    send_telegram(
                        f"✅ إشارة شراء {symbol}\n"
                        f"السعر: {price:.4f}\n"
                        f"الكمية: {qty}\n"
                        f"🚫 وقف خسارة: {sl_price:.4f}\n"
                        f"🎯 جني ربح 1: {tp1_price:.4f}\n"
                        f"🎯 جني ربح 2: {tp2_price:.4f}"
                    )
                    await asyncio.sleep(1.5)

            send_telegram("✅ فحص السوق اكتمل، سيتم التكرار خلال 25 دقيقة.")
        except Exception as e:
            send_telegram(f"❌ حدث خطأ في الدورة: {str(e)}")

        await asyncio.sleep(25 * 60)  # 25 دقيقة

if __name__ == "__main__":
    asyncio.run(trading_loop())
