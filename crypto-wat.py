import asyncio
import random
import requests
from binance_api import get_price, get_klines, get_balance, place_order
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TRADE_AMOUNT_PERCENT, STOP_LOSS_PERCENT, TAKE_PROFIT_MIN, TAKE_PROFIT_MAX, CHECK_INTERVAL_MINUTES

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("Telegram Error:", response.status_code, response.text)
    except Exception as e:
        print("Telegram Exception:", e)

def calculate_signals(klines):
    closes = [float(k[4]) for k in klines]
    if len(closes) < 200:
        return False
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200
    return ma20 > ma50 > ma200

def read_symbols(filename="symbols.txt"):
    try:
        with open(filename) as f:
            return [line.strip().upper() for line in f if line.strip()]
    except:
        return []

async def trading_loop():
    while True:
        try:
            symbols = read_symbols()
            if not symbols:
                symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

            balance = get_balance("USDT")
            usdt_to_trade = balance * TRADE_AMOUNT_PERCENT
            send_telegram(f"📈 فحص السوق بدأ لـ {len(symbols)} عملة، الرصيد: {balance:.2f} USDT")

            for symbol in symbols:
                klines = get_klines(symbol)
                if not klines:
                    continue
                if calculate_signals(klines):
                    price = float(klines[-1][4])
                    qty = round(usdt_to_trade / price, 4)
                    order = place_order(symbol, "BUY", qty)
                    sl_price = price * (1 - STOP_LOSS_PERCENT / 100)
                    tp_price = price * (1 + random.uniform(TAKE_PROFIT_MIN, TAKE_PROFIT_MAX) / 100)
                    send_telegram(f"✅ إشارة شراء {symbol}\nالسعر: {price:.4f}\nالكمية: {qty}\n🚫 وقف خسارة: {sl_price:.4f}\n🎯 جني ربح: {tp_price:.4f}\n\n{order}")
                    await asyncio.sleep(1.5)

            send_telegram("✅ فحص السوق اكتمل، سيتم التكرار خلال 25 دقيقة.")
        except Exception as e:
            send_telegram(f"❌ حدث خطأ في الدورة: {e}")

        await asyncio.sleep(CHECK_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    asyncio.run(trading_loop())
