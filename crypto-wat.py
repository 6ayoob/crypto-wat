import time
import hmac
import hashlib
import requests
import base64
import json
from datetime import datetime
from config import *

BASE_URL = "https://www.okx.com"

# قائمة العملات للتداول (50 عملة)
SYMBOLS = TRADING_SYMBOLS  # استيراد من config.py

# إرسال رسالة تيليجرام مع معالجة الأخطاء
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# توقيع طلبات OKX
def get_okx_signature(timestamp, method, request_path, body, secret_key):
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256)
    d = mac.digest()
    return base64.b64encode(d).decode()

# تجهيز الهيدر لكل طلب
def get_okx_headers(endpoint, method="GET", body=""):
    timestamp = datetime.utcnow().isoformat("T", "milliseconds") + "Z"
    sign = get_okx_signature(timestamp, method, endpoint, body, OKX_SECRET_KEY)
    return {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "Content-Type": "application/json"
    }

# جلب رصيد USDT مع طباعة الرد الكامل للتشخيص
def get_usdt_balance():
    endpoint = "/api/v5/account/balance?ccy=USDT"
    url = BASE_URL + endpoint
    try:
        res = requests.get(url, headers=get_okx_headers(endpoint))
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.text}")
        res.raise_for_status()
        data = res.json()
        balance = float(data["data"][0]["details"][0]["cashBal"])
        print(f"✅ الرصيد: {balance} USDT")
        return balance
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب الرصيد: {e}")
        return 0.0

# جلب بيانات الشموع (أسعار الإغلاق)
def get_candles(symbol, limit=50):
    endpoint = f"/api/v5/market/candles?instId={symbol}&bar=1h&limit={limit}"
    url = BASE_URL + endpoint
    try:
        res = requests.get(url)
        res.raise_for_status()
        candles = res.json()["data"]
        closes = [float(c[4]) for c in reversed(candles)]
        return closes
    except Exception as e:
        print(f"⚠️ فشل في جلب بيانات {symbol}: {e}")
        return []

# حساب المتوسط المتحرك البسيط (SMA)
def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

# تنفيذ أمر شراء بالسوق
def place_market_order(symbol, size):
    endpoint = "/api/v5/trade/order"
    url = BASE_URL + endpoint
    body = json.dumps({
        "instId": symbol,
        "tdMode": "cash",
        "side": "buy",
        "ordType": "market",
        "sz": str(size)
    })
    try:
        res = requests.post(url, headers=get_okx_headers(endpoint, "POST", body), data=body)
        res.raise_for_status()
        print(f"✅ تم شراء {symbol} بحجم {size} USDT")
        send_telegram(f"✅ تم شراء {symbol} بقيمة {size} USDT")
    except Exception as e:
        send_telegram(f"❌ فشل شراء {symbol}: {e}")

# البحث والتنفيذ بناءً على SMA 9 و21 وتجاوز السعر الحالي المتوسطين
def scan_and_trade():
    balance = get_usdt_balance()
    if balance < 10:
        send_telegram(f"⚠️ الرصيد غير كافٍ: {balance:.2f} USDT")
        return

    trade_size = balance * 0.30  # 30% من الرصيد للتداول

    opportunities = []

    for symbol in SYMBOLS:
        closes = get_candles(symbol)
        if len(closes) < 50:
            continue
        sma9 = sma(closes, 9)
        sma21 = sma(closes, 21)

        # شرط الدخول: السعر الحالي أكبر من المتوسطين sma9 > sma21
        if sma9 and sma21 and closes[-1] > sma9 > sma21:
            opportunities.append(symbol)
            size_in_coins = trade_size / closes[-1]
            place_market_order(symbol, size_in_coins)
        time.sleep(0.2)  # لتفادي حظر API

    if not opportunities:
        send_telegram("📉 لا توجد فرص تداول حالياً")
    else:
        msg = "🚀 فرص تداول مكتشفة:\n" + "\n".join(opportunities)
        send_telegram(msg)

# نقطة البداية
if __name__ == "__main__":
    send_telegram(f"🤖 تم تشغيل البوت بنجاح! عدد العملات التي سيتم فحصها: {len(SYMBOLS)}")
    while True:
        try:
            scan_and_trade()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(3600)  # كل ساعة
