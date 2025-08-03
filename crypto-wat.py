import time
import hmac
import hashlib
import requests
import base64
import json
from datetime import datetime
from config import *

# قائمة العملات التي سيتم فحصها (مثال لـ 50 عملة)
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "TRX-USDT", "LINK-USDT",
    "MATIC-USDT", "LTC-USDT", "NEAR-USDT", "UNI-USDT", "ATOM-USDT",
    "OP-USDT", "ETC-USDT", "XLM-USDT", "INJ-USDT", "RNDR-USDT",
    "SUI-USDT", "PEPE-USDT", "TIA-USDT", "SEI-USDT", "BCH-USDT",
    "CRO-USDT", "RUNE-USDT", "APT-USDT", "MKR-USDT", "FTM-USDT",
    "THETA-USDT", "AAVE-USDT", "GALA-USDT", "AR-USDT", "CRV-USDT",
    "KAVA-USDT", "GMX-USDT", "FET-USDT", "1INCH-USDT", "ENJ-USDT",
    "DYDX-USDT", "ZIL-USDT", "CELO-USDT", "ANKR-USDT", "YFI-USDT",
    "WAVES-USDT", "CHZ-USDT", "ALGO-USDT", "CAKE-USDT", "BAND-USDT"
]

BASE_URL = "https://www.okx.com"

# دالة إرسال رسالة إلى تيليجرام
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# دالة إنشاء التوقيع للتوثيق

def get_okx_signature(timestamp, method, request_path, body, secret_key):
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256)
    d = mac.digest()
    return base64.b64encode(d).decode()

# دالة إنشاء الهيدر للمصادقة

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

# دالة جلب الرصيد

def get_usdt_balance():
    endpoint = "/api/v5/account/balance?ccy=USDT"
    url = BASE_URL + endpoint
    try:
        res = requests.get(url, headers=get_okx_headers(endpoint))
        res.raise_for_status()
        data = res.json()
        balance = float(data["data"][0]["details"][0]["cashBal"])
        print(f"✅ الرصيد: {balance} USDT")
        return balance
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب الرصيد: {e}")
        return 0.0

# دالة جلب بيانات الشموع لكل عملة

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

# حساب متوسط متحرك بسيط

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

# دالة تنفيذ صفقة شراء

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
        print(f"✅ تم شراء {symbol}")
        send_telegram(f"✅ تم شراء {symbol} بقيمة {size} USDT")
    except Exception as e:
        send_telegram(f"❌ فشل شراء {symbol}: {e}")

# دالة رئيسية للبحث عن فرص تداول

def scan_and_trade():
    balance = get_usdt_balance()
    if balance < 10:
        send_telegram(f"⚠️ الرصيد غير كافٍ: {balance:.2f} USDT")
        return

    trade_size = balance * 0.30  # 30٪ من الرصيد
    opportunities = []

    for symbol in SYMBOLS:
        closes = get_candles(symbol)
        if len(closes) < 50:
            continue
        sma9 = sma(closes, 9)
        sma21 = sma(closes, 21)

        if sma9 and sma21 and closes[-1] > sma9 > sma21:
            opportunities.append(symbol)
            place_market_order(symbol, trade_size / closes[-1])
        time.sleep(0.2)

    if not opportunities:
        send_telegram("📉 لا توجد فرص تداول حالياً")
    else:
        send_telegram(f"🚀 فرص تداول مكتشفة:
" + "\n".join(opportunities))

# بدء البوت
if __name__ == "__main__":
    send_telegram(f"🤖 تم تشغيل البوت بنجاح! عدد العملات التي سيتم فحصها: {len(SYMBOLS)}")
    while True:
        scan_and_trade()
        time.sleep(60 * 60)  # كل ساعة
