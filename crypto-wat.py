import time
import hmac
import hashlib
import requests
import base64
import json
from datetime import datetime
from config import *

BASE_URL = "https://www.okx.com"

SYMBOLS = TRADING_SYMBOLS  # استيراد من config.py

open_trades = []

# إرسال رسالة تيليجرام مع معالجة الأخطاء
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# توقيع طلبات OKX مع معالجة body للطلبات GET
def get_okx_signature(timestamp, method, request_path, body, secret_key):
    if method == "GET" or not body:
        body = ""
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
        send_telegram(f"❌ خطأ في جلب الرصيد: {e}\nResponse: {res.text if 'res' in locals() else 'No response'}")
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

# تنفيذ أمر شراء/بيع بالسوق مع طباعة الرد للتشخيص
def place_market_order(symbol, side, size):
    endpoint = "/api/v5/trade/order"
    url = BASE_URL + endpoint
    body = json.dumps({
        "instId": symbol,
        "tdMode": "cash",
        "side": side,
        "ordType": "market",
        "sz": str(size)
    })
    try:
        res = requests.post(url, headers=get_okx_headers(endpoint, "POST", body), data=body)
        print(f"Order Status Code: {res.status_code}")
        print(f"Order Response: {res.text}")
        res.raise_for_status()
        print(f"✅ تم تنفيذ أمر {side} على {symbol} بحجم {size} USDT")
        send_telegram(f"✅ تم تنفيذ أمر {side} على {symbol} بحجم {size} USDT")
        return res.json()
    except Exception as e:
        send_telegram(f"❌ فشل تنفيذ أمر {side} على {symbol}: {e}\nResponse: {res.text if 'res' in locals() else 'No response'}")
        return None

# منطق التداول: فتح صفقات جديدة بناءً على SMA مع الحد الأقصى للصفقات المفتوحة
def trade_logic():
    global open_trades
    balance = get_usdt_balance()
    if balance < 10:
        send_telegram(f"⚠️ الرصيد غير كافٍ: {balance:.2f} USDT")
        return

    max_open_trades = 2  # الحد الأقصى لعدد الصفقات المفتوحة
    if len(open_trades) >= max_open_trades:
        print(f"🛑 تم الوصول للحد الأقصى للصفقات المفتوحة ({max_open_trades})، لا يتم فتح صفقات جديدة.")
        return

    trade_size = balance * 0.50  # 50% من الرصيد لكل صفقة

    for symbol in SYMBOLS:
        if any(t["symbol"] == symbol for t in open_trades):
            continue

        if len(open_trades) >= max_open_trades:
            break

        closes = get_candles(symbol)
        if len(closes) < 21:
            continue
        sma9 = sma(closes, 9)
        sma21 = sma(closes, 21)

        if sma9 and sma21 and closes[-1] > sma9 > sma21:
            price = closes[-1]
            qty = round(trade_size / price, 4)
            result = place_market_order(symbol, "buy", qty)
            if result and result.get("code") == "0":
                open_trades.append({
                    "symbol": symbol,
                    "qty": qty,
                    "entry_price": price,
                    "target1": price * 1.05,
                    "target2": price * 1.10,
                    "stop_loss": price * 0.98,
                    "sold_target1": False
                })
                send_telegram(f"✅ تم شراء {symbol} بسعر {price:.4f} كمية: {qty}")
        time.sleep(0.2)

# متابعة الصفقات المفتوحة وتطبيق استراتيجيات الخروج (أهداف الربح ووقف الخسارة)
def follow_trades():
    global open_trades
    updated = []
    for t in open_trades:
        closes = get_candles(t["symbol"])
        if not closes:
            updated.append(t)
            continue
        current_price = closes[-1]

        # بيع نصف الكمية عند هدف الربح الأول (5%)
        if not t["sold_target1"] and current_price >= t["target1"]:
            half_qty = round(t["qty"] / 2, 4)
            res = place_market_order(t["symbol"], "sell", half_qty)
            if res and res.get("code") == "0":
                send_telegram(f"🎯 بيع 50% من {t['symbol']} عند +5% بسعر {current_price:.4f}")
                t["sold_target1"] = True
                t["qty"] -= half_qty
            updated.append(t)

        # بيع الباقي عند هدف الربح الثاني (10%)
        elif current_price >= t["target2"]:
            res = place_market_order(t["symbol"], "sell", t["qty"])
            if res and res.get("code") == "0":
                send_telegram(f"🏁 بيع الباقي من {t['symbol']} عند +10% بسعر {current_price:.4f}")
            # لا نضيفها للتحديث لأنها بيعت بالكامل

        # وقف الخسارة عند خسارة 2%
        elif current_price <= t["stop_loss"]:
            res = place_market_order(t["symbol"], "sell", t["qty"])
            if res and res.get("code") == "0":
                send_telegram(f"🚨 وقف خسارة {t['symbol']} عند {current_price:.4f}")
            # لا نضيفها للتحديث لأنها بيعت بالكامل

        else:
            updated.append(t)

    open_trades = updated

if __name__ == "__main__":
    send_telegram(f"🤖 تم تشغيل البوت بنجاح! عدد العملات التي سيتم فحصها: {len(SYMBOLS)}")
    while True:
        try:
            trade_logic()
            follow_trades()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(300)  # كل 5 دقائق
