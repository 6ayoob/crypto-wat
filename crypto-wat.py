import requests
import time
import hmac
import hashlib
import base64
import json
from datetime import datetime, timezone

# ===================== إعدادات OKX =====================
API_KEY = "ضع_مفتاحك_الجديد_هنا"
SECRET_KEY = "ضع_السر_الجديد_هنا"
PASSPHRASE = "ضع_كلمة_المرور_الجديدة"

# ===================== إعدادات Telegram =====================
TELEGRAM_TOKEN = "ضع_توكن_تيلجرام_الخاص_بك"
TELEGRAM_CHAT_ID = "ضع_ChatID_هنا"

# ===================== إعدادات التداول =====================
MAX_POSITIONS = 2
TRADE_PERCENT = 0.5  # 50%
TRADE_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT"]

# ============================================================
# 📌 إرسال رسالة إلى Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# ============================================================
# 📌 إنشاء التوقيع الصحيح OKX
def generate_signature(timestamp, method, request_path, body=""):
    if body is None:
        body = ""
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

# ============================================================
# 📌 إرسال طلب إلى OKX API
def okx_request(method, endpoint, body=None, params=None):
    url = f"https://www.okx.com{endpoint}"
    timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
    body_str = "" if method.upper() == "GET" else json.dumps(body) if body else ""
    sign = generate_signature(timestamp, method, endpoint, body_str)

    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

    try:
        response = requests.request(
            method, url, headers=headers,
            json=body if method.upper() == "POST" else None,
            params=params
        )
        data = response.json()
        if data.get("code") != "0":
            print(f"❌ OKX Error: {data}")
        return data
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

# ============================================================
# 📌 فحص الرصيد
def get_usdt_balance():
    result = okx_request("GET", "/api/v5/account/balance", params={"ccy": "USDT"})
    if result and result.get("data"):
        try:
            balance = float(result["data"][0]["details"][0]["availBal"])
            print(f"✅ الرصيد الحالي: {balance} USDT")
            return balance
        except:
            print("❌ فشل قراءة الرصيد")
    return 0

# ============================================================
# 📌 تنفيذ أمر شراء/بيع
def place_order(symbol, side, amount):
    body = {
        "instId": symbol,
        "tdMode": "cash",
        "side": side,
        "ordType": "market",
        "sz": str(amount)
    }
    result = okx_request("POST", "/api/v5/trade/order", body=body)
    if result and result.get("code") == "0":
        send_telegram_message(f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol} بنجاح!")
        return True
    else:
        send_telegram_message(f"❌ فشل تنفيذ أمر {side.upper()} لـ {symbol}: {result}")
        return False

# ============================================================
# 📌 جلب الصفقات المفتوحة
def get_open_positions():
    result = okx_request("GET", "/api/v5/account/positions", params={"instType": "SPOT"})
    if result and result.get("data"):
        return [p["instId"] for p in result["data"] if float(p.get("pos", 0)) > 0]
    return []

# ============================================================
# 📌 تنفيذ الإستراتيجية
def run_strategy():
    send_telegram_message("🤖 تم تشغيل بوت OKX بنجاح!")
    balance = get_usdt_balance()
    if balance < 10:
        send_telegram_message(f"⚠️ الرصيد غير كافٍ للتداول: {balance:.2f} USDT")
        return

    open_positions = get_open_positions()
    if len(open_positions) >= MAX_POSITIONS:
        send_telegram_message("⚠️ الحد الأقصى من الصفقات المفتوحة تم الوصول إليه.")
        return

    amount_to_trade = balance * TRADE_PERCENT
    trades_executed = 0

    for symbol in TRADE_SYMBOLS:
        if symbol in open_positions:
            continue
        if trades_executed >= (MAX_POSITIONS - len(open_positions)):
            break

        price_data = okx_request("GET", "/api/v5/market/ticker", params={"instId": symbol})
        if price_data and price_data.get("data"):
            last_price = float(price_data["data"][0]["last"])
            qty = round(amount_to_trade / last_price, 6)
            if place_order(symbol, "buy", qty):
                trades_executed += 1
                time.sleep(1)

# ============================================================
# 🚀 بدء التشغيل
if __name__ == "__main__":
    send_telegram_message("🚀 بدء التشغيل...")
    run_strategy()
