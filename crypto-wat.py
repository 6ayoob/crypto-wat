import requests
import time
import hmac
import hashlib
import json

# إعدادات OKX
API_KEY = "22a91dd1-4c62-4f0b-bd80-cda7abc31824"
SECRET_KEY = "3EBFAC5036B54B4EE77C369F2C5E66EE"
PASSPHRASE = "Ta123456&"  # تأكد من أنها مطابقة تمامًا

# إعدادات Telegram
TELEGRAM_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
TELEGRAM_CHAT_ID = "658712542"

# إعدادات التداول
MAX_POSITIONS = 2
TRADE_PERCENT = 0.5  # 50%

# قائمة 50 رمز تداول متنوعة عبر قطاعات مختلفة
TRADE_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "ADA-USDT",
    "XRP-USDT", "DOT-USDT", "DOGE-USDT", "LTC-USDT", "AVAX-USDT",
    "SHIB-USDT", "MATIC-USDT", "ATOM-USDT", "FTM-USDT", "ALGO-USDT",
    "NEAR-USDT", "LINK-USDT", "XLM-USDT", "VET-USDT", "ICP-USDT",
    "EOS-USDT", "TRX-USDT", "FLOW-USDT", "AXS-USDT", "GRT-USDT",
    "SAND-USDT", "MANA-USDT", "KSM-USDT", "CHZ-USDT", "KAVA-USDT",
    "ZIL-USDT", "RUNE-USDT", "HNT-USDT", "ENJ-USDT", "BAT-USDT",
    "QTUM-USDT", "DASH-USDT", "COMP-USDT", "SNX-USDT", "NEO-USDT",
    "STX-USDT", "XMR-USDT", "YFI-USDT", "CRV-USDT", "LUNA-USDT",
    "ZRX-USDT", "WAVES-USDT", "1INCH-USDT", "GALA-USDT", "AR-USDT"
]

# وظائف Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"❌ Telegram Error: {response.text}")
    except Exception as e:
        print(f"❌ Telegram Exception: {e}")

# توقيع الطلبات لـ OKX
def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    return hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).hexdigest()

# إرسال طلب إلى OKX
def okx_request(method, endpoint, body=None, params=None):
    url = f"https://www.okx.com{endpoint}"
    timestamp = str(time.time())
    body_str = json.dumps(body) if body else ""
    sign = generate_signature(timestamp, method, endpoint, body_str)

    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

    try:
        response = requests.request(method, url, headers=headers, json=body, params=params)
        data = response.json()
        if "code" in data and data["code"] != "0":
            print(f"❌ OKX Error: {data}")
        return data
    except Exception as e:
        print(f"❌ Exception in OKX request: {e}")
        return None

# إحضار الرصيد
def get_usdt_balance():
    result = okx_request("GET", "/api/v5/account/balance", params={"ccy": "USDT"})
    if result and "data" in result:
        balance = float(result["data"][0]["details"][0]["availBal"])
        print(f"✅ الرصيد الحالي: {balance} USDT")
        return balance
    return 0

# تنفيذ صفقة شراء
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
        send_telegram_message(f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol} بقيمة {amount} بنجاح.")
        return True
    else:
        send_telegram_message(f"❌ فشل تنفيذ الأمر: {result}")
        return False

# عدد الصفقات المفتوحة
def get_open_positions():
    result = okx_request("GET", "/api/v5/account/positions", params={"instType": "SPOT"})
    if result and "data" in result:
        open_symbols = [pos["instId"] for pos in result["data"] if float(pos.get("pos", 0)) > 0]
        print(f"⚠️ الصفقات المفتوحة حالياً: {open_symbols}")
        return open_symbols
    return []

# 🚀 تنفيذ الإستراتيجية مع فحص جميع العملات
def run_strategy():
    send_telegram_message("🤖 تم تشغيل البوت بنجاح!")

    balance = get_usdt_balance()
    if balance < 10:
        send_telegram_message(f"⚠️ الرصيد غير كافٍ: {balance:.2f} USDT")
        return

    open_positions = get_open_positions()
    if len(open_positions) >= MAX_POSITIONS:
        send_telegram_message(f"⚠️ تم الوصول للحد الأقصى من الصفقات المفتوحة: {len(open_positions)}")
        return

    amount_to_trade = (balance * TRADE_PERCENT)

    # نفحص العملات واحدة تلو الأخرى ونشتري أول عملتين (حتى حد الصفقات المفتوحة)
    trades_executed = 0
    for symbol in TRADE_SYMBOLS:
        if symbol in open_positions:
            continue  # نتخطى العملات المفتوحة

        if trades_executed >= (MAX_POSITIONS - len(open_positions)):
            break  # وصلنا للحد الأقصى

        # جلب آخر سعر العملة
        price_data = okx_request("GET", "/api/v5/market/ticker", params={"instId": symbol})
        if price_data and "data" in price_data and len(price_data["data"]) > 0:
            last_price = float(price_data["data"][0]["last"])
            quantity = round(amount_to_trade / last_price, 6)
            success = place_order(symbol, "buy", quantity)
            if success:
                trades_executed += 1
                time.sleep(1)  # تأخير بسيط لتفادي حظر API
        else:
            print(f"❌ تعذر الحصول على سعر العملة: {symbol}")

# بدء البوت
if __name__ == "__main__":
    run_strategy()
